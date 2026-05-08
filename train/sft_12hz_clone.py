"""SFT Qwen3-TTS-12Hz-1.7B-Base in voice-clone mode.

Differences from sft_12hz.py:
- Loads Base model (which has self.speaker_encoder)
- Computes speaker_embedding from each example's ref_mels via speaker_encoder
- Fills input_codec_embedding[:, 6, :] with speaker_embedding
  (VoiceDesign skips this slot; Base uses it for speaker conditioning)
- Saves checkpoint as tts_model_type='base' and keeps speaker_encoder weights

For self-cloning training, ref_audio = same as audio (the clip is its own reference).
The model learns to clone a speaker (extracted from the clip) and faithfully
generate the same text in the same voice. At inference we swap in any
LibriVox-quality reference and the model clones that voice instead.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \\
        sft_12hz_clone.py \\
            --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
            --output_model_path /workspace/sft_clone_out \\
            --train_jsonl /workspace/data/train_clone_codes.jsonl \\
            --batch_size 1 --lr 2e-6 --num_epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# Reuse the existing TTSDataset (it already loads ref_mel per example)
sys.path.insert(0, "/workspace/qwen3tts-repo/finetuning")
from dataset import TTSDataset


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="sft_clone_out")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="vocence_libri_clone")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    from huggingface_hub import snapshot_download as _snapshot_download
    if "/" in args.init_model_path and not os.path.isdir(args.init_model_path):
        MODEL_PATH = _snapshot_download(args.init_model_path)
    else:
        MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Sanity check — Base must have speaker_encoder
    if qwen3tts.model.tts_model_type != "base":
        raise SystemExit(
            f"expected tts_model_type='base', got '{qwen3tts.model.tts_model_type}'. "
            "Use Qwen3-TTS-12Hz-1.7B-Base, not VoiceDesign or CustomVoice."
        )

    train_data = [json.loads(line) for line in open(args.train_jsonl)]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    unwrap_model = accelerator.unwrap_model(model)
    model.train()

    speaker_encoder = unwrap_model.speaker_encoder

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                # === KEY DIFFERENCE FROM VoiceDesign SFT ===
                # Compute speaker embedding from ref mels and inject into slot 6.
                # speaker_encoder returns (speaker_emb, ...); we take [0].
                # Use unwrap_model for .device/.dtype (DDP-wrapped model lacks these).
                speaker_embedding = speaker_encoder(
                    ref_mels.to(unwrap_model.device).to(unwrap_model.dtype)
                )[0]

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = unwrap_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = unwrap_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                # Fill the speaker slot at codec position 6 — Base model condition path.
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = unwrap_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = unwrap_model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = unwrap_model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            # Keep tts_model_type='base' so generate_voice_clone works
            config_dict["tts_model_type"] = "base"
            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped.state_dict().items()}
            # KEEP speaker_encoder weights for Base model — required for inference
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


if __name__ == "__main__":
    train()
