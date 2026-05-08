"""DPO training for Qwen3-TTS-VoiceDesign (12 Hz codec).

Implements the Koel-TTS recipe (NVIDIA, ICML 2025):
    - Both chosen and rejected come from the policy's own distribution
      (NEVER use the source recording as 'chosen' — Koel-TTS proved this hurts)
    - β = 0.01 (low; speech tokens need it)
    - lr = 2e-7 (DPO needs gentle steps)
    - Standard DPO loss

Loss:
    L = -log σ( β · ( log π(yc|x) − log π(yr|x) − log π_ref(yc|x) + log π_ref(yr|x) ) )

Where logits are taken from the *talker*'s codec_0 channel — the auxiliary
sub-talker loss used by SFT is dropped to keep DPO clean.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
        dpo_12hz.py \
            --init_model_path /workspace/sft_out_rft/avg_last2 \
            --output_model_path /workspace/dpo_out \
            --train_jsonl /workspace/data/dpo_train_v3.jsonl \
            --batch_size 1 --beta 0.01 --lr 2e-7 --num_epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
import librosa
import numpy as np


# --------------------------------------------------------------------------
# Dataset: each row contains BOTH chosen_audio_codes and rejected_audio_codes.
# We reuse the SFT collate logic for each side (chosen, rejected) so the
# model sees the exact same input format as during SFT.
# --------------------------------------------------------------------------

class DPODataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig):
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_texts(self, text):
        out = self.processor(text=text, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    @torch.inference_mode()
    def _extract_mels(self, audio: np.ndarray, sr: int):
        assert sr == 24000, f"ref_audio must be 24 kHz; got {sr}"
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_texts(text)

        chosen_codes = torch.tensor(item["chosen_audio_codes"], dtype=torch.long)
        rejected_codes = torch.tensor(item["rejected_audio_codes"], dtype=torch.long)

        ref_path = item["ref_audio"]
        wav, sr = librosa.load(ref_path, sr=None, mono=True)
        ref_mel = self._extract_mels(wav.astype(np.float32), int(sr))

        return {
            "text_ids": text_ids[:, :-5],     # (1, t_text)
            "chosen_codes": chosen_codes,      # (t_codec_chosen, 16)
            "rejected_codes": rejected_codes,  # (t_codec_rejected, 16)
            "ref_mel": ref_mel,
        }


def _build_one_side_batch(batch, codec_field: str, config: Qwen3TTSConfig):
    """Pack a batch for ONE side (chosen or rejected) using the SFT layout.

    Returns the dict with input_ids, codec_ids, masks, codec_0_labels — same
    keys as sft_12hz.py's collate_fn, so we can reuse the forward path.
    """
    item_lengths = [
        b["text_ids"].shape[1] + b[codec_field].shape[0] for b in batch
    ]
    max_len = max(item_lengths) + 8
    bsz = len(batch)

    input_ids = torch.zeros((bsz, max_len, 2), dtype=torch.long)
    codec_ids = torch.zeros((bsz, max_len, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    codec_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    codec_0_labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, data in enumerate(batch):
        text_ids = data["text_ids"]
        audio_codecs = data[codec_field]
        audio_codec_0 = audio_codecs[:, 0]

        t_text = text_ids.shape[1]
        t_codec = audio_codec_0.shape[0]

        # text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8:8 + t_text - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + t_text - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + t_text - 2:8 + t_text + t_codec, 0] = config.tts_pad_token_id
        text_embedding_mask[i, :8 + t_text + t_codec] = True

        # codec channel
        input_ids[i, 3:8, 1] = torch.tensor([
            config.talker_config.codec_nothink_id,
            config.talker_config.codec_think_bos_id,
            config.talker_config.codec_think_eos_id,
            0,
            config.talker_config.codec_pad_id,
        ])
        input_ids[i, 8:8 + t_text - 3, 1] = config.talker_config.codec_pad_id
        input_ids[i, 8 + t_text - 3, 1] = config.talker_config.codec_pad_id
        input_ids[i, 8 + t_text - 2, 1] = config.talker_config.codec_bos_id
        input_ids[i, 8 + t_text - 1:8 + t_text - 1 + t_codec, 1] = audio_codec_0
        input_ids[i, 8 + t_text - 1 + t_codec, 1] = config.talker_config.codec_eos_token_id

        codec_0_labels[i, 8 + t_text - 1:8 + t_text - 1 + t_codec] = audio_codec_0
        codec_0_labels[i, 8 + t_text - 1 + t_codec] = config.talker_config.codec_eos_token_id

        codec_ids[i, 8 + t_text - 1:8 + t_text - 1 + t_codec, :] = audio_codecs

        codec_embedding_mask[i, 3:8 + t_text + t_codec] = True
        codec_embedding_mask[i, 6] = False

        codec_mask[i, 8 + t_text - 1:8 + t_text - 1 + t_codec] = True
        attention_mask[i, :8 + t_text + t_codec] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }


def make_collate(config):
    def collate(batch):
        chosen = _build_one_side_batch(batch, "chosen_codes", config)
        rejected = _build_one_side_batch(batch, "rejected_codes", config)
        ref_mels = torch.cat([b["ref_mel"] for b in batch], dim=0)
        return {"chosen": chosen, "rejected": rejected, "ref_mels": ref_mels}
    return collate


# --------------------------------------------------------------------------
# Forward + log-prob extraction (mirrors sft_12hz.py forward path)
# --------------------------------------------------------------------------

def _seq_log_prob(model, batch, device):
    """Sum of log P(codec_0[t] | context) across the codec sequence.

    Returns (B,) tensor — total per-sample log-prob of the chosen sequence.
    Mirrors sft_12hz.py's forward but extracts logits instead of CE loss.
    """
    input_ids = batch["input_ids"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    text_embedding_mask = batch["text_embedding_mask"].to(device)
    codec_embedding_mask = batch["codec_embedding_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    codec_0_labels = batch["codec_0_labels"].to(device)
    codec_mask = batch["codec_mask"].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    text_emb = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_embeddings = text_emb + codec_emb

    for i in range(1, 16):
        ce = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        ce = ce * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + ce

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,                        # we'll compute log-probs ourselves
        output_hidden_states=False,
    )
    logits = outputs.logits  # (B, T, V)
    labels = codec_0_labels[:, 1:]  # (B, T)

    valid = (labels != -100)
    safe_labels = labels.clamp(min=0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    selected = selected * valid
    return selected.sum(dim=1)  # (B,)


def _dpo_loss(logp_p_chosen, logp_p_rejected, logp_r_chosen, logp_r_rejected, beta: float):
    policy_diff = logp_p_chosen - logp_p_rejected
    ref_diff = logp_r_chosen - logp_r_rejected
    margin = policy_diff - ref_diff
    loss = -F.logsigmoid(beta * margin).mean()
    # for monitoring
    chosen_reward = beta * (logp_p_chosen - logp_r_chosen).mean().item()
    rejected_reward = beta * (logp_p_rejected - logp_r_rejected).mean().item()
    accuracy = (margin > 0).float().mean().item()
    return loss, {"acc": accuracy, "rc": chosen_reward, "rr": rejected_reward}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True,
                        help="path to the SFT checkpoint to start from (and freeze for ref)")
    parser.add_argument("--output_model_path", type=str, default="dpo_out")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--speaker_name", type=str, default="vocence_libri_dpo")
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")

    from huggingface_hub import snapshot_download as _snapshot_download
    init_path = args.init_model_path
    if "/" in init_path and not os.path.isdir(init_path):
        init_path = _snapshot_download(init_path)

    # Policy (trainable)
    policy = Qwen3TTSModel.from_pretrained(
        init_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Reference (frozen) — load the SAME weights
    reference = Qwen3TTSModel.from_pretrained(
        init_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    for p in reference.model.parameters():
        p.requires_grad = False
    reference.model.eval()

    config = AutoConfig.from_pretrained(init_path)

    train_data = [json.loads(l) for l in open(args.train_jsonl)]
    dataset = DPODataset(train_data, policy.processor, config)
    collate = make_collate(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate)

    optimizer = AdamW(policy.model.parameters(), lr=args.lr, weight_decay=0.01)

    policy_model, ref_model, optimizer, loader = accelerator.prepare(
        policy.model, reference.model, optimizer, loader
    )

    unwrap_policy = accelerator.unwrap_model(policy_model)
    unwrap_ref = accelerator.unwrap_model(ref_model)
    policy_model.train()
    ref_model.eval()

    device = accelerator.device

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(policy_model):
                # Policy forward (with grad)
                logp_p_chosen = _seq_log_prob(unwrap_policy, batch["chosen"], device)
                logp_p_rejected = _seq_log_prob(unwrap_policy, batch["rejected"], device)
                # Reference forward (no grad)
                with torch.no_grad():
                    logp_r_chosen = _seq_log_prob(unwrap_ref, batch["chosen"], device)
                    logp_r_rejected = _seq_log_prob(unwrap_ref, batch["rejected"], device)

                loss, stats = _dpo_loss(
                    logp_p_chosen, logp_p_rejected,
                    logp_r_chosen, logp_r_rejected,
                    beta=args.beta,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | DPO loss: {loss.item():.4f}"
                    f" | acc: {stats['acc']:.3f}"
                    f" | rc: {stats['rc']:.4f} rr: {stats['rr']:.4f}"
                )

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(init_path, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(init_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "voice_design"   # keep VoiceDesign
            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped = accelerator.unwrap_model(policy_model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped.state_dict().items()}
            for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
                del state_dict[k]
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


if __name__ == "__main__":
    train()
