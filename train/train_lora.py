"""LoRA fine-tune Parler-TTS Large on Vocence-format (description, prompt, audio) triples.

Design notes:
- Audio encoder (DAC) is frozen — LoRA targets the text encoder + decoder attention.
- DAC encodes 44.1kHz mono → 9 codebooks × ~86 tokens/sec; ~25s clip → ~2150 tokens/codebook.
- We use the model's own forward() loss (cross-entropy over codebook tokens).

Usage:
    python train_lora.py \
        --dataset ./parler_dataset \
        --output ./lora_out \
        --num-epochs 3 \
        --per-device-batch-size 2

This is a minimal-but-working recipe. Tune lr, batch size, LoRA rank as needed.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from parler_tts import ParlerTTSForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

PARLER_MODEL = "parler-tts/parler-tts-large-v1"


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load WAV with soundfile, mono float32, resample to target_sr if needed."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return wav


@dataclass
class CollatorParlerTTS:
    """Build a single training batch.

    Uses the model's audio encoder (DAC) to convert wav -> codebook tokens lazily here.
    Audio encoder is moved to the same device as input batches in the training step.
    Pad codebook tokens with -100 so they don't contribute to loss.
    """

    tokenizer: AutoTokenizer
    audio_encoder: torch.nn.Module
    audio_sr: int
    pad_token_id: int
    max_audio_sec: float = 25.0

    def __call__(self, features: list[dict]) -> dict:
        descriptions = [f["description"] for f in features]
        prompts = [f["prompt"] for f in features]
        wavs = [load_wav(f["audio_path"], self.audio_sr) for f in features]

        # Truncate to max_audio_sec to bound memory.
        max_samples = int(self.max_audio_sec * self.audio_sr)
        wavs = [w[:max_samples] for w in wavs]

        desc = self.tokenizer(
            descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        pr = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=600
        )

        # Encode each clip individually to avoid alignment headaches with padded batches.
        # DAC.encode returns audio_codes shape (chunks=1, B=1, N_q, T_a).
        device = next(self.audio_encoder.parameters()).device
        ae_dtype = self.audio_encoder.dtype
        per_clip_codes: list[torch.Tensor] = []  # each [N_q, T_a_i]
        with torch.no_grad():
            for w in wavs:
                x = torch.from_numpy(w).float().to(device).to(ae_dtype).view(1, 1, -1)
                enc_out = self.audio_encoder.encode(x)
                codes = enc_out.audio_codes if hasattr(enc_out, "audio_codes") else enc_out["audio_codes"]
                # (chunks, 1, N_q, T_a) -> (N_q, T_a)
                codes = codes.squeeze(0).squeeze(0)
                per_clip_codes.append(codes)

        # Pad to common T_a, build labels [B, T_a_max, N_q] with -100 on padding.
        # Move to CPU before returning so DataLoader can pin_memory; Trainer moves to GPU later.
        T_max = max(c.shape[1] for c in per_clip_codes)
        N_q = per_clip_codes[0].shape[0]
        B = len(per_clip_codes)
        labels = torch.full((B, T_max, N_q), fill_value=-100, dtype=torch.long)
        for i, c in enumerate(per_clip_codes):
            T_i = c.shape[1]
            labels[i, :T_i, :] = c.transpose(0, 1).long().cpu()

        return {
            "input_ids": desc.input_ids.cpu(),
            "attention_mask": desc.attention_mask.cpu(),
            "prompt_input_ids": pr.input_ids.cpu(),
            "prompt_attention_mask": pr.attention_mask.cpu(),
            "labels": labels,
        }


def freeze_audio_encoder(model: ParlerTTSForConditionalGeneration) -> None:
    if hasattr(model, "audio_encoder"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        model.audio_encoder.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"[train] loading dataset from {args.dataset}")
    ds = load_from_disk(args.dataset)
    if not isinstance(ds, DatasetDict):
        raise SystemExit("Dataset must have train/test splits — re-run build_parler_dataset.py")
    print(f"[train] train={len(ds['train'])} test={len(ds['test'])}")

    print(f"[train] loading model {PARLER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        PARLER_MODEL, torch_dtype=torch.bfloat16
    )

    freeze_audio_encoder(model)

    # LoRA on the text encoder + decoder attention/feed-forward projections
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            # T5-style text encoder
            "q", "k", "v", "o",
            # decoder attention
            "q_proj", "k_proj", "v_proj", "out_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    audio_sr = int(model.config.sampling_rate)
    print(f"[train] target audio_sr={audio_sr}")
    collator = CollatorParlerTTS(
        tokenizer=tokenizer,
        audio_encoder=model.base_model.model.audio_encoder if hasattr(model, "base_model") else model.audio_encoder,
        audio_sr=audio_sr,
        pad_token_id=tokenizer.pad_token_id or model.config.pad_token_id,
    )

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,  # CUDA in collator → must run in main process
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output + "/final")
    print(f"[train] saved LoRA adapter to {args.output}/final")


if __name__ == "__main__":
    main()
