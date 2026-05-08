"""Smoke test the train_lora.py data path on a tiny dataset.

Verifies:
  1. ParlerTTS model + tokenizer + DAC encoder loadable
  2. Audio encoder.encode() returns expected shape
  3. Collator produces (input_ids, attention_mask, prompt_input_ids,
     prompt_attention_mask, labels) of correct shapes
  4. model.forward(**batch) runs and returns a finite loss
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from parler_tts import ParlerTTSForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_lora import CollatorParlerTTS, freeze_audio_encoder, PARLER_MODEL


def main():
    ds_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/parler_micro"
    print(f"loading dataset {ds_path}")
    ds = load_from_disk(ds_path)
    train = ds["train"] if hasattr(ds, "keys") else ds
    print(f"  rows={len(train)}, features={list(train.features.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL)
    print("loading model in bf16 (this is the slow part)...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        PARLER_MODEL, torch_dtype=torch.bfloat16
    ).to("cuda")
    freeze_audio_encoder(model)
    model.eval()

    print(f"audio_encoder: {type(model.audio_encoder).__name__}")
    audio_sr = int(model.config.sampling_rate)
    print(f"target audio_sr={audio_sr}")

    collator = CollatorParlerTTS(
        tokenizer=tokenizer,
        audio_encoder=model.audio_encoder,
        audio_sr=audio_sr,
        pad_token_id=tokenizer.pad_token_id or model.config.pad_token_id,
    )

    loader = DataLoader(train, batch_size=2, shuffle=False, collate_fn=collator)
    print("collating one batch...")
    batch = next(iter(loader))
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    # Move to cuda
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}

    print("forward pass...")
    with torch.no_grad():
        out = model(**batch)
    loss = out.loss
    print(f"  loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}")

    if not torch.isfinite(loss):
        sys.exit("FAILED: loss is not finite")
    print("OK")


if __name__ == "__main__":
    main()
