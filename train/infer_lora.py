"""Load Parler-TTS Large + a LoRA adapter and generate a few WAVs to verify
the adapter loads cleanly and produces audio.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from peft import PeftModel
from transformers import AutoTokenizer

PARLER = "parler-tts/parler-tts-large-v1"


def main():
    adapter_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/lora_pilot/final"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/lora_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {PARLER}")
    tokenizer = AutoTokenizer.from_pretrained(PARLER)
    base = ParlerTTSForConditionalGeneration.from_pretrained(
        PARLER, torch_dtype=torch.float16
    ).to("cuda")

    print(f"loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    sr = int(model.config.sampling_rate if hasattr(model, "config") else base.config.sampling_rate)
    print(f"sampling rate: {sr}")

    cases = [
        (
            "A middle-aged male speaker with a moderately pitched, formal voice "
            "delivers the words at a moderate pace in a neutral manner with a British accent. "
            "The recording is very high quality, clean, with no background noise, and sounds like natural human speech.",
            "the morning light filtered through the ancient oak trees as the village awoke",
        ),
        (
            "A young adult female speaker with a high-pitched, friendly voice "
            "delivers the words quickly in an excited and enthusiastic manner with an American accent. "
            "The recording is very high quality, clean, with no background noise, and sounds like natural human speech.",
            "you absolutely have to come see this place the cafe at the corner has the most incredible cinnamon rolls",
        ),
    ]

    for i, (description, prompt) in enumerate(cases):
        d = tokenizer(description, return_tensors="pt").to("cuda")
        p = tokenizer(prompt, return_tensors="pt").to("cuda")
        t = time.time()
        with torch.inference_mode():
            gen = model.generate(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids,
                prompt_attention_mask=p.attention_mask,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=int(28 * 86),
            )
        wav = gen.to(torch.float32).cpu().numpy().squeeze()
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        out = out_dir / f"sample_{i:02d}.wav"
        sf.write(out, wav, sr)
        print(f"  case {i}: dt={time.time()-t:.2f}s dur={wav.size/sr:.2f}s peak={float(np.abs(wav).max()):.3f} -> {out}")


if __name__ == "__main__":
    main()
