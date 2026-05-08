"""Microbenchmark: sequential vs batched generate_voice_design on Qwen3-TTS."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qwen_batched import generate_batched

import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/workspace/sft_out_hq/avg_last3"
TEXT = ("On a bottle he indicated a chair Rome put down his traveling bag "
        "he took a glass I am curious he observed why simper Tyrannus")
INSTRUCT = "gender: male | age_group: adult | pitch: mid | speed: normal | emotion: neutral | tone: formal | accent: us"

def main():
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    # Warmup
    print("[bench] warmup", flush=True)
    _ = generate_batched(model, TEXT, INSTRUCT, plan=[(1, 0.8)], max_new_tokens=400)

    # Sequential bs=1 × 8
    print("[bench] sequential bs=1 × 8", flush=True)
    t0 = time.time()
    for _ in range(8):
        _ = generate_batched(model, TEXT, INSTRUCT, plan=[(1, 0.85)], max_new_tokens=400)
    seq_dt = time.time() - t0
    print(f"[bench] sequential bs=1 × 8 -> {seq_dt:.1f}s", flush=True)

    # Batched bs=8
    print("[bench] batched bs=8", flush=True)
    t0 = time.time()
    cands = generate_batched(model, TEXT, INSTRUCT, plan=[(8, 0.85)], max_new_tokens=400)
    bat_dt = time.time() - t0
    print(f"[bench] batched bs=8 -> {bat_dt:.1f}s ({len(cands)} cands)", flush=True)

    # Koel 4+4
    print("[bench] koel plan (4,0.7)+(4,0.9)", flush=True)
    t0 = time.time()
    cands = generate_batched(model, TEXT, INSTRUCT, plan=[(4, 0.7), (4, 0.9)], max_new_tokens=400)
    k_dt = time.time() - t0
    print(f"[bench] koel 4+4 -> {k_dt:.1f}s ({len(cands)} cands)", flush=True)

    print(f"\n[bench] speedup batched vs sequential: {seq_dt/bat_dt:.2f}x")
    print(f"[bench] speedup koel vs sequential: {seq_dt/k_dt:.2f}x")


if __name__ == "__main__":
    main()
