"""Microbenchmark generate_voice_clone vs generate_voice_design timing."""
import time, sys
from pathlib import Path
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

BASE = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
VD = "/workspace/sft_out_rft/avg_last2"

REF = "/workspace/data/voice_refs/ref_1_female_adult_eb722315.wav"
REF_TXT = open(REF.replace(".wav", ".txt")).read().strip()
TEXT = "On a bottle he indicated a chair Rome put down his traveling bag he took a glass."
INSTRUCT = "gender: female | age_group: adult | pitch: mid | speed: normal | emotion: neutral | tone: formal"

def time_call(fn, n=3):
    fn()  # warmup
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n

print("=== VoiceClone (Base) ===", flush=True)
m_clone = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:1", dtype=torch.bfloat16, attn_implementation="sdpa")

def vc():
    waves, sr = m_clone.generate_voice_clone(
        text=TEXT, language="English",
        ref_audio=REF, ref_text=REF_TXT,
        max_new_tokens=400, do_sample=True,
        temperature=0.85, top_p=0.92, top_k=50, repetition_penalty=1.10,
    )
    return waves

dt_vc = time_call(vc, n=3)
print(f"  generate_voice_clone: {dt_vc:.1f}s per call")

print()
print("=== VoiceDesign (RFT) ===", flush=True)
m_vd = Qwen3TTSModel.from_pretrained(VD, device_map="cuda:2", dtype=torch.bfloat16, attn_implementation="sdpa")

def vd():
    waves, sr = m_vd.generate_voice_design(
        text=TEXT, instruct=INSTRUCT, language="English",
        max_new_tokens=400, do_sample=True,
        temperature=0.75, top_p=0.92, top_k=50, repetition_penalty=1.10,
    )
    return waves

dt_vd = time_call(vd, n=3)
print(f"  generate_voice_design: {dt_vd:.1f}s per call")

print(f"\n  ratio: clone is {dt_vc/dt_vd:.1f}x slower than design")
