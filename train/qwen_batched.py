"""Batched best-of-N generation helper for Qwen3-TTS-VoiceDesign.

`generate_voice_design` accepts list inputs natively, so we can produce K
candidates in a single forward pass instead of K sequential calls. Realistic
speedup on L40: ~3-4x for K=8 (the GPU saturates the matmul instead of idling
between calls).

We support a temperature ladder by issuing one batched call per (count, temp)
group — diversity at the candidate level (Koel-TTS recipe) + amortized
batching overhead.

Public API:
    generate_batched(model, text, instruct, plan, max_new_tokens=600,
                     top_k=50, top_p=0.92, repetition_penalty=1.10,
                     language="English") -> List[(np.ndarray, int)]

`plan` is a list of (count, temperature) tuples, e.g.
    [(4, 0.7), (4, 0.9)]  -> 8 candidates total, 4 at T=0.7, 4 at T=0.9
"""
from __future__ import annotations

import numpy as np


def _to_mono_f32(seg) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    return arr


def _call_batched(model, texts, instructs, language, **kwargs):
    """Call generate_voice_design with batched lists, dropping unsupported kwargs."""
    drops = (
        (),
        ("max_new_tokens",),
        ("max_new_tokens", "top_k"),
        ("max_new_tokens", "top_k", "repetition_penalty"),
        ("max_new_tokens", "top_k", "repetition_penalty", "top_p"),
    )
    last_err = None
    for drop in drops:
        try:
            kw = {k: v for k, v in kwargs.items() if k not in drop}
            wavs, sr = model.generate_voice_design(
                text=texts, instruct=instructs, language=language, **kw,
            )
            return wavs, sr
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"all generate_voice_design kwarg combinations failed: {last_err}")


def generate_batched(
    model,
    text: str,
    instruct: str,
    plan: list[tuple[int, float]],
    max_new_tokens: int = 400,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.10,
    language: str = "English",
) -> list[tuple[np.ndarray, int]]:
    """Generate sum(count) candidates following the (count, temperature) plan.

    Returns a flat list of (wav, sr) tuples in plan-order (group by group).
    """
    candidates: list[tuple[np.ndarray, int]] = []
    for count, temp in plan:
        if count <= 0:
            continue
        texts = [text] * count
        instructs = [instruct] * count
        wavs, sr = _call_batched(
            model, texts, instructs, language=language,
            do_sample=True, temperature=float(temp),
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        # `wavs` is List[np.ndarray] of length `count` per the Qwen API
        for w in wavs:
            arr = _to_mono_f32(w)
            candidates.append((arr, int(sr)))
    return candidates


# ---------------- recipes ----------------

def koel_recipe_8(model, text: str, instruct: str, **kw):
    """4 cands @ T=0.7 + 4 cands @ T=0.9 — Koel-TTS-aligned diversity recipe."""
    return generate_batched(model, text, instruct,
                            plan=[(4, 0.7), (4, 0.9)], **kw)


def koel_recipe_16(model, text: str, instruct: str, **kw):
    """4×4 ladder for offline DPO data gen — wider diversity, longer wall."""
    return generate_batched(model, text, instruct,
                            plan=[(4, 0.6), (4, 0.75), (4, 0.9), (4, 1.0)], **kw)
