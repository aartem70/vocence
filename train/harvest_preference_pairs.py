"""Harvest preference labels for K candidates per spec.

For each spec in the input file:
    1. Generate K candidates with the policy model (default v4)
    2. For each candidate:
        - Run pointwise extraction (GPT-4o-audio) to get hyp_traits + transcription
        - Run pairwise NATURALNESS judgment vs source → FIRST/SECOND
        - Compute UTMOSv2 + faster-whisper features
    3. Compute the same judge-style weighted score per candidate
    4. Write a JSONL row with all candidates' info

Output is the foundation for:
    - Training a better picker (predict pairwise outcome from features)
    - Building DPO data (chosen=source, rejected=worst-scored candidate)
    - Self-distillation SFT (chosen=best-scored candidate)

Cost: ~K × 0.08$ per spec (2 GPT-4o-audio calls × K). Default K=3 → $0.24/spec.
For 200 specs that's ~$48.

Usage:
    python harvest_preference_pairs.py \
        --backend local \
        --model-path /workspace/sft_out_hq/avg_last3 \
        --specs /workspace/data/heldout_specs.jsonl \
        --clips-dir /workspace/data/clips \
        --candidates 3 --n 200 \
        --device cuda:1 \
        --out /workspace/data/pref_pairs.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import local_eval as base
from local_eval_bo5 import CompositeScorer  # we'll reuse the same scorer

GPT_MODEL = "gpt-4o-audio-preview"


def _fmt_wav_b64(arr: np.ndarray, sr: int) -> str:
    import base64
    buf = io.BytesIO()
    sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _gpt_pointwise(client, audio_b64: str, sem) -> dict:
    async with sem:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": GPT_MODEL,
                "modalities": ["text"],
                "messages": [
                    {"role": "system", "content": base.DESCRIPTION_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "input_audio",
                         "input_audio": {"data": audio_b64, "format": "wav"}}
                    ]},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return base.parse_traits(text)


async def _gpt_pairwise_naturalness(client, src_b64, hyp_b64, task_desc, sem,
                                    swap: bool = False) -> str | None:
    """Judge two clips. Returns 'FIRST' or 'SECOND' indicating which is more natural.

    Convention (swap=False): clip A = source, clip B = candidate. 'FIRST' = source preferred, 'SECOND' = candidate preferred.
    Convention (swap=True): clip A = candidate, clip B = source. 'FIRST' = candidate preferred, 'SECOND' = source preferred.
    """
    async with sem:
        if swap:
            audios = [
                {"type": "input_audio", "input_audio": {"data": hyp_b64, "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": src_b64, "format": "wav"}},
            ]
        else:
            audios = [
                {"type": "input_audio", "input_audio": {"data": src_b64, "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": hyp_b64, "format": "wav"}},
            ]
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": GPT_MODEL,
                "modalities": ["text"],
                "messages": [
                    {"role": "system",
                     "content": base.NATURALNESS_SYSTEM_TEMPLATE.format(task_description=task_desc)},
                    {"role": "user", "content": audios},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return base.parse_natural_first_or_second(text)


def _generate_k_candidates(model, text, instruction, k):
    """Batched generation following the Koel-TTS-style temp ladder.

    For k=3 we do (2, 0.7) + (1, 0.9). For k>=4 we split evenly across
    two temperatures. Single batched call per group amortizes GPU overhead;
    measured 3-7x speedup vs sequential bs=1 calls.
    """
    from qwen_batched import generate_batched
    if k <= 1:
        plan = [(1, 0.85)]
    elif k == 2:
        plan = [(1, 0.7), (1, 0.9)]
    elif k == 3:
        plan = [(2, 0.7), (1, 0.9)]
    else:
        half = k // 2
        plan = [(half, 0.7), (k - half, 0.9)]
    return generate_batched(model, text, instruction, plan=plan, max_new_tokens=400)


def _spec_traits_to_natural(spec: dict) -> str:
    """Build a natural-language instruction matching what the judge sends engines."""
    parts = []
    for k in ("gender", "age_group", "pitch", "speed", "emotion", "tone", "accent"):
        v = spec.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


async def _process_spec(client, spec, model, scorer, clips_dir, k, sem, gen_sem,
                        audio_out_dir=None, pairwise_only=False,
                        skip_composite: bool = False):
    text = (spec.get("transcription") or "").strip()
    instruction = _spec_traits_to_natural(spec)
    src_path = clips_dir / spec["wav_path"]
    src_wav, src_sr = sf.read(str(src_path))
    if src_wav.ndim > 1:
        src_wav = src_wav.mean(axis=1)
    src_wav = np.asarray(src_wav, dtype=np.float32)
    src_b64 = _fmt_wav_b64(src_wav, src_sr)

    # Generation is GPU-bound and blocking — gate it with gen_sem and run in a thread
    # so other specs can keep their GPT calls in flight while we generate.
    async with gen_sem:
        candidates = await asyncio.to_thread(_generate_k_candidates, model, text, instruction, k)
    if not candidates:
        return None

    # Fire GPT calls concurrently. For SYMMETRIC mode we judge each candidate
    # in BOTH orders (source-first and candidate-first) and only keep the verdict
    # if both orders agree — this kills the ~46.5% position bias the judge has.
    cand_b64s = [_fmt_wav_b64(arr, sr) for (arr, sr) in candidates]
    pair_tasks = [_gpt_pairwise_naturalness(client, src_b64, b, instruction, sem)
                  for b in cand_b64s]
    pair_swap_tasks = [_gpt_pairwise_naturalness(client, src_b64, b, instruction, sem, swap=True)
                       for b in cand_b64s]
    if pairwise_only:
        all_pairs = await asyncio.gather(*(pair_tasks + pair_swap_tasks), return_exceptions=True)
        pair_results = all_pairs[: len(cand_b64s)]
        pair_swap_results = all_pairs[len(cand_b64s) :]
        point_results = [None] * len(cand_b64s)
    else:
        point_tasks = [_gpt_pointwise(client, b, sem) for b in cand_b64s]
        all_results = await asyncio.gather(*(point_tasks + pair_tasks + pair_swap_tasks),
                                           return_exceptions=True)
        point_results = all_results[: len(cand_b64s)]
        pair_results = all_results[len(cand_b64s) : 2 * len(cand_b64s)]
        pair_swap_results = all_results[2 * len(cand_b64s) :]

    cand_records = []
    for cand_idx, ((arr, sr), traits_or_err, naturalness_or_err, swap_or_err) in enumerate(
            zip(candidates, point_results, pair_results, pair_swap_results)):
        if isinstance(naturalness_or_err, Exception) or isinstance(swap_or_err, Exception) or (
            not pairwise_only and isinstance(traits_or_err, Exception)):
            print(f"[harvest] GPT call failed (skip)", flush=True)
            continue
        traits = None if pairwise_only else traits_or_err
        naturalness = naturalness_or_err
        # Symmetric verdict: both orders must agree on the candidate being preferred
        swap_verdict = swap_or_err  # 'FIRST' = candidate preferred under swap
        # Original convention: source=FIRST, hyp=SECOND. naturalness=='SECOND' means hyp preferred.
        # Swap convention: hyp=FIRST, source=SECOND. swap_verdict=='FIRST' means hyp preferred.
        original_pref = (naturalness == "SECOND")
        swap_pref = (swap_verdict == "FIRST")
        agreed = (original_pref == swap_pref)
        # composite picker score (what we use locally) — skip if requested for speed.
        # We only need GPT-4o-audio labels for DPO; UTMOSv2/Whisper aren't used for
        # pair filtering. Skipping saves ~15-20s/spec.
        if skip_composite:
            composite = 0.5
            utmos = 0.5
            whisper_ok = 0.5
        else:
            composite = scorer.score(arr, sr, text)
            utmos = scorer._utmos_score(arr, sr)
            whisper_ok = scorer._whisper_wer(arr, sr, text)
        # persist the candidate audio so a downstream picker can be trained on raw WAVs
        cand_audio_path = None
        if audio_out_dir is not None:
            cand_audio_path = str(audio_out_dir / f"{spec['clip_id']}_c{cand_idx}.wav")
            sf.write(cand_audio_path, arr, int(sr), subtype="PCM_16")

        # judge-style weighted score (only when we have pointwise traits)
        elements = {}
        if not pairwise_only:
            elements["script"] = base.score_element("script", text, traits["transcription"])
            for k_ in ("gender", "speed", "emotion", "age_group", "pitch", "accent", "tone"):
                elements[k_] = base.score_element(k_, spec.get(k_, "neutral"), traits[k_])
        elements["naturalness"] = 1.0 if naturalness == "SECOND" else 0.0  # SECOND = our gen preferred
        weighted = (
            sum(base.ELEMENT_WEIGHTS[e] * v for e, v in elements.items())
            if not pairwise_only else float(elements["naturalness"])
        )
        cand_records.append({
            "naturalness_judgment": naturalness,  # FIRST = source preferred, SECOND = candidate preferred
            "preferred_over_source": original_pref,
            "swap_judgment": swap_verdict,
            "swap_preferred_over_source": swap_pref,
            "agreed": bool(agreed),
            "composite": composite, "utmos": utmos, "whisper_score": whisper_ok,
            "elements": elements, "weighted": weighted,
            "traits": traits,
            "audio_path": cand_audio_path,
        })
    return {
        "clip_id": spec.get("clip_id"),
        "text": text,
        "instruction": instruction,
        "spec_traits": {k: spec.get(k) for k in ("gender", "speed", "emotion", "age_group", "pitch", "accent", "tone")},
        "src_path": str(src_path),
        "candidates": cand_records,
    }


async def _run(args):
    import torch
    from qwen_tts import Qwen3TTSModel
    print(f"[harvest] loading model on {args.device}", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    scorer = CompositeScorer()
    print("[harvest] scorer ready", flush=True)

    rows = []
    for line in Path(args.specs).open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n] if args.n > 0 else rows
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]
        print(f"[harvest] shard {args.shard_idx}/{args.num_shards}: {len(rows)} specs (k={args.candidates})", flush=True)
    else:
        print(f"[harvest] {len(rows)} specs to process, k={args.candidates}", flush=True)

    clips_dir = Path(args.clips_dir).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sem_judge = asyncio.Semaphore(args.judge_concurrency)
    gen_sem = asyncio.Semaphore(1)  # one generation at a time on the GPU
    write_lock = asyncio.Lock()
    counter = {"done": 0}

    audio_out_dir = None
    if args.audio_out_dir:
        audio_out_dir = Path(args.audio_out_dir)
        audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[harvest] saving candidate WAVs to {audio_out_dir}", flush=True)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)) as client:
        with out.open("w") as fh:
            async def _one(i, spec):
                t0 = time.time()
                try:
                    rec = await _process_spec(client, spec, model, scorer, clips_dir,
                                              args.candidates, sem_judge, gen_sem,
                                              audio_out_dir=audio_out_dir,
                                              pairwise_only=args.pairwise_only,
                                              skip_composite=args.skip_composite)
                except Exception as e:
                    print(f"[harvest] {i+1}/{len(rows)} ERROR: {type(e).__name__}: {e}", flush=True)
                    return
                if rec is None:
                    return
                async with write_lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    counter["done"] += 1
                pref_count = sum(1 for c in rec["candidates"] if c["preferred_over_source"])
                k_total = len(rec["candidates"])
                print(f"[harvest] {counter['done']}/{len(rows)} (idx={i+1}) "
                      f"preferred={pref_count}/{k_total} dt={time.time()-t0:.1f}s", flush=True)

            # Schedule all specs concurrently; gen_sem serializes GPU work,
            # asyncio overlaps GPT calls across specs.
            await asyncio.gather(*[_one(i, s) for i, s in enumerate(rows)])

    print(f"[harvest] done -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local"], default="local")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=200, help="0 = all specs")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-out-dir", default=None,
                    help="if set, save each generated candidate WAV here (one per candidate)")
    ap.add_argument("--pairwise-only", action="store_true",
                    help="skip the pointwise trait extraction call. Halves GPT cost and "
                         "latency. Use when only training a naturalness picker (the trait "
                         "scores aren't needed).")
    ap.add_argument("--skip-composite", action="store_true",
                    help="skip UTMOSv2 + Whisper composite scoring. Saves ~15-20s/spec on the "
                         "GPU side. Safe for DPO data collection (pair filter uses only the "
                         "GPT-4o-audio agreement signal, not composite).")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="this worker's index in [0, num_shards)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of parallel workers; specs will be split modulo this")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
