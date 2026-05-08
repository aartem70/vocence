"""Order-swap re-score: test for GPT-4o-audio position bias on our candidates.

For each (source, candidate) pair previously labeled in harvest 2/3:
    - We already have one verdict (FIRST/SECOND from the original presentation)
    - Re-query GPT-4o-audio with the OPPOSITE order
    - Compare verdicts:
        * AGREED (FIRST/SECOND match the same audio in both orders): high-confidence true preference
        * DISAGREED: judge is order-biased on this pair, exclude from score

Reports the "agreed" win rate vs the original noisy rate. If agreed rate is
much higher (e.g., 0.65 vs original 0.50), our naturalness ceiling is
position bias not the model.

Cost: ~$0.04 per re-query. 200 candidates ≈ $8.

Usage:
    python order_swap_test.py \
        --pairs /workspace/data/pref_pairs2_s0.jsonl,...,s3.jsonl \
        --n 200 --out /workspace/data/order_swap.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import sys
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import local_eval as base


GPT_MODEL = "gpt-4o-audio-preview"


def _b64(arr: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_path(path: str) -> str:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return _b64(np.asarray(wav, dtype=np.float32), int(sr))


async def _judge(client, src_b64, hyp_b64, task_desc, sem, swap=False):
    """Pairwise naturalness judge.

    swap=False: source is FIRST, candidate is SECOND (original convention).
        FIRST = source preferred, SECOND = candidate preferred.
    swap=True: candidate is FIRST, source is SECOND.
        FIRST = candidate preferred, SECOND = source preferred.
    """
    async with sem:
        if swap:
            audios = [{"type": "input_audio", "input_audio": {"data": hyp_b64, "format": "wav"}},
                      {"type": "input_audio", "input_audio": {"data": src_b64, "format": "wav"}}]
        else:
            audios = [{"type": "input_audio", "input_audio": {"data": src_b64, "format": "wav"}},
                      {"type": "input_audio", "input_audio": {"data": hyp_b64, "format": "wav"}}]
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


def _extract_records(pair_files):
    """Yield (instruction, source_path, candidate_path, original_verdict, original_preferred)
    for every candidate that was scored in harvest 2/3 with its audio_path saved."""
    for fp in pair_files:
        for line in Path(fp).open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = rec.get("instruction", "")
            src = rec.get("src_path")
            for c in rec.get("candidates") or []:
                ap = c.get("audio_path")
                if not ap or not Path(ap).exists() or not src or not Path(src).exists():
                    continue
                yield {
                    "instruction": inst, "src_path": src, "audio_path": ap,
                    "original_judgment": c.get("naturalness_judgment"),
                    "original_preferred": c.get("preferred_over_source"),
                    "clip_id": rec.get("clip_id"),
                }


async def main_async(args):
    pair_files = [Path(p) for p in args.pairs.split(",")]
    records = list(_extract_records(pair_files))
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.n] if args.n else records
    print(f"[swap] processing {len(records)} candidates", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = n_agreed = n_pref = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        with out.open("w") as fh:
            async def _one(rec):
                nonlocal n_done, n_agreed, n_pref
                src_b64 = _b64_path(rec["src_path"])
                hyp_b64 = _b64_path(rec["audio_path"])
                # Original was: src=FIRST, hyp=SECOND. judgment 'SECOND' = candidate preferred.
                # Swapped: hyp=FIRST, src=SECOND. judgment 'FIRST' = candidate preferred.
                try:
                    swapped = await _judge(client, src_b64, hyp_b64, rec["instruction"], sem, swap=True)
                except Exception as e:
                    print(f"[swap] {rec['clip_id']} ERROR: {type(e).__name__}: {e}", flush=True)
                    return
                orig_pref = rec["original_preferred"]  # bool
                # In swapped order, candidate preferred ↔ judgment == 'FIRST'
                swapped_pref = (swapped == "FIRST") if swapped else None
                agreed = (orig_pref == swapped_pref) if swapped_pref is not None else False
                rec_out = dict(rec)
                rec_out["swapped_judgment"] = swapped
                rec_out["swapped_preferred"] = swapped_pref
                rec_out["agreed"] = bool(agreed)
                # "true" preference: only if both orders agree the candidate was preferred
                rec_out["true_preferred"] = bool(agreed and orig_pref)
                fh.write(json.dumps(rec_out) + "\n")
                fh.flush()
                n_done += 1
                if agreed:
                    n_agreed += 1
                    if rec_out["true_preferred"]:
                        n_pref += 1
                if n_done % 20 == 0:
                    print(
                        f"[swap] {n_done}/{len(records)}  agree_rate={n_agreed/n_done:.3f}  "
                        f"true_pref_rate(among agreed)={n_pref/max(1,n_agreed):.3f}",
                        flush=True,
                    )

            await asyncio.gather(*[_one(r) for r in records])

    print(f"\n=== Final: n={n_done} ===")
    print(f"  agree_rate (judge stable across orders): {n_agreed/max(1,n_done):.3f}")
    print(f"  true preferred rate (among agreed):       {n_pref/max(1,n_agreed):.3f}")
    print(f"  vs original (single-order) rate:          ~0.50 (per pref_pairs harvest)")
    print(f"\n  → naturalness ceiling = {n_pref/max(1,n_agreed):.3f} when only agreed verdicts count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
