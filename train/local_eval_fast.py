"""Fast multi-shard eval with batched candidate generation + spec-level parallelism.

Combines three optimizations over local_eval_bo5.py:
  (1) Batched candidate generation (qwen_batched.generate_batched, ~7x speedup)
  (2) Spec-level parallelism via asyncio.gather + gen_sem (GPT calls overlap)
  (3) Shard across N GPUs (mirrors harvest_preference_pairs sharding)

For 30 specs the original eval takes ~75 min; this should land ~6-10 min on 4 GPUs.

Per-shard usage (CUDA_VISIBLE_DEVICES=<gpu> sets the physical GPU):
    python local_eval_fast.py --backend local --model-path ... \\
        --specs ... --clips-dir ... --n 30 --num-candidates 5 \\
        --shard-idx 0 --num-shards 4 --out /tmp/eval_shard0.json

Or use the launcher `run_eval_fast.sh` to spawn 4 shards.
"""
from __future__ import annotations

import argparse
import asyncio
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
from local_eval_bo5 import CompositeScorer


def _generate_k_candidates(model, text, k):
    """Batched generation following the temp ladder. Returns list of (arr, sr)."""
    from qwen_batched import generate_batched
    if k <= 1:
        plan = [(1, 0.85)]
    elif k == 2:
        plan = [(1, 0.7), (1, 0.9)]
    elif k <= 4:
        half = max(1, k // 2)
        plan = [(half, 0.7), (k - half, 0.9)]
    else:
        # K>=5: spread across 3 temperatures
        a = k // 3
        b = (k - a) // 2
        c = k - a - b
        plan = [(a, 0.7), (b, 0.85), (c, 1.0)]
    return generate_batched(model, text, instruction="", plan=plan,
                            max_new_tokens=600)


class FastBackend:
    """Loads Qwen3-TTS model, generates K candidates batched, picks best with composite scorer."""

    def __init__(self, model_path: str, num_candidates: int = 5,
                 dtype_str: str = "bf16", device: str = "cuda:0",
                 postprocess: bool = False, postprocess_kwargs: dict | None = None):
        import torch
        from qwen_tts import Qwen3TTSModel
        self._device = device
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
        self._model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=dtype,
            attn_implementation="sdpa",
        )
        self._n = max(1, int(num_candidates))
        self._scorer = CompositeScorer()
        self._postprocess = postprocess
        self._postprocess_kwargs = postprocess_kwargs or {}
        self._postprocess_fn = None
        if self._postprocess:
            try:
                from audio_postprocess import postprocess as _pp
                self._postprocess_fn = _pp
            except Exception as e:
                print(f"[backend] postproc disabled: {type(e).__name__}: {e}", flush=True)
                self._postprocess = False
        print(f"[backend] best-of-{self._n} ready on {device}", flush=True)

    def synthesize(self, text: str, instruction: str):
        """Generate K candidates batched, score with composite, return best (arr, sr)."""
        kwargs = dict(
            text=[text] * self._n, language=["English"] * self._n,
            instruct=[instruction] * self._n,
            max_new_tokens=600, do_sample=True,
            temperature=0.85, top_p=0.92, top_k=50, repetition_penalty=1.10,
        )
        for drop in ([], ["max_new_tokens"], ["max_new_tokens", "top_k"],
                     ["max_new_tokens", "top_k", "repetition_penalty"]):
            try:
                trim = {k: v for k, v in kwargs.items() if k not in drop}
                waves, sr = self._model.generate_voice_design(**trim)
                break
            except TypeError:
                continue
        else:
            raise RuntimeError("all generate kwarg drops failed")
        candidates = []
        for w in waves:
            arr = np.asarray(w, dtype=np.float32).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            score = self._scorer.score(arr, int(sr), text)
            candidates.append((score, arr, int(sr)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        sc, arr, sr = candidates[0]
        if self._postprocess and self._postprocess_fn is not None:
            arr, sr = self._postprocess_fn(arr, sr, **self._postprocess_kwargs)
        return arr, sr


async def _eval_one(client, spec, backend, gen_sem, sem_judge, tmp_dir: Path, clips_dir: Path):
    """Evaluate one spec end-to-end. Returns dict or None on failure."""
    t0 = time.time()
    text = (spec.get("transcription") or "").strip()
    instruction = " | ".join(f"{k}: {spec.get(k, '')}" for k in
                             ("gender", "age_group", "pitch", "speed", "emotion", "tone", "accent")
                             if spec.get(k))
    # Generation must be serialized on the GPU
    async with gen_sem:
        try:
            arr, sr = await asyncio.to_thread(backend.synthesize, text, instruction)
        except Exception as e:
            return {"clip_id": spec.get("clip_id"), "error": f"gen: {type(e).__name__}: {e}"}
    engine_path = tmp_dir / f"engine_{spec['clip_id'][:16]}.wav"
    sf.write(str(engine_path), arr, int(sr))
    src_path = clips_dir / spec["wav_path"]
    try:
        pt = base.call_pointwise(client, str(engine_path), sem_judge)
        pw = base.call_pairwise(client, str(src_path), str(engine_path), instruction, sem_judge)
        traits, pw_res = await asyncio.gather(pt, pw)
    except Exception as e:
        return {"clip_id": spec.get("clip_id"), "error": f"judge: {type(e).__name__}: {e}"}
    elements = {
        "script": base.score_element("script", text, traits["transcription"]),
        "naturalness": 1.0 if pw_res.get("engine_more_natural") else 0.0,
    }
    for k in ("gender", "speed", "emotion", "age_group", "pitch", "accent", "tone"):
        elements[k] = base.score_element(k, spec.get(k, "neutral"), traits[k])
    weighted = sum(base.ELEMENT_WEIGHTS[e] * v for e, v in elements.items())
    won = weighted >= 0.9
    return {
        "clip_id": spec.get("clip_id"),
        "weighted": weighted, "win": won, "elements": elements,
        "dt": time.time() - t0,
    }


async def run(args):
    rows = []
    for line in Path(args.specs).open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.n > 0:
        rows = rows[: args.n]
    # Shard
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]
        print(f"[eval] shard {args.shard_idx}/{args.num_shards}: {len(rows)} specs", flush=True)
    else:
        print(f"[eval] running on {len(rows)} samples", flush=True)

    pp_kwargs = {
        "eq_profile_path": args.eq_profile,
        "eq_strength": args.eq_strength,
        "mp3_bitrate_kbps": args.mp3_bitrate,
        "noise_floor_db": args.noise_floor_db,
    }
    backend = FastBackend(
        args.model_path, num_candidates=args.num_candidates,
        device=args.device, postprocess=not args.no_postprocess,
        postprocess_kwargs=pp_kwargs,
    )

    tmp_dir = Path(args.tmp_dir); tmp_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir).resolve()
    gen_sem = asyncio.Semaphore(1)            # one gen at a time on the GPU
    sem_judge = asyncio.Semaphore(args.judge_concurrency)
    counter = {"done": 0}

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        async def _wrapper(i, spec):
            r = await _eval_one(client, spec, backend, gen_sem, sem_judge, tmp_dir, clips_dir)
            if r is None:
                return None
            counter["done"] += 1
            if "error" in r:
                print(f"[eval] {counter['done']}/{len(rows)} (idx={i+1}) ERROR: {r['error']}", flush=True)
                return None
            print(f"[eval] {counter['done']}/{len(rows)} (idx={i+1}) "
                  f"weighted={r['weighted']:.3f} win={r['win']} dt={r['dt']:.1f}s", flush=True)
            return r

        results = await asyncio.gather(*[_wrapper(i, s) for i, s in enumerate(rows)])
        results = [r for r in results if r is not None]

    n = max(1, len(results))
    win_rate = sum(1 for r in results if r["win"]) / n
    mean_w = sum(r["weighted"] for r in results) / n
    elt_mean = {k: sum(r["elements"][k] for r in results) / n for k in results[0]["elements"]} if results else {}
    out = {
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "n": len(results), "win_rate": win_rate, "mean_weighted": mean_w,
        "results": results, "elements_mean": elt_mean,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== shard {args.shard_idx} aggregate (n={len(results)}) ===")
    print(f"  win rate:      {win_rate:.3f}")
    print(f"  mean weighted: {mean_w:.3f}")
    for k, v in elt_mean.items():
        wgt = base.ELEMENT_WEIGHTS.get(k, 0)
        print(f"    {k:12s} {v:.3f}  (weight={wgt:.2f})")
    print(f"  saved to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local"], default="local")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all specs")
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tmp-dir", default="/tmp/eval_fast")
    ap.add_argument("--out", default="/tmp/local_eval_fast.json")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--eq-profile", default=None)
    ap.add_argument("--eq-strength", type=float, default=0.7)
    ap.add_argument("--mp3-bitrate", type=int, default=None)
    ap.add_argument("--noise-floor-db", type=float, default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
