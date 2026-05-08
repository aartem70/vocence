"""local_eval variant using Qwen3-TTS-Base in VoiceClone mode.

Curated reference voices stored as (wav, txt) pairs in --refs-dir; filename
encodes gender and age (e.g. ref_1_female_adult_<id>.wav). Each spec selects
the best-matching reference and runs generate_voice_clone (ICL).

Reuses local_eval's call_pointwise / call_pairwise / scoring helpers so
results are directly comparable to other v3/v4/v5 evals on the same heldout.

Usage:
    python local_eval_voiceclone.py \\
        --base-model-path <path-to-Qwen3-TTS-12Hz-1.7B-Base> \\
        --refs-dir /workspace/data/voice_refs \\
        --specs /workspace/data/heldout_specs.jsonl \\
        --clips-dir /workspace/data/clips \\
        --n 30 --num-candidates 5 --device cuda:1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import local_eval as base
from local_eval_bo5 import CompositeScorer


def load_references(refs_dir: Path):
    refs = []
    for wav_file in sorted(refs_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
        m = re.match(r"ref_\d+_([a-z]+)_([a-z_]+)_[0-9a-f]+", wav_file.stem)
        if not m:
            continue
        refs.append({
            "wav_path": str(wav_file),
            "text": txt_file.read_text().strip(),
            "gender": m.group(1),
            "age_group": m.group(2),
        })
    return refs


def pick_reference(refs, spec):
    target_g = spec.get("gender") or "neutral"
    target_a = spec.get("age_group") or "adult"
    def s(r):
        return (10 if r["gender"] == target_g else 0) + (1 if r["age_group"] == target_a else 0)
    return max(refs, key=s)


class VoiceCloneEngine:
    def __init__(self, model_path, refs, num_candidates=5, device="cuda:0"):
        from qwen_tts import Qwen3TTSModel
        self._refs = refs
        self._n = max(1, int(num_candidates))
        self._device = device
        self._model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._scorer = CompositeScorer()
        print(f"[engine] best-of-{self._n} ready on {device}", flush=True)

    def synthesize(self, text, spec):
        ref = pick_reference(self._refs, spec)
        # Batched generation: K candidates in one forward pass.
        # max_new_tokens bumped to 2400 (= 200s of 12 Hz audio) so long
        # heldout transcriptions aren't truncated mid-sentence.
        # Lower temperature + higher rep penalty for content fidelity.
        # The model's free-running mode hallucinates the text; making sampling
        # more deterministic reduces drift from the input transcription.
        kwargs = dict(
            text=[text] * self._n, language=["English"] * self._n,
            ref_audio=[ref["wav_path"]] * self._n,
            ref_text=[ref["text"]] * self._n,
            max_new_tokens=2400, do_sample=True,
            temperature=0.3, top_p=0.85, top_k=20,
            repetition_penalty=1.20,
        )
        for drop in ([], ["max_new_tokens"], ["max_new_tokens", "top_k"],
                     ["max_new_tokens", "top_k", "repetition_penalty"]):
            trim = {k: v for k, v in kwargs.items() if k not in drop}
            try:
                waves, sr = self._model.generate_voice_clone(**trim)
                break
            except TypeError:
                continue
        else:
            raise RuntimeError("all generate attempts failed")
        candidates = []
        for w in waves:
            arr = np.asarray(w, dtype=np.float32).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            score = self._scorer.score(arr, int(sr), text)
            candidates.append((score, arr, int(sr)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        sc, arr, sr = candidates[0]
        scores = [round(c[0], 3) for c in candidates]
        print(f"[engine] ref={Path(ref['wav_path']).stem} cand_scores={scores} picked={sc:.3f}", flush=True)
        return arr, sr, ref


async def run(args):
    refs = load_references(Path(args.refs_dir))
    if not refs:
        raise SystemExit(f"no references in {args.refs_dir}")
    print(f"[eval] {len(refs)} references loaded", flush=True)
    engine = VoiceCloneEngine(args.base_model_path, refs,
                              num_candidates=args.num_candidates, device=args.device)

    rows = []
    for line in Path(args.specs).open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]
    print(f"[eval] running on {len(rows)} samples", flush=True)

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir).resolve()
    sem_judge = asyncio.Semaphore(args.judge_concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        async def _eval_one(i, spec):
            t0 = time.time()
            text = (spec.get("transcription") or "").strip()
            try:
                arr, sr, ref = engine.synthesize(text, spec)
            except Exception as e:
                print(f"[eval] {i+1}/{len(rows)} GEN ERROR: {type(e).__name__}: {e}", flush=True)
                return None
            engine_path = Path(args.tmp_dir) / f"engine_{spec['clip_id'][:16]}.wav"
            sf.write(str(engine_path), arr, int(sr))
            src_path = clips_dir / spec["wav_path"]
            try:
                pt_task = base.call_pointwise(client, str(engine_path), sem_judge)
                instruction = " | ".join(f"{k}: {spec.get(k, '')}" for k in
                                         ("gender","age_group","pitch","speed","emotion","tone","accent")
                                         if spec.get(k))
                pw_task = base.call_pairwise(client, str(src_path), str(engine_path),
                                             instruction, sem_judge)
                traits, pw = await asyncio.gather(pt_task, pw_task)
            except Exception as e:
                print(f"[eval] {i+1}/{len(rows)} JUDGE ERROR: {type(e).__name__}: {e}", flush=True)
                return None

            elements = {
                "script": base.score_element("script", text, traits["transcription"]),
                "naturalness": 1.0 if pw.get("engine_more_natural") else 0.0,
            }
            for k in ("gender", "speed", "emotion", "age_group", "pitch", "accent", "tone"):
                elements[k] = base.score_element(k, spec.get(k, "neutral"), traits[k])
            weighted = sum(base.ELEMENT_WEIGHTS[e] * v for e, v in elements.items())
            won = weighted >= 0.9
            print(f"[eval] {i+1}/{len(rows)} weighted={weighted:.3f} win={won} dt={time.time()-t0:.1f}s", flush=True)
            return {"weighted": weighted, "win": won, "elements": elements,
                    "ref": Path(ref["wav_path"]).stem}

        # Sequential because synthesize() is blocking GPU work
        for i, spec in enumerate(rows):
            r = await _eval_one(i, spec)
            if r:
                results.append(r)

    n = max(1, len(results))
    win_rate = sum(1 for r in results if r["win"]) / n
    mean_w = sum(r["weighted"] for r in results) / n
    elt_mean = {k: sum(r["elements"][k] for r in results) / n for k in results[0]["elements"]} if results else {}
    print()
    print(f"=== VoiceClone AGGREGATE (n={len(results)}) ===")
    print(f"  win rate: {win_rate:.3f}")
    print(f"  mean weighted: {mean_w:.3f}")
    for k, v in elt_mean.items():
        wgt = base.ELEMENT_WEIGHTS.get(k, 0)
        print(f"    {k:12s} {v:.3f}  (weight={wgt:.2f})")

    # Save raw results
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "win_rate": win_rate, "mean_weighted": mean_w,
            "results": results, "elements_mean": elt_mean,
        }, f, indent=2)
    print(f"  saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-path", required=True)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--tmp-dir", default="/tmp/eval_voiceclone")
    ap.add_argument("--out", default="/tmp/local_eval_voiceclone.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
