"""local_eval variant using CosyVoice2-0.5B zero_shot voice cloning.

CosyVoice2 zero-shot picks the gender+age-matched reference and clones onto
the spec text. Compared to Qwen3-TTS-Base voice clone, this should:
  - Maintain naturalness (proven by VoiceClone test: 0.80 vs Qwen-VoiceDesign 0.60)
  - Have better script accuracy (Qwen's voice clone hit 0.49 due to truncation)
  - Be faster (RTF ~0.8)

Reuses local_eval's call_pointwise / call_pairwise / scoring helpers so
results are directly comparable to other v3/v4/v5 evals.

Usage:
    python local_eval_cosyvoice.py \\
        --refs-dir /workspace/data/voice_refs \\
        --specs /workspace/data/heldout_specs.jsonl \\
        --clips-dir /workspace/data/clips \\
        --n 30 --device cuda:1
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")

import local_eval as base


def load_references(refs_dir: Path):
    refs = []
    for wav_file in sorted(refs_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
        m = re.match(r"ref_\d+_([a-z]+)_([a-z_]+)_[0-9a-f]+", wav_file.stem)
        if not m:
            continue
        full_text = txt_file.read_text().strip()
        # CosyVoice warns if synthesis text is shorter than ref_text. Truncate ref_text
        # to first 8 words (~3-4 seconds) which is plenty for ICL conditioning.
        ref_text = " ".join(full_text.split()[:8])
        refs.append({
            "wav_path": str(wav_file),
            "ref_text": ref_text,
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


class CosyVoiceEngine:
    def __init__(self, model_dir: str, refs):
        from cosyvoice.cli.cosyvoice import AutoModel
        self._model = AutoModel(model_dir=model_dir)
        self._refs = refs
        print(f"[engine] CosyVoice2 ready, sr={self._model.sample_rate}", flush=True)

    def synthesize(self, text, spec):
        ref = pick_reference(self._refs, spec)
        # For English content, cross_lingual mode is the supported path
        # (zero_shot mangles non-Chinese content). The <|en|> token tags the
        # target language; the reference audio carries the cloned voice.
        prefixed = f"<|en|>{text}"
        results = list(self._model.inference_cross_lingual(
            prefixed, ref["wav_path"], text_frontend=False
        ))
        if not results:
            raise RuntimeError("inference_cross_lingual returned no chunks")
        # Concat all chunks (CosyVoice may stream)
        chunks = [r["tts_speech"].numpy().squeeze() for r in results]
        if any(c.ndim > 1 for c in chunks):
            chunks = [c.mean(axis=0) if c.ndim > 1 else c for c in chunks]
        wav = np.concatenate(chunks).astype(np.float32)
        sr = self._model.sample_rate
        return wav, sr, ref


async def run(args):
    refs = load_references(Path(args.refs_dir))
    if not refs:
        raise SystemExit(f"no refs in {args.refs_dir}")
    print(f"[eval] {len(refs)} refs loaded", flush=True)
    engine = CosyVoiceEngine(args.model_dir, refs)

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
        for i, spec in enumerate(rows):
            t0 = time.time()
            text = (spec.get("transcription") or "").strip()
            try:
                arr, sr, ref = engine.synthesize(text, spec)
            except Exception as e:
                print(f"[eval] {i+1}/{len(rows)} GEN ERROR: {type(e).__name__}: {e}", flush=True)
                continue
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
                continue

            elements = {
                "script": base.score_element("script", text, traits["transcription"]),
                "naturalness": 1.0 if pw.get("engine_more_natural") else 0.0,
            }
            for k in ("gender", "speed", "emotion", "age_group", "pitch", "accent", "tone"):
                elements[k] = base.score_element(k, spec.get(k, "neutral"), traits[k])
            weighted = sum(base.ELEMENT_WEIGHTS[e] * v for e, v in elements.items())
            won = weighted >= 0.9
            print(f"[eval] {i+1}/{len(rows)} weighted={weighted:.3f} win={won} dt={time.time()-t0:.1f}s", flush=True)
            results.append({"weighted": weighted, "win": won, "elements": elements,
                            "ref": Path(ref["wav_path"]).stem})

    n = max(1, len(results))
    win_rate = sum(1 for r in results if r["win"]) / n
    mean_w = sum(r["weighted"] for r in results) / n
    elt_mean = {k: sum(r["elements"][k] for r in results) / n for k in results[0]["elements"]} if results else {}
    print()
    print(f"=== CosyVoice2 AGGREGATE (n={len(results)}) ===")
    print(f"  win rate: {win_rate:.3f}")
    print(f"  mean weighted: {mean_w:.3f}")
    for k, v in elt_mean.items():
        wgt = base.ELEMENT_WEIGHTS.get(k, 0)
        print(f"    {k:12s} {v:.3f}  (weight={wgt:.2f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"win_rate": win_rate, "mean_weighted": mean_w,
                   "results": results, "elements_mean": elt_mean}, f, indent=2)
    print(f"  saved to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/root/.cache/huggingface/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/e6287195fc87df8cacb06d1e18e87434b22f506a")
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--tmp-dir", default="/tmp/eval_cosyvoice")
    ap.add_argument("--out", default="/tmp/local_eval_cosyvoice.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
