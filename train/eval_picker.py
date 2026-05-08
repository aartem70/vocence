"""Compare best-of-N pick rates: random / composite / trained picker / oracle.

Reuses the harvested pref_pairs2 dataset (200 specs x 4 candidates each, with
audio + GPT-4o-audio FIRST/SECOND labels). For each spec:
    1. Score every candidate with the trained picker
    2. argmax -> picked candidate
    3. Was the picked candidate preferred_over_source ?
    4. Aggregate win rate

Held-out split: same spec-level split as training (val_frac=0.15, seed=42).

Usage:
    python eval_picker.py \
        --pairs /workspace/data/pref_pairs2_s0.jsonl,...,s3.jsonl \
        --picker /workspace/data/picker_v1.pt \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--picker", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["val", "all"], default="val",
                    help="evaluate on val split only, or all specs (sanity)")
    args = ap.parse_args()

    pair_files = [Path(p) for p in args.pairs.split(",")]
    rows = []
    for fp in pair_files:
        for line in fp.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # spec-level split — match train_picker.py logic
    spec_ids = sorted({r["clip_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(spec_ids)
    n_val = max(1, int(len(spec_ids) * args.val_frac))
    val_set = set(spec_ids[:n_val])
    if args.mode == "val":
        rows = [r for r in rows if r["clip_id"] in val_set]
    print(f"[eval] mode={args.mode}: {len(rows)} specs", flush=True)

    # Load picker
    ckpt = torch.load(args.picker, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone_name", "microsoft/wavlm-base")
    use_static = bool(ckpt.get("use_static_features", True))

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_picker import WavLMPicker
    model = WavLMPicker(backbone_name, freeze=True, use_static_features=use_static)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device).eval()

    target_sr = 16000
    max_samples = int(12.0 * target_sr)

    def score_wav(audio_path, static):
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != target_sr:
            t = torch.from_numpy(wav).unsqueeze(0)
            wav = AF.resample(t, sr, target_sr).squeeze(0).numpy()
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(args.device)
        mask = torch.ones_like(wav_t, dtype=torch.bool, device=args.device)
        static_t = torch.tensor([static], dtype=torch.float32, device=args.device) if use_static else None
        with torch.no_grad():
            logit = model(wav_t, mask, static_t)
        return float(logit.item())

    # Win-rate calculations
    n_specs = 0
    wins_random = 0
    wins_composite = 0
    wins_picker = 0
    wins_oracle = 0
    rng2 = random.Random(args.seed)
    for r in rows:
        cands = r.get("candidates") or []
        cands = [c for c in cands if c.get("audio_path") and Path(c["audio_path"]).exists()]
        if not cands:
            continue
        n_specs += 1
        prefs = [1 if c["preferred_over_source"] else 0 for c in cands]

        # random pick: each candidate chosen w/ uniform prob -> expected = mean(prefs)
        wins_random += sum(prefs) / len(prefs)
        # composite picker (argmax of stored composite score)
        ci = int(np.argmax([c["composite"] for c in cands]))
        wins_composite += prefs[ci]
        # oracle
        wins_oracle += int(any(prefs))
        # trained picker
        scores = []
        for c in cands:
            s = score_wav(c["audio_path"],
                          [c.get("utmos", 0.5), c.get("whisper_score", 0.5), c.get("composite", 0.5)])
            scores.append(s)
        pi = int(np.argmax(scores))
        wins_picker += prefs[pi]

    print(f"\n=== Results (n={n_specs} specs, K={len(cands)}) ===", flush=True)
    print(f"  random pick:         {wins_random/n_specs:.3f}")
    print(f"  composite picker:    {wins_composite/n_specs:.3f}")
    print(f"  trained WavLM picker:{wins_picker/n_specs:.3f}")
    print(f"  oracle (best-of-K):  {wins_oracle/n_specs:.3f}")
    print(f"  picker captured fraction of (oracle - random) gap: "
          f"{(wins_picker - wins_random) / max(1e-6, wins_oracle - wins_random):.3f}")


if __name__ == "__main__":
    main()
