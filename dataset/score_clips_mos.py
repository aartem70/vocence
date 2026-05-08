"""Score every clip in specs.jsonl with UTMOSv2, append `mos` column.

Used to filter the SFT training set to high-naturalness LibriVox readers/books.
The judge's pairwise naturalness judge compares our output to the *source*
recording, so training on amateur or noisy LibriVox readings teaches the model
to sound amateur. Keep only the top quartile to reduce that drag.

Usage:
    python score_clips_mos.py \
        --specs /workspace/data/specs.jsonl \
        --clips-dir /workspace/data/clips \
        --out /workspace/data/specs_with_mos.jsonl \
        --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max", type=int, default=0, help="cap for testing (0 = all)")
    args = ap.parse_args()

    import torch
    import utmosv2
    device = torch.device(args.device)
    model = utmosv2.create_model(pretrained=True, device=device)
    print(f"[score] UTMOSv2 ready on {args.device}", flush=True)

    clips_dir = Path(args.clips_dir).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in Path(args.specs).open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if args.max:
        rows = rows[: args.max]
    print(f"[score] {len(rows)} clips to score", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    with out_path.open("w") as fh:
        for i, r in enumerate(rows):
            wav_path = clips_dir / r["wav_path"]
            try:
                wav, sr = sf.read(str(wav_path))
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = np.ascontiguousarray(wav, dtype=np.float32)
                mos = model.predict(data=wav, sr=int(sr))
                if hasattr(mos, "item"):
                    mos = float(mos.item() if mos.ndim == 0 else mos.flatten()[0].item())
                elif isinstance(mos, np.ndarray):
                    mos = float(mos.flatten()[0])
                else:
                    mos = float(mos)
                r["mos"] = mos
                n_ok += 1
            except Exception as e:
                r["mos"] = None
                r["_mos_error"] = f"{type(e).__name__}: {e}"
                n_fail += 1
            fh.write(json.dumps(r) + "\n")
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (len(rows) - i - 1) / max(rate, 0.001)
                print(
                    f"[score] {i+1}/{len(rows)} ok={n_ok} fail={n_fail} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    print(f"[score] done. ok={n_ok} fail={n_fail} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
