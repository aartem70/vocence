"""Average the last N epoch checkpoints' weights into a new directory.

Reduces variance and typically nets a small but reliable improvement (well-known
in machine translation, image classification, and TTS).

Usage:
    python avg_checkpoints.py /workspace/sft_out 3
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    out_root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    ckpts = sorted([p for p in out_root.iterdir() if p.name.startswith("checkpoint-epoch-")],
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if len(ckpts) < n:
        print(f"only {len(ckpts)} checkpoints; need {n}")
        sys.exit(1)
    last = ckpts[-n:]
    print(f"averaging: {[p.name for p in last]}")

    import torch
    from safetensors.torch import load_file, save_file

    # Use the most recent checkpoint as the template for non-weight files
    avg_dir = out_root / f"avg_last{n}"
    if avg_dir.exists():
        shutil.rmtree(avg_dir)
    shutil.copytree(last[-1], avg_dir)

    sd_files = [c / "model.safetensors" for c in last]
    sds = [load_file(str(f)) for f in sd_files]

    keys = list(sds[0].keys())
    avg_sd = {}
    for k in keys:
        if not all(k in s for s in sds):
            avg_sd[k] = sds[-1][k]
            continue
        try:
            stacked = torch.stack([s[k].float() for s in sds])
            avg_sd[k] = stacked.mean(dim=0).to(sds[-1][k].dtype)
        except Exception:
            avg_sd[k] = sds[-1][k]

    save_file(avg_sd, str(avg_dir / "model.safetensors"))
    print(f"averaged {len(keys)} tensors -> {avg_dir}")


if __name__ == "__main__":
    main()
