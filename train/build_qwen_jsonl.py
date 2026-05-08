"""Convert our specs.jsonl → Qwen3-TTS's train_raw.jsonl format.

Qwen schema (from QwenLM/Qwen3-TTS/finetuning/README.md):
    {"audio": "/abs/path.wav", "text": "<transcript>", "ref_audio": "/abs/path/ref.wav"}

We use the SAME ref_audio for all samples per Qwen's recommendation (improves speaker
consistency during cloning fine-tune; the VoiceDesign model's instruction-following
capability is inherited from its pretraining and not the focus of this fine-tune).

Picking a representative LibriVox clip as ref_audio (clean adult speaker, ~22s).

Usage:
    python build_qwen_jsonl.py --specs /workspace/data/specs.jsonl \\
        --clips-dir /workspace/data/clips \\
        --out /workspace/data/train_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-audio", default=None,
                    help="path to reference WAV (defaults to first valid clip in specs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-text-words", type=int, default=4,
                    help="reject samples whose transcription is too short to train on")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    rows = []
    for line in Path(args.specs).open():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        wav = clips_dir / s["wav_path"]
        text = (s.get("transcription") or "").strip()
        if not wav.exists() or not text:
            continue
        if len(text.split()) < args.min_text_words:
            continue
        rows.append({"audio": str(wav), "text": text, "_clip_id": s.get("clip_id", "")})

    if args.limit:
        rows = rows[: args.limit]
    print(f"[build_qwen_jsonl] kept {len(rows)} usable rows")

    if not rows:
        raise SystemExit("no rows usable")

    ref = args.ref_audio or rows[0]["audio"]
    if not Path(ref).exists():
        raise SystemExit(f"ref_audio missing: {ref}")
    print(f"[build_qwen_jsonl] ref_audio: {ref}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"audio": r["audio"], "text": r["text"], "ref_audio": ref}) + "\n")
    print(f"[build_qwen_jsonl] wrote {out}")


if __name__ == "__main__":
    main()
