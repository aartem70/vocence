"""V2 build_qwen_jsonl: speaker-level (book-level) train/heldout split.

Avoids leakage: same audiobook's clips never span both train and heldout.

Inputs:
    --specs            specs.jsonl (with book_id, transcription, etc.)
    --clips-dir        directory of WAVs (referenced by spec.wav_path)
    --train-out        train_raw.jsonl (Qwen format: audio, text, ref_audio)
    --heldout-specs    out: held-out subset of specs.jsonl (for local_eval)
    --ref-audio        absolute path to a 24-kHz reference WAV
    --heldout-ratio    fraction by-book to hold out (default 0.05)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--heldout-specs", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--heldout-ratio", type=float, default=0.05)
    ap.add_argument("--min-text-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    rows: list[dict] = []
    for line in Path(args.specs).open():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        wav = clips_dir / s["wav_path"]
        text = (s.get("transcription") or "").strip()
        if not wav.exists() or not text or len(text.split()) < args.min_text_words:
            continue
        s["_wav_full"] = str(wav)
        rows.append(s)
    print(f"[build] {len(rows)} usable specs")

    # Group by book_id (LibriVox audiobook). Falls back to clip_id-as-book if missing.
    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        b = str(r.get("book_id") or r.get("section_url") or r.get("clip_id"))
        by_book[b].append(r)
    books = list(by_book.keys())
    rng = random.Random(args.seed)
    rng.shuffle(books)
    n_heldout = max(1, int(args.heldout_ratio * len(books)))
    heldout_books = set(books[:n_heldout])

    train_rows = [r for r in rows if str(r.get("book_id") or r.get("section_url") or r.get("clip_id")) not in heldout_books]
    heldout_rows = [r for r in rows if str(r.get("book_id") or r.get("section_url") or r.get("clip_id")) in heldout_books]
    print(f"[build] split by book: train={len(train_rows)}, heldout={len(heldout_rows)} "
          f"(heldout from {n_heldout}/{len(books)} books)")

    if not Path(args.ref_audio).exists():
        raise SystemExit(f"ref_audio missing: {args.ref_audio}")

    # train_raw.jsonl in Qwen format
    train_out = Path(args.train_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    with train_out.open("w") as fh:
        for r in train_rows:
            fh.write(json.dumps({
                "audio": r["_wav_full"], "text": r["transcription"], "ref_audio": args.ref_audio
            }) + "\n")
    print(f"[build] wrote {train_out}")

    # heldout: keep original spec format for local_eval.py
    heldout_out = Path(args.heldout_specs)
    heldout_out.parent.mkdir(parents=True, exist_ok=True)
    with heldout_out.open("w") as fh:
        for r in heldout_rows:
            r2 = {k: v for k, v in r.items() if not k.startswith("_")}
            fh.write(json.dumps(r2) + "\n")
    print(f"[build] wrote {heldout_out}")


if __name__ == "__main__":
    main()
