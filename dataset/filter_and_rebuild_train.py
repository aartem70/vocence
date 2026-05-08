"""Filter specs by UTMOSv2 MOS and rebuild train_raw.jsonl + heldout_specs.jsonl.

Strategy:
    1. Read specs_with_mos.jsonl (output of score_clips_mos.py)
    2. Compute book-level mean MOS (more stable than clip-level)
    3. Drop clips whose book mean MOS is below the cutoff (default: 50th percentile)
        — book-level so we drop entire bad readers, not random clips
    4. Apply min/max length and text-quality filters as before
    5. Rebuild speaker-level train/heldout split (5% by book)

Defaults: keep top 50% of books by mean MOS. With ~110 books and ~75 clips/book
that yields roughly 4000 train clips (vs 7668 before) of much higher quality.

Usage:
    python filter_and_rebuild_train.py \
        --specs-mos /workspace/data/specs_with_mos.jsonl \
        --clips-dir /workspace/data/clips \
        --train-out /workspace/data/train_raw_hq.jsonl \
        --heldout-specs /workspace/data/heldout_specs_hq.jsonl \
        --ref-audio /workspace/data/ref_24k.wav \
        --book-mos-percentile 50
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs-mos", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--heldout-specs", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--book-mos-percentile", type=float, default=50.0,
                    help="drop books whose mean MOS is below this percentile (0-100)")
    ap.add_argument("--min-clip-mos", type=float, default=2.5,
                    help="also drop individual clips with MOS below this (NaN-safe)")
    ap.add_argument("--heldout-ratio", type=float, default=0.05)
    ap.add_argument("--min-text-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    rows: list[dict] = []
    n_total = n_no_mos = n_too_short = n_no_wav = 0
    for line in Path(args.specs_mos).open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_total += 1
        wav = clips_dir / r["wav_path"]
        text = (r.get("transcription") or "").strip()
        if not wav.exists():
            n_no_wav += 1
            continue
        if not text or len(text.split()) < args.min_text_words:
            n_too_short += 1
            continue
        mos = r.get("mos")
        if mos is None:
            n_no_mos += 1
            continue
        r["_wav_full"] = str(wav)
        r["_mos"] = float(mos)
        rows.append(r)
    print(f"[filter] read {n_total} specs; usable_with_mos={len(rows)} "
          f"(no_wav={n_no_wav}, too_short={n_too_short}, no_mos={n_no_mos})")

    # Book-level mean MOS
    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        b = str(r.get("book_id") or r.get("section_url") or r.get("clip_id"))
        by_book[b].append(r)
    book_mos = {b: statistics.mean(c["_mos"] for c in cs) for b, cs in by_book.items()}
    sorted_mos = sorted(book_mos.values())
    cutoff_idx = int(len(sorted_mos) * args.book_mos_percentile / 100.0)
    cutoff_idx = max(1, min(len(sorted_mos) - 1, cutoff_idx))
    book_mos_cutoff = sorted_mos[cutoff_idx]
    keep_books = {b for b, m in book_mos.items() if m >= book_mos_cutoff}
    print(f"[filter] books={len(by_book)} mos_cutoff={book_mos_cutoff:.3f} "
          f"(p{args.book_mos_percentile}) kept_books={len(keep_books)}")

    # Per-clip floor (NaN-safe)
    kept_rows = [
        r for r in rows
        if str(r.get("book_id") or r.get("section_url") or r.get("clip_id")) in keep_books
        and r["_mos"] >= args.min_clip_mos
    ]
    n_after_clip = len(kept_rows)
    print(f"[filter] after_clip_floor(>={args.min_clip_mos}): {n_after_clip}")

    # Speaker-level split (book-level) on kept books only
    kept_book_ids = sorted({
        str(r.get("book_id") or r.get("section_url") or r.get("clip_id"))
        for r in kept_rows
    })
    rng = random.Random(args.seed)
    rng.shuffle(kept_book_ids)
    n_heldout = max(1, int(args.heldout_ratio * len(kept_book_ids)))
    heldout_books = set(kept_book_ids[:n_heldout])

    train_rows = [r for r in kept_rows if str(r.get("book_id") or r.get("section_url") or r.get("clip_id")) not in heldout_books]
    heldout_rows = [r for r in kept_rows if str(r.get("book_id") or r.get("section_url") or r.get("clip_id")) in heldout_books]
    print(f"[filter] split by book: train={len(train_rows)}, heldout={len(heldout_rows)} "
          f"(heldout from {n_heldout}/{len(kept_book_ids)} books)")

    if not Path(args.ref_audio).exists():
        raise SystemExit(f"ref_audio missing: {args.ref_audio}")

    train_out = Path(args.train_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    with train_out.open("w") as fh:
        for r in train_rows:
            fh.write(json.dumps({
                "audio": r["_wav_full"], "text": r["transcription"], "ref_audio": args.ref_audio
            }) + "\n")
    print(f"[filter] wrote {train_out}  ({len(train_rows)} clips)")

    heldout_out = Path(args.heldout_specs)
    heldout_out.parent.mkdir(parents=True, exist_ok=True)
    with heldout_out.open("w") as fh:
        for r in heldout_rows:
            r2 = {k: v for k, v in r.items() if not k.startswith("_")}
            fh.write(json.dumps(r2) + "\n")
    print(f"[filter] wrote {heldout_out}  ({len(heldout_rows)} clips)")


if __name__ == "__main__":
    main()
