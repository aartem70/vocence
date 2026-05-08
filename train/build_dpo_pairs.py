"""Build chosen/rejected pair dataset for DPO training on Qwen3-TTS.

Per the Koel-TTS recipe: BOTH chosen and rejected come from the model's own
distribution. Using the LibriVox source as 'chosen' degrades performance.

Strategy:
    For each spec with K candidates:
        - "chosen"  = candidate with preferred_over_source=True AND highest composite
        - "rejected"= candidate with preferred_over_source=False AND lowest composite
    Skip specs where there's no winning AND losing candidate (we need both for DPO).

Output JSONL has one row per pair:
    {text, instruction, chosen_audio: PATH, rejected_audio: PATH,
     chosen_score, rejected_score, score_gap}

For RPO, the score_gap is used to weight the gradient.

Usage:
    python build_dpo_pairs.py \
        --pairs pref_pairs2_s0.jsonl,...,s3.jsonl \
        --out /workspace/data/dpo_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-strict", action="store_true",
                    help="require chosen=preferred AND rejected=not-preferred. "
                         "If unset, falls back to highest-vs-lowest weighted score.")
    ap.add_argument("--multi-pair", action="store_true",
                    help="for specs with multiple winners and losers, write the full "
                         "Cartesian product of (winner, loser) pairs — bootstraps the "
                         "training set when single-pair mode is data-starved.")
    ap.add_argument("--multi-pair-cap", type=int, default=8,
                    help="cap on (winner × loser) pairs per spec to prevent dominating")
    args = ap.parse_args()

    pair_files = [Path(p) for p in args.pairs.split(",")]
    rows = []
    for fp in pair_files:
        for line in fp.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_strict = n_fallback = n_skipped = 0

    with out.open("w") as fh:
        for rec in rows:
            cands = [c for c in rec.get("candidates") or []
                     if c.get("audio_path") and Path(c["audio_path"]).exists()]
            if len(cands) < 2:
                n_skipped += 1
                continue

            preferred = [c for c in cands if c["preferred_over_source"]]
            losers = [c for c in cands if not c["preferred_over_source"]]

            chosen = rejected = None
            multi_pairs = []
            if preferred and losers:
                if args.multi_pair:
                    # Cartesian product of winners × losers, capped.
                    sorted_w = sorted(preferred, key=lambda c: -c["weighted"])
                    sorted_l = sorted(losers, key=lambda c: c["weighted"])
                    n_w = min(len(sorted_w), args.multi_pair_cap)
                    n_l = min(len(sorted_l), args.multi_pair_cap)
                    for w in sorted_w[:n_w]:
                        for l in sorted_l[:n_l]:
                            multi_pairs.append((w, l))
                            if len(multi_pairs) >= args.multi_pair_cap:
                                break
                        if len(multi_pairs) >= args.multi_pair_cap:
                            break
                else:
                    chosen = max(preferred, key=lambda c: c["weighted"])
                    rejected = min(losers, key=lambda c: c["weighted"])
                n_strict += 1
            elif not args.require_strict:
                # fallback: highest vs lowest weighted score
                ordered = sorted(cands, key=lambda c: c["weighted"])
                if abs(ordered[-1]["weighted"] - ordered[0]["weighted"]) < 1e-6:
                    n_skipped += 1
                    continue
                rejected = ordered[0]
                chosen = ordered[-1]
                n_fallback += 1
            else:
                n_skipped += 1
                continue

            pairs_to_write = multi_pairs if multi_pairs else [(chosen, rejected)]
            for w, l in pairs_to_write:
                fh.write(json.dumps({
                    "text": rec["text"],
                    "instruction": rec["instruction"],
                    "spec_traits": rec.get("spec_traits", {}),
                    "chosen_audio": w["audio_path"],
                    "rejected_audio": l["audio_path"],
                    "chosen_weighted": w["weighted"],
                    "rejected_weighted": l["weighted"],
                    "score_gap": w["weighted"] - l["weighted"],
                    "chosen_preferred": w["preferred_over_source"],
                    "rejected_preferred": l["preferred_over_source"],
                }) + "\n")

    print(f"[dpo-pairs] wrote {out}")
    print(f"[dpo-pairs] strict pairs (preferred vs not): {n_strict}")
    print(f"[dpo-pairs] fallback pairs (high-vs-low score): {n_fallback}")
    print(f"[dpo-pairs] skipped specs: {n_skipped}")


if __name__ == "__main__":
    main()
