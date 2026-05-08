"""Build DPO chosen/rejected pairs from on-policy symmetric harvest.

Per NVR-Prosody recipe (arXiv:2509.18531):
  - Use ONLY pairs where both judging orders agreed (kills position bias).
  - For each spec: chosen = highest-rated preferred candidate, rejected = lowest-rated non-preferred.
  - If no clear winner/loser pair available, skip the spec.
  - Rolling reference = current SFT — handled at training time, not data prep.

Output JSONL has one row per pair:
    {text, instruction, chosen_audio, rejected_audio, score_gap}

Usage:
    python build_nvr_pairs.py \\
        --pairs onpolicy_pairs_s0.jsonl,...,s3.jsonl \\
        --out /workspace/data/nvr_pairs.jsonl \\
        --multi-pair-cap 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--multi-pair-cap", type=int, default=4,
                    help="cap on (winner, loser) pairs per spec")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_specs = n_pairs = n_skipped = 0

    with out.open("w") as fh:
        for fp in args.pairs.split(","):
            for line in Path(fp).open():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cands = [c for c in rec.get("candidates") or []
                         if c.get("audio_path") and Path(c["audio_path"]).exists()
                         and c.get("agreed")]
                if len(cands) < 2:
                    n_skipped += 1
                    continue
                preferred = [c for c in cands if c.get("preferred_over_source")]
                losers = [c for c in cands if not c.get("preferred_over_source")]
                if not preferred or not losers:
                    n_skipped += 1
                    continue
                n_specs += 1
                # Multi-pair: top-N winners x bottom-N losers (by composite score)
                sorted_w = sorted(preferred, key=lambda c: -c.get("composite", 0))
                sorted_l = sorted(losers, key=lambda c: c.get("composite", 0))
                budget = args.multi_pair_cap
                for w in sorted_w:
                    for l in sorted_l:
                        if budget <= 0:
                            break
                        fh.write(json.dumps({
                            "text": rec["text"],
                            "instruction": rec["instruction"],
                            "spec_traits": rec.get("spec_traits", {}),
                            "chosen_audio": w["audio_path"],
                            "rejected_audio": l["audio_path"],
                            "score_gap": w.get("composite", 0) - l.get("composite", 0),
                        }) + "\n")
                        n_pairs += 1
                        budget -= 1
                    if budget <= 0:
                        break

    print(f"[nvr-pairs] specs with usable agreed pairs: {n_specs}")
    print(f"[nvr-pairs] DPO pairs written: {n_pairs}")
    print(f"[nvr-pairs] skipped specs (no agreed winner+loser): {n_skipped}")


if __name__ == "__main__":
    main()
