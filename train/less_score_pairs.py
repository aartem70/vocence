"""LESS-inspired influence scoring for DPO pair selection.

True LESS (Xia et al., ICML 2024) uses LoRA gradients projected to ~k dimensions
to compute training-example influence on a validation set. For our 1.7B TTS
model that's a multi-day engineering effort.

This is a pragmatic proxy that captures LESS's KEY INSIGHT:

    "Pairs whose gradients would most reduce loss on the target validation set
     are the most influential. Approximate via task similarity."

We approximate gradient similarity by **task-feature similarity**:
    - Each DPO pair has (text, instruction) — its TASK fingerprint
    - The "target" is heldout specs we currently LOSE on
    - LESS-score = max cosine(pair_embedding, losing_spec_embedding) × score_gap_weight

Pairs aligned with currently-losing-specs get high scores. Combined with
DPO score_gap (= preference signal strength), this prioritizes pairs that:
    (a) teach what we currently fail to learn (LESS proxy)
    (b) have a clear preference signal (DPO data-quality gate)

Output: same as build_nvr_pairs.py but with `less_score` field, sorted desc.

Usage:
    python less_score_pairs.py \\
        --pairs /workspace/data/nvr_pairs.jsonl \\
        --target-eval /tmp/local_eval_v5_rft.json \\
        --target-specs /workspace/data/heldout_specs.jsonl \\
        --top-k 100 \\
        --out /workspace/data/nvr_pairs_less100.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_eval_results(eval_json_path: str) -> dict[str, dict]:
    """Load per-spec eval results (clip_id -> {win: bool, weighted: float})."""
    if not Path(eval_json_path).exists():
        return {}
    with open(eval_json_path) as f:
        data = json.load(f)
    out = {}
    for r in data.get("results") or []:
        cid = r.get("clip_id") or r.get("ref")
        if cid:
            out[str(cid)] = {"win": r.get("win", False), "weighted": r.get("weighted", 0.0)}
    return out


def _load_target_specs(specs_path: str, eval_results: dict) -> list[dict]:
    """Build the LOSER target set: specs we currently lose on (or near 0.9 threshold).

    If we don't have per-spec eval results yet, fall back to using the full heldout
    as the target — this still moves data toward heldout-style content.
    """
    specs = []
    for line in Path(specs_path).open():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = str(s.get("clip_id") or s.get("wav_path", "").split("/")[-1].rsplit(".", 1)[0])
        # If we have eval results, weight by how badly we lost
        ev = eval_results.get(cid)
        if ev is not None:
            if ev["win"]:
                continue  # already winning, low influence
            # not winning — weight = how close to threshold (closer = more influential)
            weight = max(0.0, 0.95 - ev["weighted"])
        else:
            weight = 1.0  # treat all heldout as targets
        s["_target_weight"] = weight
        specs.append(s)
    return specs


def _embed_with_minilm(texts: list[str]) -> np.ndarray:
    """Lightweight text embedding for similarity. Falls back to bag-of-words if model missing."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = m.encode(texts, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)
    except Exception:
        # Fallback: TF-IDF-style bag of words
        from collections import Counter
        vocab = set()
        toks_list = [(t or "").lower().split() for t in texts]
        for toks in toks_list:
            vocab.update(toks)
        vocab = sorted(vocab)
        vidx = {v: i for i, v in enumerate(vocab)}
        embs = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, toks in enumerate(toks_list):
            c = Counter(toks)
            for tok, n in c.items():
                embs[i, vidx[tok]] = n
            norm = np.linalg.norm(embs[i]) + 1e-8
            embs[i] /= norm
        return embs


def _build_signature(text: str, instruction: str | None) -> str:
    """Combine text + instruction into one fingerprint string."""
    return f"{instruction or ''} || {text or ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="DPO pairs jsonl (output of build_nvr_pairs)")
    ap.add_argument("--target-eval", default="",
                    help="local_eval_*.json from current best model — used to mark losers as targets")
    ap.add_argument("--target-specs", required=True,
                    help="heldout_specs.jsonl — defines the target distribution")
    ap.add_argument("--top-k", type=int, default=100,
                    help="number of highest-influence pairs to keep")
    ap.add_argument("--score-gap-weight", type=float, default=0.5,
                    help="how much to weigh score_gap vs task-similarity (0 = pure similarity, 1 = pure gap)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = []
    for line in Path(args.pairs).open():
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"[less] loaded {len(pairs)} candidate pairs")

    eval_results = _load_eval_results(args.target_eval) if args.target_eval else {}
    target_specs = _load_target_specs(args.target_specs, eval_results)
    print(f"[less] target set: {len(target_specs)} specs"
          f" ({sum(1 for s in target_specs if s.get('_target_weight', 1) > 0)} with non-zero weight)")
    if not target_specs:
        raise SystemExit("no target specs — eval JSON might be misformatted or all heldout specs were winners")

    # Build signatures
    pair_sigs = [_build_signature(p["text"], p.get("instruction")) for p in pairs]
    target_sigs = [_build_signature(s.get("transcription", ""), None) for s in target_specs]
    target_weights = np.asarray([s.get("_target_weight", 1.0) for s in target_specs], dtype=np.float32)

    # Embed
    all_texts = pair_sigs + target_sigs
    print(f"[less] embedding {len(all_texts)} texts...")
    embs = _embed_with_minilm(all_texts)
    pair_embs = embs[: len(pair_sigs)]
    target_embs = embs[len(pair_sigs) :]

    # Similarity matrix: each pair vs each target
    # Using max-similarity-to-any-target × target_weight
    sims = pair_embs @ target_embs.T  # (n_pairs, n_targets)
    weighted_sims = sims * target_weights[None, :]
    # max influence per pair
    max_inf = weighted_sims.max(axis=1)

    # Combine with score_gap (DPO data quality)
    score_gaps = np.asarray([float(p.get("score_gap", 0.0)) for p in pairs], dtype=np.float32)
    # Normalize each axis to [0,1]
    if max_inf.max() > max_inf.min():
        max_inf_n = (max_inf - max_inf.min()) / (max_inf.max() - max_inf.min() + 1e-9)
    else:
        max_inf_n = np.zeros_like(max_inf)
    if score_gaps.max() > score_gaps.min():
        gap_n = (score_gaps - score_gaps.min()) / (score_gaps.max() - score_gaps.min() + 1e-9)
    else:
        gap_n = np.zeros_like(score_gaps)
    less_score = (1 - args.score_gap_weight) * max_inf_n + args.score_gap_weight * gap_n

    # Rank
    order = np.argsort(-less_score)
    keep = order[: args.top_k]
    print(f"[less] keeping top {args.top_k} pairs (less_score range: "
          f"{less_score[keep].min():.3f} - {less_score[keep].max():.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as fh:
        for i in keep:
            p = dict(pairs[int(i)])
            p["less_score"] = float(less_score[int(i)])
            p["less_max_target_sim"] = float(max_inf[int(i)])
            fh.write(json.dumps(p) + "\n")
    print(f"[less] wrote {args.out}")


if __name__ == "__main__":
    main()
