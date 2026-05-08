"""Launcher: run 4 sharded local_eval_fast.py instances in parallel + merge results.

Each shard pinned to one GPU via CUDA_VISIBLE_DEVICES, processes a 1/N slice of
the eval set, writes its own result JSON. After all shards finish, this script
merges them into one aggregate.

Usage (on box):
    python run_eval_fast.py \\
        --model-path /workspace/sft_out_rft/avg_last2 \\
        --specs /workspace/data/heldout_specs.jsonl \\
        --clips-dir /workspace/data/clips \\
        --n 30 --num-candidates 5 --num-shards 4 \\
        --out-dir /tmp/eval_fast_run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--eq-profile", default=None)
    ap.add_argument("--eq-strength", type=float, default=0.7)
    ap.add_argument("--mp3-bitrate", type=int, default=None)
    ap.add_argument("--noise-floor-db", type=float, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--script", default="/workspace/local_eval_fast.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for s in range(args.num_shards):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(s)
        log_path = out_dir / f"shard{s}.log"
        out_path = out_dir / f"shard{s}.json"
        cmd = [
            "python", "-u", args.script,
            "--backend", "local",
            "--model-path", args.model_path,
            "--specs", args.specs,
            "--clips-dir", args.clips_dir,
            "--n", str(args.n),
            "--num-candidates", str(args.num_candidates),
            "--seed", str(args.seed),
            "--judge-concurrency", str(args.judge_concurrency),
            "--device", "cuda:0",  # CUDA_VISIBLE_DEVICES already pins it
            "--shard-idx", str(s),
            "--num-shards", str(args.num_shards),
            "--tmp-dir", f"/tmp/eval_fast_shard{s}",
            "--out", str(out_path),
        ]
        if args.no_postprocess:
            cmd.append("--no-postprocess")
        if args.eq_profile:
            cmd.extend(["--eq-profile", args.eq_profile, "--eq-strength", str(args.eq_strength)])
        if args.mp3_bitrate is not None:
            cmd.extend(["--mp3-bitrate", str(args.mp3_bitrate)])
        if args.noise_floor_db is not None:
            cmd.extend(["--noise-floor-db", str(args.noise_floor_db)])

        log_fh = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((p, log_fh, log_path, out_path))
        print(f"[launcher] shard {s} on cuda:{s} -> pid={p.pid}", flush=True)

    print(f"[launcher] waiting for {len(procs)} shards...", flush=True)
    t0 = time.time()
    for p, fh, log_path, _ in procs:
        rc = p.wait()
        fh.close()
        print(f"[launcher] shard pid={p.pid} exit_code={rc}  log={log_path}", flush=True)

    # Merge
    print(f"[launcher] all shards done in {time.time()-t0:.1f}s. merging...", flush=True)
    all_results = []
    for _, _, _, out_path in procs:
        if not out_path.exists():
            continue
        with out_path.open() as f:
            data = json.load(f)
        all_results.extend(data.get("results") or [])

    n = max(1, len(all_results))
    win_rate = sum(1 for r in all_results if r["win"]) / n
    mean_w = sum(r["weighted"] for r in all_results) / n
    elt_mean = ({k: sum(r["elements"][k] for r in all_results) / n
                 for k in all_results[0]["elements"]}
                if all_results else {})

    print(f"\n=== MERGED AGGREGATE (n={len(all_results)} across {args.num_shards} shards) ===")
    print(f"  win rate:      {win_rate:.3f}")
    print(f"  mean weighted: {mean_w:.3f}")
    for k, v in elt_mean.items():
        from local_eval import ELEMENT_WEIGHTS
        wgt = ELEMENT_WEIGHTS.get(k, 0)
        print(f"    {k:12s} {v:.3f}  (weight={wgt:.2f})")

    merged = {
        "n": len(all_results),
        "num_shards": args.num_shards,
        "win_rate": win_rate, "mean_weighted": mean_w,
        "elements_mean": elt_mean, "results": all_results,
        "wallclock_s": time.time() - t0,
    }
    merged_path = out_dir / "merged.json"
    with merged_path.open("w") as f:
        json.dump(merged, f, indent=2)
    print(f"  merged -> {merged_path}")


if __name__ == "__main__":
    main()
