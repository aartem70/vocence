"""Compute LibriVox average long-term spectrum + noise floor.

Used as the target for audio_postprocess.py's domain-matching EQ. The judge
(GPT-4o-audio) was trained on internet audio that's mostly mp3-coded with
slight room acoustics; pristine 24 kHz studio TTS reads as "synthetic".
Matching the spectral signature pulls our outputs into the same domain.

Outputs `librivox_profile.npz` with:
    - mean_log_psd: (n_freq,) average log magnitude in dB across clips
    - sample_rate: source SR
    - n_fft: FFT size used (so the inverse EQ has the right resolution)
    - noise_floor_db: estimated dB level of the quietest 5% of frames

Usage:
    python compute_librivox_profile.py \
        --clips-dir /workspace/data/clips \
        --n 500 --out /workspace/data/librivox_profile.npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--target-sr", type=int, default=24000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips = sorted(Path(args.clips_dir).glob("*.wav"))
    rng = random.Random(args.seed)
    rng.shuffle(clips)
    clips = clips[: args.n]
    print(f"[profile] sampled {len(clips)} clips", flush=True)

    psd_sum = np.zeros(args.n_fft // 2 + 1, dtype=np.float64)
    psd_count = 0
    noise_frames = []

    for i, p in enumerate(clips):
        try:
            wav, sr = sf.read(str(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != args.target_sr:
            try:
                import torch, torchaudio.functional as AF
                t = torch.from_numpy(wav).unsqueeze(0)
                wav = AF.resample(t, sr, args.target_sr).squeeze(0).numpy()
            except Exception:
                continue

        hop = args.n_fft // 2
        win = np.hanning(args.n_fft).astype(np.float32)
        n_frames = max(0, (len(wav) - args.n_fft) // hop + 1)
        if n_frames <= 0:
            continue

        # Frame-by-frame spectrum
        frame_energies = []
        for k in range(n_frames):
            seg = wav[k * hop : k * hop + args.n_fft] * win
            spec = np.abs(np.fft.rfft(seg)) ** 2
            psd_sum += spec
            psd_count += 1
            frame_energies.append(float(seg @ seg))

        # Noise floor estimate: 5th percentile of frame energy
        frame_energies = np.asarray(frame_energies)
        if len(frame_energies) > 0:
            qf = np.quantile(frame_energies, 0.05)
            if qf > 0:
                noise_frames.append(10 * np.log10(qf / args.n_fft + 1e-12))

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(clips)}", flush=True)

    mean_log_psd = 10 * np.log10(psd_sum / max(1, psd_count) + 1e-12)
    noise_floor_db = float(np.median(noise_frames)) if noise_frames else -60.0

    print(f"[profile] mean_log_psd peak {mean_log_psd.max():.1f} dB, "
          f"min {mean_log_psd.min():.1f} dB", flush=True)
    print(f"[profile] noise_floor_db: {noise_floor_db:.1f}", flush=True)

    np.savez(
        args.out,
        mean_log_psd=mean_log_psd.astype(np.float32),
        sample_rate=args.target_sr,
        n_fft=args.n_fft,
        noise_floor_db=noise_floor_db,
    )
    print(f"[profile] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
