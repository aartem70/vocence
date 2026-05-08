"""local_eval with best-of-N + UTMOSv2/faster-whisper composite scoring (mirrors production engine.py)."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

# Import everything from local_eval, then override LocalBackend
sys.path.insert(0, str(Path(__file__).parent))
import local_eval as base


class CompositeScorer:
    """0.3 * UTMOSv2/5 + 0.7 * (1 - faster-whisper WER); same as production engine.py."""

    def __init__(self):
        self._utmos = None
        try:
            import torch
            import utmosv2
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._utmos = utmosv2.create_model(pretrained=True, device=device)
            print("[scorer] UTMOSv2 ready", flush=True)
        except Exception as e:
            print(f"[scorer] UTMOSv2 unavailable: {type(e).__name__}: {e}", flush=True)

        self._whisper = None
        try:
            import torch
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._whisper = WhisperModel("base", device=device, compute_type=compute)
            print("[scorer] faster-whisper ready", flush=True)
        except Exception as e:
            print(f"[scorer] faster-whisper unavailable: {type(e).__name__}: {e}", flush=True)

    def _utmos_score(self, wav, sr):
        if self._utmos is None:
            return 0.5
        try:
            arr = np.ascontiguousarray(wav, dtype=np.float32)
            mos = self._utmos.predict(data=arr, sr=int(sr))
            if hasattr(mos, "item"):
                mos = float(mos.item() if mos.ndim == 0 else mos.flatten()[0].item())
            elif isinstance(mos, np.ndarray):
                mos = float(mos.flatten()[0])
            else:
                mos = float(mos)
            return max(0.0, min(1.0, mos / 5.0))
        except Exception:
            return 0.5

    def _whisper_wer(self, wav, sr, target_text):
        if self._whisper is None or not target_text.strip():
            return 0.5
        try:
            if sr != 16000:
                import torch
                import torchaudio.functional as AF
                wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
                wav = AF.resample(wav_t, sr, 16000).squeeze(0).numpy()
            segments, _ = self._whisper.transcribe(wav, language="en", beam_size=1)
            hyp = " ".join(seg.text for seg in segments).strip()
            return max(0.0, 1.0 - base.wer(target_text, hyp))
        except Exception:
            return 0.5

    def score(self, wav, sr, target_text):
        return 0.3 * self._utmos_score(wav, sr) + 0.7 * self._whisper_wer(wav, sr, target_text)


class LocalBackendBoN(base.LocalBackend):
    """Generates N candidates, picks the highest composite score, optionally post-processes."""

    def __init__(self, model_path, num_candidates=5, dtype="bf16", postprocess=True,
                 eq_profile_path=None, mp3_bitrate=None, noise_floor_db=None,
                 eq_strength=0.7):
        super().__init__(model_path, dtype=dtype)
        self._n = max(1, int(num_candidates))
        self._scorer = CompositeScorer()
        self._postprocess_enabled = bool(postprocess)
        self._postprocess_fn = None
        self._pp_kwargs = {
            "eq_profile_path": eq_profile_path,
            "eq_strength": float(eq_strength),
            "mp3_bitrate_kbps": mp3_bitrate,
            "noise_floor_db": noise_floor_db,
        }
        if self._postprocess_enabled:
            try:
                from audio_postprocess import postprocess as _pp
                self._postprocess_fn = _pp
                print(f"[backend] postproc enabled: eq={eq_profile_path} mp3={mp3_bitrate} noise={noise_floor_db}", flush=True)
            except Exception as e:
                print(f"[backend] post-processing disabled: {type(e).__name__}: {e}", flush=True)
                self._postprocess_enabled = False
        print(f"[backend] best-of-{self._n} ready", flush=True)

    def synthesize(self, text, instruction):
        kwargs = dict(
            text=text, instruct=instruction, language="English",
            max_new_tokens=600, do_sample=True,
            temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05,
        )
        candidates = []
        for i in range(self._n):
            for drop in ([], ["max_new_tokens"], ["max_new_tokens", "top_k"],
                         ["max_new_tokens", "top_k", "repetition_penalty"]):
                try:
                    trim = {k: v for k, v in kwargs.items() if k not in drop}
                    waves, sr = self.model.generate_voice_design(**trim)
                    first = waves[0] if isinstance(waves, (list, tuple)) else waves
                    arr = np.asarray(first, dtype=np.float32).squeeze()
                    if arr.ndim > 1:
                        arr = arr.mean(axis=0)
                    score = self._scorer.score(arr, int(sr), text)
                    candidates.append((score, arr, int(sr)))
                    break
                except TypeError:
                    continue
        if not candidates:
            raise RuntimeError("all generate attempts failed")
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_arr, best_sr = candidates[0]
        scores_list = [round(c[0], 3) for c in candidates]
        print(f"[backend] cand_scores={scores_list} picked={best_score:.3f}", flush=True)
        if self._postprocess_enabled and self._postprocess_fn is not None:
            best_arr, best_sr = self._postprocess_fn(best_arr, best_sr, **self._pp_kwargs)
        return best_arr, best_sr


# Override base module's LocalBackend reference so run() picks up our subclass.
base.LocalBackend = LocalBackendBoN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "container"], required=True)
    ap.add_argument("--model-path", help="local checkpoint dir")
    ap.add_argument("--container-url")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--tmp-dir", default="/tmp/local_eval_bo5")
    ap.add_argument("--out", default="/tmp/local_eval_bo5_results.json")
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--no-postprocess", action="store_true",
                    help="disable LUFS/bandwidth post-processing on the picked candidate")
    ap.add_argument("--eq-profile", default=None,
                    help="path to LibriVox spectrum profile .npz for matching EQ")
    ap.add_argument("--eq-strength", type=float, default=0.7)
    ap.add_argument("--mp3-bitrate", type=int, default=None,
                    help="if set, mp3 round-trip at this kbps (e.g. 96)")
    ap.add_argument("--noise-floor-db", type=float, default=None,
                    help="if set, inject pink noise at this dB level (e.g. -72)")
    args = ap.parse_args()

    # Patch LocalBackendBoN __init__ to receive runtime params from CLI.
    nc = args.num_candidates
    pp = not args.no_postprocess
    eq_profile = args.eq_profile
    eq_strength = args.eq_strength
    mp3_br = args.mp3_bitrate
    noise_db = args.noise_floor_db
    orig_init = LocalBackendBoN.__init__

    def _init(self, model_path, dtype="bf16"):
        orig_init(self, model_path, num_candidates=nc, dtype=dtype, postprocess=pp,
                  eq_profile_path=eq_profile, eq_strength=eq_strength,
                  mp3_bitrate=mp3_br, noise_floor_db=noise_db)

    LocalBackendBoN.__init__ = _init

    asyncio.run(base.run(args))


if __name__ == "__main__":
    main()
