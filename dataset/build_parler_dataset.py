"""Build a Parler-TTS-format HF Dataset from clips + spec JSONL.

Output features:
    - description: str  (natural language built from 8 trait enums; SAME mapping as engine.py)
    - prompt:      str  (transcription)
    - audio:       Audio (path + sampling rate; HF Datasets handles the load lazily)

Description mapping is byte-identical to vocence-engine-v1's engine.py so train and inference
share the same input distribution. Tweak _build_description carefully — any change here means
re-formatting train data and pushing a new engine.py revision.

Usage:
    python build_parler_dataset.py \
        --clips-dir ./clips --specs ./specs.jsonl --out ./parler_dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

# ===== copied from engine.py — keep in sync =====
_GENDER = {"male": "male", "female": "female", "neutral": "androgynous"}
_PITCH = {"low": "low-pitched", "mid": "moderately pitched", "high": "high-pitched"}
_SPEED = {"slow": "slowly", "normal": "at a moderate pace", "fast": "quickly"}
_AGE = {"child": "child", "young_adult": "young adult", "adult": "middle-aged", "senior": "elderly"}
_EMOTION = {
    "neutral": "neutral", "happy": "happy and cheerful",
    "sad": "sad and melancholic", "angry": "angry and intense",
    "calm": "calm", "excited": "excited and enthusiastic",
    "serious": "serious", "fearful": "anxious and fearful",
}
_TONE = {
    "warm": "warm", "cold": "cold and detached", "friendly": "friendly",
    "formal": "formal", "casual": "casual", "authoritative": "authoritative",
}
_ACCENT = {
    "us": "with an American accent", "uk": "with a British accent",
    "au": "with an Australian accent", "in": "with an Indian accent",
    "neutral": "with a neutral accent", "other": "",
}


def build_description(spec: dict[str, str]) -> str:
    gender = _GENDER.get(spec.get("gender", "neutral"), "neutral")
    pitch = _PITCH.get(spec.get("pitch", "mid"), "moderately pitched")
    speed = _SPEED.get(spec.get("speed", "normal"), "at a moderate pace")
    age = _AGE.get(spec.get("age_group", "adult"), "middle-aged")
    emotion = _EMOTION.get(spec.get("emotion", "neutral"), "neutral")
    tone = _TONE.get(spec.get("tone", "warm"), "warm")
    accent = _ACCENT.get(spec.get("accent", "neutral"), "")
    accent_clause = f" {accent}" if accent else ""
    return (
        f"A {age} {gender} speaker with a {pitch}, {tone} voice "
        f"delivers the words {speed} in a {emotion} manner{accent_clause}. "
        "The recording is very high quality, clean, with no background noise, "
        "and sounds like natural human speech."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-sr", type=int, default=44100,
                        help="(unused — collator resamples on the fly; kept for compat)")
    parser.add_argument("--test-size", type=float, default=0.05)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    specs_path = Path(args.specs)
    out_path = Path(args.out)

    rows: list[dict] = []
    for line in specs_path.open():
        s = json.loads(line)
        wav = clips_dir / s["wav_path"]
        if not wav.exists():
            continue
        if not s.get("transcription"):
            continue
        spec = {k: s.get(k, "") for k in
                ("gender", "pitch", "speed", "age_group", "emotion", "tone", "accent")}
        rows.append({
            "audio_path": str(wav),
            "prompt": s["transcription"].strip(),
            "description": build_description(spec),
            "spec": spec,
        })
    print(f"[build_dataset] {len(rows)} usable rows")

    ds = Dataset.from_list(rows)
    if args.test_size and len(rows) > 20:
        ds = ds.train_test_split(test_size=args.test_size, seed=42)
    ds.save_to_disk(str(out_path))
    print(f"[build_dataset] saved to {out_path}")
    print(f"[build_dataset] sample row: {rows[0]}")


if __name__ == "__main__":
    main()
