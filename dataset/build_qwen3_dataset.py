"""Build a Qwen3-TTS-VoiceDesign-format HF Dataset from clips + spec JSONL.

Description format mirrors ratrys/sft-tts-800's `_structured_to_natural` exactly,
so our fine-tune learns the same instruction style Qwen3-TTS-VoiceDesign was
pretrained to follow:

    "A {age} {gender} speaker with a {tone} tone speaks {speed_adv} and
     {emotion_adv} at a {pitch} pitch, with a {accent} accent."

Output features:
    - audio_path: str   (local wav path; collator loads with soundfile)
    - prompt:     str   (text to speak)
    - description: str  (Qwen3-TTS instruction string)
    - spec:       dict  (raw 8-trait dict, kept for diagnostics)

Usage:
    python build_qwen3_dataset.py --clips-dir ./clips --specs ./specs.jsonl --out ./qwen3_dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

# ===== mirror of ratrys/sft-tts-800's engine.py _structured_to_natural =====
_SPEED_ADVERBS = {
    "slow": "slowly",
    "normal": "at a normal pace",
    "fast": "quickly",
}
_EMOTION_ADVERBS = {
    "neutral": "in a neutral manner",
    "happy": "happily",
    "sad": "sadly",
    "angry": "angrily",
    "calm": "calmly",
    "excited": "excitedly",
    "serious": "seriously",
    "fearful": "fearfully",
}
_ACCENT_NAMES = {
    "us": "American",
    "uk": "British",
    "au": "Australian",
    "in": "Indian",
    "neutral": "neutral",
    "other": "neutral",
}
_AGE_PHRASES = {
    "child": "child",
    "young_adult": "young adult",
    "adult": "adult",
    "senior": "senior",
}


def _a(word: str) -> str:
    """Return 'a' or 'an' based on first letter."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def build_description(spec: dict[str, str]) -> str:
    """Mirror ratrys's _structured_to_natural exactly."""
    age = _AGE_PHRASES.get(spec.get("age_group", "adult"), "adult")
    gender = spec.get("gender", "neutral")
    tone = spec.get("tone", "casual")
    pitch = spec.get("pitch", "mid")
    speed_adv = _SPEED_ADVERBS.get(spec.get("speed", "normal"), "at a normal pace")
    emotion_adv = _EMOTION_ADVERBS.get(spec.get("emotion", "neutral"), "in a neutral manner")
    accent = _ACCENT_NAMES.get(spec.get("accent", "neutral"), "neutral")
    return (
        f"{_a(age).capitalize()} {age} {gender} speaker with a {tone} tone speaks "
        f"{speed_adv} and {emotion_adv} at a {pitch} pitch, with {_a(accent)} {accent} accent."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-size", type=float, default=0.05)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    rows: list[dict] = []
    for line in Path(args.specs).open():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        wav = clips_dir / s["wav_path"]
        if not wav.exists() or not s.get("transcription", "").strip():
            continue
        spec = {k: s.get(k, "") for k in
                ("gender", "pitch", "speed", "age_group", "emotion", "tone", "accent")}
        rows.append({
            "audio_path": str(wav),
            "prompt": s["transcription"].strip(),
            "description": build_description(spec),
            "spec": spec,
        })
    print(f"[build_qwen3] {len(rows)} usable rows")
    if rows:
        print(f"[build_qwen3] sample description: {rows[0]['description']!r}")

    ds = Dataset.from_list(rows)
    if args.test_size and len(rows) > 20:
        ds = ds.train_test_split(test_size=args.test_size, seed=42)
    ds.save_to_disk(args.out)
    print(f"[build_qwen3] saved to {args.out}")


if __name__ == "__main__":
    main()
