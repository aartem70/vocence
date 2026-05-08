"""Run GPT-4o-audio on each clip with the EXACT prompt the judge uses.

Mirrors vocence/pipeline/evaluation.py:
- DESCRIPTION_SYSTEM (verbatim)
- JSON schema (transcription + 7 trait enums)
- _normalize_trait_value coercion via _TRAIT_ALIASES
- AudioJudge.judge_audio_pointwise calling pattern (single audio, system prompt only,
  no_concatenation, temperature=0.0, max_tokens=500)

This guarantees label distribution match between our training data and what
judges score against at inference time.

Usage:
    OPENAI_API_KEY=... python extract_specs.py --clips-dir ./clips --out specs.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import openai

# ============================================================
# CANONICAL — copied verbatim from vocence/pipeline/evaluation.py
# Do not paraphrase: same labels = same training distribution.
# ============================================================
VOICE_TRAIT_ENUMS: dict[str, list[str]] = {
    "gender":    ["male", "female", "neutral"],
    "pitch":     ["low", "mid", "high"],
    "speed":     ["slow", "normal", "fast"],
    "age_group": ["child", "young_adult", "adult", "senior"],
    "emotion":   ["neutral", "happy", "sad", "angry", "calm", "excited", "serious", "fearful"],
    "tone":      ["warm", "cold", "friendly", "formal", "casual", "authoritative"],
    "accent":    ["us", "uk", "au", "in", "neutral", "other"],
}

_FALLBACK_TRAITS: dict[str, str] = {
    "transcription": "",
    "gender":    "neutral", "pitch": "mid", "speed": "normal",
    "age_group": "adult",   "emotion": "neutral", "tone": "casual", "accent": "neutral",
}

_TRAIT_ALIASES: dict[str, dict[str, str]] = {
    "gender":    {"unknown": "neutral", "nonbinary": "neutral", "non_binary": "neutral"},
    "pitch":     {"normal": "mid", "medium": "mid"},
    "speed":     {"medium": "normal"},
    "age_group": {"teenager": "young_adult", "twenties": "young_adult",
                  "thirties": "adult", "forties": "adult", "fifties": "adult",
                  "sixties": "senior", "seventies": "senior", "eighties": "senior",
                  "nineties": "senior", "unknown": "adult"},
    "emotion":   {"bored": "neutral"},
    "tone":      {"neutral": "casual"},
    "accent":    {"american": "us", "british": "uk", "australian": "au", "indian": "in",
                  "english": "uk", "unknown": "neutral"},
}

DESCRIPTION_SYSTEM = """You are an expert at analyzing speech for text-to-speech evaluation.
Analyze the audio and return a JSON object with these exact keys. For each categorical trait you MUST pick exactly one value from the listed options.

- transcription: the exact words spoken, lowercased, punctuation preserved (string)
- gender: one of [male, female, neutral]
- pitch: one of [low, mid, high]
- speed: one of [slow, normal, fast]
- age_group: one of [child, young_adult, adult, senior]
- emotion: one of [neutral, happy, sad, angry, calm, excited, serious, fearful]
- tone: one of [warm, cold, friendly, formal, casual, authoritative]
- accent: one of [us, uk, au, in, neutral, other]

Return ONLY valid JSON, no markdown, no commentary. Every value must be one of the listed options exactly as written.

Example:
{"transcription": "hello world", "gender": "male", "pitch": "mid", "speed": "normal", "age_group": "adult", "emotion": "neutral", "tone": "casual", "accent": "us"}"""


def normalize_trait(key: str, value: Any) -> str:
    if value is None:
        return _FALLBACK_TRAITS[key]
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = _TRAIT_ALIASES.get(key, {})
    v = aliases.get(v, v)
    return v if v in VOICE_TRAIT_ENUMS[key] else _FALLBACK_TRAITS[key]


def parse_traits_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    out: dict[str, Any] = {"transcription": str(parsed.get("transcription") or "").strip()}
    for k in VOICE_TRAIT_ENUMS:
        out[k] = normalize_trait(k, parsed.get(k))
    return out


def encode_audio_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


async def call_gpt4o_audio(client: openai.AsyncOpenAI, audio_b64: str) -> str:
    """Mirror AudioJudge.judge_audio_pointwise(no_concatenation, temperature=0, max_tokens=500)."""
    resp = await client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {"role": "system", "content": DESCRIPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""


async def process_clip(sem: asyncio.Semaphore, client: openai.AsyncOpenAI,
                        clip_path: Path, clip_id: str) -> dict | None:
    async with sem:
        try:
            audio_b64 = encode_audio_b64(clip_path)
            text = await call_gpt4o_audio(client, audio_b64)
            traits = parse_traits_response(text)
            return {"clip_id": clip_id, "wav_path": clip_path.name, **traits, "raw_response": text}
        except Exception as e:
            print(f"[extract_specs]   {clip_id}: error {e}", file=sys.stderr)
            return None


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    out_path = Path(args.out)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    # Resume support: skip clip_ids already in out_path.
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    print(f"[extract_specs] {len(done_ids)} already done")

    clips = sorted(clips_dir.glob("*.wav"))
    if args.limit:
        clips = clips[: args.limit]
    todo = [(c.stem, c) for c in clips if c.stem not in done_ids]
    print(f"[extract_specs] {len(todo)} clips to process (concurrency={args.concurrency})")

    client = openai.AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    out_f = out_path.open("a")
    try:
        tasks = [asyncio.create_task(process_clip(sem, client, p, cid)) for cid, p in todo]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            r = await task
            if r:
                out_f.write(json.dumps(r) + "\n")
                out_f.flush()
            if i % 10 == 0:
                print(f"[extract_specs]   {i}/{len(todo)}")
    finally:
        out_f.close()


if __name__ == "__main__":
    asyncio.run(main_async())
