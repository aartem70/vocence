"""Local evaluation that mirrors the judge's full scoring pipeline.

For each (source_clip, spec) pair from the held-out set:
  1. Generate audio via our model (local Qwen3-TTS or remote container)
  2. Pointwise extraction (GPT-4o-audio + DESCRIPTION_SYSTEM) → traits
  3. Pairwise naturalness vs source (GPT-4o-audio + NATURALNESS_SYSTEM_TEMPLATE) → FIRST/SECOND
  4. Score 9 elements with the judge's exact weights
  5. Aggregate: # passing 0.9 threshold = predicted binary win rate

Use this BEFORE deploying any new container / model revision.

Cost: ~$0.08/sample (2 GPT-4o-audio calls). 50 samples ≈ $4.

Usage:
    # Local model (on GPU box)
    python local_eval.py --backend local --model-path /path/to/checkpoint \
        --specs /path/to/held_out_specs.jsonl --clips-dir /path/to/clips --n 50

    # Deployed container
    python local_eval.py --backend container --container-url https://USER-NAME.containers.ai \
        --specs ... --clips-dir ... --n 50
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

# ============================================================
# Judge constants — copied verbatim from vocence/pipeline/evaluation.py
# ============================================================
VOICE_TRAIT_ENUMS = {
    "gender": ["male", "female", "neutral"],
    "pitch": ["low", "mid", "high"],
    "speed": ["slow", "normal", "fast"],
    "age_group": ["child", "young_adult", "adult", "senior"],
    "emotion": ["neutral", "happy", "sad", "angry", "calm", "excited", "serious", "fearful"],
    "tone": ["warm", "cold", "friendly", "formal", "casual", "authoritative"],
    "accent": ["us", "uk", "au", "in", "neutral", "other"],
}
ORDINAL_TRAITS = {"pitch", "speed", "age_group"}
_FALLBACK = {"transcription": "", "gender": "neutral", "pitch": "mid", "speed": "normal",
             "age_group": "adult", "emotion": "neutral", "tone": "casual", "accent": "neutral"}
_ALIAS = {
    "gender": {"unknown": "neutral", "nonbinary": "neutral", "non_binary": "neutral"},
    "pitch": {"normal": "mid", "medium": "mid"},
    "speed": {"medium": "normal"},
    "age_group": {"teenager": "young_adult", "twenties": "young_adult",
                  "thirties": "adult", "forties": "adult", "fifties": "adult",
                  "sixties": "senior", "seventies": "senior", "eighties": "senior",
                  "nineties": "senior", "unknown": "adult"},
    "emotion": {"bored": "neutral"},
    "tone": {"neutral": "casual"},
    "accent": {"american": "us", "british": "uk", "australian": "au", "indian": "in",
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

NATURALNESS_SYSTEM_TEMPLATE = """You are an audio naturalness judge. You will hear two audio clips. Both are TTS-task outputs for the same task:

{task_description}

Question: which clip sounds MORE NATURAL as human speech? Consider clarity, prosody, intonation, and absence of robotic or synthetic artifacts. Do not consider content correctness — only naturalness.

Respond with exactly one word on the first line: FIRST or SECOND
- FIRST = the first clip is more natural
- SECOND = the second clip is more natural
Optionally add a short reason on the next line."""

_RAW_WEIGHTS = {
    "script": 0.30, "naturalness": 0.15,
    "gender": 0.10, "speed": 0.10, "emotion": 0.10, "age_group": 0.10,
    "pitch": 0.05, "accent": 0.05, "tone": 0.05,
}
_W_TOTAL = sum(_RAW_WEIGHTS.values())
ELEMENT_WEIGHTS = {k: v / _W_TOTAL for k, v in _RAW_WEIGHTS.items()}
PASS_THRESHOLD = 0.9

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_trait(key, val):
    if val is None:
        return _FALLBACK[key]
    v = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    v = _ALIAS.get(key, {}).get(v, v)
    return v if v in VOICE_TRAIT_ENUMS[key] else _FALLBACK[key]


def parse_traits(text):
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
    out = {"transcription": str(parsed.get("transcription") or "").strip()}
    for k in VOICE_TRAIT_ENUMS:
        out[k] = normalize_trait(k, parsed.get(k))
    return out


def wer(ref, hyp):
    r = _WORD_RE.findall((ref or "").lower())
    h = _WORD_RE.findall((hyp or "").lower())
    if not r:
        return 1.0 if h else 0.0
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return min(1.0, dp[n][m] / n)


def score_element(key, expected, actual):
    if key == "script":
        return max(0.0, 1.0 - wer(expected, actual))
    if key in ORDINAL_TRAITS:
        try:
            i = VOICE_TRAIT_ENUMS[key].index(expected)
            j = VOICE_TRAIT_ENUMS[key].index(actual)
            d = abs(i - j)
            return 1.0 if d == 0 else (0.5 if d == 1 else 0.0)
        except ValueError:
            return 0.0
    return 1.0 if expected == actual else 0.0


def parse_natural_first_or_second(text):
    """Judge parses the first line of the response. FIRST or SECOND."""
    if not text:
        return None
    first_line = text.strip().split("\n", 1)[0].strip().upper()
    if "FIRST" in first_line and "SECOND" not in first_line:
        return "FIRST"
    if "SECOND" in first_line and "FIRST" not in first_line:
        return "SECOND"
    # try first word
    w = re.findall(r"[A-Z]+", first_line)
    if w and w[0] in ("FIRST", "SECOND"):
        return w[0]
    return None


# ============================================================
# Audio I/O
# ============================================================
def wav_bytes_to_path(wav_bytes, tmp_dir, name):
    p = Path(tmp_dir) / f"{name}.wav"
    p.write_bytes(wav_bytes)
    return str(p)


def encode_audio_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ============================================================
# Backends: local (Qwen3-TTS) or container
# ============================================================
class LocalBackend:
    def __init__(self, model_path, dtype="bf16"):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, device_map="cuda:0",
            dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
            attn_implementation="sdpa",
        )

    def synthesize(self, text, instruction):
        kwargs = dict(
            text=text, instruct=instruction, language="English",
            max_new_tokens=600, do_sample=True,
            temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05,
        )
        for drop in ([], ["max_new_tokens"], ["max_new_tokens", "top_k"],
                     ["max_new_tokens", "top_k", "repetition_penalty"]):
            try:
                trim = {k: v for k, v in kwargs.items() if k not in drop}
                waves, sr = self.model.generate_voice_design(**trim)
                first = waves[0] if isinstance(waves, (list, tuple)) else waves
                arr = np.asarray(first, dtype=np.float32).squeeze()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                return arr, int(sr)
            except TypeError:
                continue
        raise RuntimeError("all generate_voice_design kwarg combinations failed")


class RemoteBackend:
    def __init__(self, base_url, api_key):
        self.url = base_url.rstrip("/") + "/speak"
        self.api_key = api_key

    async def synthesize(self, text, instruction, client):
        r = await client.post(
            self.url, headers={"Authorization": self.api_key},
            json={"text": text, "instruction": instruction}, timeout=300,
        )
        r.raise_for_status()
        wav_bytes = r.content
        wav, sr = sf.read(io.BytesIO(wav_bytes))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)


# ============================================================
# OpenAI judge calls
# ============================================================
async def call_pointwise(client, wav_path, sem):
    async with sem:
        b64 = encode_audio_b64(wav_path)
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                     "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-audio-preview",
                "modalities": ["text"],
                "messages": [
                    {"role": "system", "content": DESCRIPTION_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ]},
                ],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=120,
        )
        r.raise_for_status()
        return parse_traits(r.json()["choices"][0]["message"]["content"])


async def call_pairwise(client, source_path, engine_path, task_description, sem):
    async with sem:
        swap = random.choice([True, False])
        if swap:
            first_path, second_path = engine_path, source_path
            engine_is = "FIRST"
        else:
            first_path, second_path = source_path, engine_path
            engine_is = "SECOND"
        prompt = NATURALNESS_SYSTEM_TEMPLATE.format(task_description=task_description or "")
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                     "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-audio-preview",
                "modalities": ["text"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {
                            "data": encode_audio_b64(first_path), "format": "wav"}},
                        {"type": "input_audio", "input_audio": {
                            "data": encode_audio_b64(second_path), "format": "wav"}},
                    ]},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
        verdict = parse_natural_first_or_second(out)
        if verdict is None:
            return {"engine_more_natural": False, "raw": out, "presentation": engine_is}
        return {
            "engine_more_natural": verdict == engine_is,
            "raw": out, "presentation": engine_is,
        }


def build_instruction_qwen3(spec):
    """Mirror ratrys's _structured_to_natural for Qwen3-TTS."""
    age_phrases = {"child": "child", "young_adult": "young adult", "adult": "adult", "senior": "senior"}
    speed_advs = {"slow": "slowly", "normal": "at a normal pace", "fast": "quickly"}
    emo_advs = {"neutral": "in a neutral manner", "happy": "happily", "sad": "sadly",
                "angry": "angrily", "calm": "calmly", "excited": "excitedly",
                "serious": "seriously", "fearful": "fearfully"}
    accent_names = {"us": "American", "uk": "British", "au": "Australian", "in": "Indian",
                    "neutral": "neutral", "other": "neutral"}

    def _a(w):
        return "an" if w[:1].lower() in "aeiou" else "a"

    age = age_phrases.get(spec.get("age_group", "adult"), "adult")
    gender = spec.get("gender", "neutral")
    tone = spec.get("tone", "casual")
    pitch = spec.get("pitch", "mid")
    speed = speed_advs.get(spec.get("speed", "normal"), "at a normal pace")
    emotion = emo_advs.get(spec.get("emotion", "neutral"), "in a neutral manner")
    accent = accent_names.get(spec.get("accent", "neutral"), "neutral")
    return (f"{_a(age).capitalize()} {age} {gender} speaker with a {tone} tone speaks "
            f"{speed} and {emotion} at a {pitch} pitch, with {_a(accent)} {accent} accent.")


def build_instruction_validator_format(spec):
    """Judge-format pipe-separated string (what /speak actually receives)."""
    parts = [f"{k}: {spec.get(k, _FALLBACK.get(k, ''))}"
             for k in ("gender", "pitch", "speed", "age_group", "emotion", "tone", "accent")]
    return " | ".join(parts)


# ============================================================
# Eval driver
# ============================================================
async def eval_one(client, backend, sample, tmp_dir, sem_judge):
    spec = {k: sample.get(k, _FALLBACK.get(k, "")) for k in
            ("gender", "pitch", "speed", "age_group", "emotion", "tone", "accent")}
    text = sample["transcription"][:800]
    source_wav = Path(sample["source_wav"])
    if not source_wav.exists():
        return {"clip_id": sample.get("clip_id"), "error": "source missing"}

    # Backend-specific instruction format
    if isinstance(backend, RemoteBackend):
        # Container receives judge format; container's engine.py converts internally
        instruction = build_instruction_validator_format(spec)
        wav, sr = await backend.synthesize(text, instruction, client)
    else:
        # Local model: pass already-translated natural-language description
        instruction = build_instruction_qwen3(spec)
        wav, sr = backend.synthesize(text, instruction)

    engine_path = Path(tmp_dir) / f"engine_{sample.get('clip_id', 'x')}.wav"
    sf.write(engine_path, wav, sr)

    # Two GPT-4o calls in parallel: pointwise + pairwise
    task_desc = build_instruction_validator_format(spec) + " | text: " + text[:120]
    pw, nat = await asyncio.gather(
        call_pointwise(client, str(engine_path), sem_judge),
        call_pairwise(client, str(source_wav), str(engine_path), task_desc, sem_judge),
    )

    # Score
    scores = {
        "script": score_element("script", text, pw["transcription"]),
        "gender": score_element("gender", spec["gender"], pw["gender"]),
        "pitch": score_element("pitch", spec["pitch"], pw["pitch"]),
        "speed": score_element("speed", spec["speed"], pw["speed"]),
        "age_group": score_element("age_group", spec["age_group"], pw["age_group"]),
        "emotion": score_element("emotion", spec["emotion"], pw["emotion"]),
        "tone": score_element("tone", spec["tone"], pw["tone"]),
        "accent": score_element("accent", spec["accent"], pw["accent"]),
        "naturalness": 1.0 if nat["engine_more_natural"] else 0.0,
    }
    weighted = sum(scores[k] * ELEMENT_WEIGHTS[k] for k in scores)
    return {
        "clip_id": sample.get("clip_id"),
        "scores": scores,
        "weighted": weighted,
        "win": weighted >= PASS_THRESHOLD,
        "extracted": pw,
        "expected": {**spec, "transcription": text},
        "naturalness_raw": nat.get("raw", "")[:120],
    }


async def run(args):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is required")

    # Load held-out specs (random subset of N from --specs)
    specs = []
    for line in Path(args.specs).open():
        try:
            s = json.loads(line)
            if not s.get("transcription"):
                continue
            wav_path = Path(args.clips_dir) / s["wav_path"]
            if not wav_path.exists():
                continue
            s["source_wav"] = str(wav_path)
            specs.append(s)
        except json.JSONDecodeError:
            continue
    rng = random.Random(args.seed)
    rng.shuffle(specs)
    samples = specs[: args.n]
    print(f"[eval] running on {len(samples)} samples")

    # Backend
    if args.backend == "local":
        if not args.model_path:
            sys.exit("--model-path required for local backend")
        backend = LocalBackend(args.model_path)
    else:
        if not args.container_url or not os.environ.get("REMOTE_API_KEY"):
            sys.exit("--container-url and REMOTE_API_KEY required for container backend")
        backend = RemoteBackend(args.container_url, os.environ["REMOTE_API_KEY"])

    # Run
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    sem_judge = asyncio.Semaphore(args.judge_concurrency)
    async with httpx.AsyncClient() as client:
        results = []
        for i, sample in enumerate(samples):
            t0 = time.time()
            try:
                r = await eval_one(client, backend, sample, args.tmp_dir, sem_judge)
            except Exception as e:
                r = {"clip_id": sample.get("clip_id"), "error": str(e)[:200]}
            r["elapsed"] = time.time() - t0
            results.append(r)
            if "win" in r:
                print(f"[eval] {i+1}/{len(samples)} weighted={r['weighted']:.3f} win={r['win']} dt={r['elapsed']:.1f}s")
            else:
                print(f"[eval] {i+1}/{len(samples)} ERROR: {r.get('error')} dt={r['elapsed']:.1f}s")

    # Aggregate
    valid = [r for r in results if "win" in r]
    if not valid:
        print("\nNo successful evals — check errors")
        Path(args.out).write_text(json.dumps(results, indent=2))
        return

    win_rate = sum(1 for r in valid if r["win"]) / len(valid)
    mean_weighted = sum(r["weighted"] for r in valid) / len(valid)

    print(f"\n=== AGGREGATE ({len(valid)} successful evals) ===")
    print(f"  predicted binary win rate: {win_rate:.1%}")
    print(f"    deploy decision: {'GO — clears ratrys + 2pp' if win_rate >= 0.65 else ('marginal — below 65% bar' if win_rate >= 0.55 else 'NO — iterate first')}")
    print(f"  mean weighted score:       {mean_weighted:.3f}  (judge passes at >= 0.9)")
    print()
    print("  per-element mean (weight-normalized):")
    for k in ELEMENT_WEIGHTS:
        m = sum(r["scores"][k] for r in valid) / len(valid)
        print(f"    {k:12s} {m:.3f}  (weight={ELEMENT_WEIGHTS[k]:.3f}, raw_contribution={m*ELEMENT_WEIGHTS[k]:.3f})")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  full results written to: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "container"], required=True)
    ap.add_argument("--model-path", help="local checkpoint dir (--backend local)")
    ap.add_argument("--container-url", help="https://USER-NAME.containers.ai (--backend container)")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--tmp-dir", default="/tmp/local_eval")
    ap.add_argument("--out", default="/tmp/local_eval_results.json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
