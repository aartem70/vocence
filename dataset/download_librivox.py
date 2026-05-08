"""Download LibriVox chapters and extract 20-25s WAV clips that match the judge's
source-audio distribution exactly.

Mirrors vocence/gateway/http/service/tasks/source_audio_downloader.py:
- LibriVox API for English audiobooks
- pick random chapter with enough duration
- ffmpeg extract LIBRIVOX_CLIPS_PER_CHAPTER (default 10) clips of 20-25s each
- output: 22050 Hz mono PCM WAV (judge's exact format)

Usage:
    python download_librivox.py --target-clips 200 --out-dir ./clips
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

LIBRIVOX_API = "https://librivox.org/api/feed/audiobooks"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

CLIP_MIN_SEC = 20
CLIP_MAX_SEC = 25
CLIPS_PER_CHAPTER = 10
FFMPEG_DURATION_MARGIN_SEC = 0.5


def fetch_audiobooks(limit: int = 50, offset: int = 0) -> list[dict]:
    url = f"{LIBRIVOX_API}/?limit={limit}&offset={offset}&format=json&extended=1"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data).get("books") or []
    except Exception as e:
        print(f"  fetch_audiobooks(offset={offset}) failed: {e}")
        return []


def playtime_sec(section: dict) -> float:
    try:
        return float(section.get("playtime", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def pick_random_chapter(rng: random.Random, min_duration_sec: float) -> tuple[dict, dict] | None:
    for _ in range(20):
        offset = rng.randint(0, 500) * 50
        books = fetch_audiobooks(limit=50, offset=offset)
        if not books and offset > 0:
            books = fetch_audiobooks(limit=50, offset=0)
        if not books:
            continue
        books_en = [b for b in books if (b.get("language") or "").strip().lower() == "english"]
        if not books_en:
            continue
        book = rng.choice(books_en)
        sections = book.get("sections") or []
        long_enough = [s for s in sections if playtime_sec(s) >= min_duration_sec]
        if not long_enough:
            continue
        return (book, rng.choice(long_enough))
    return None


def download_chapter_mp3(listen_url: str, path: Path) -> bool:
    try:
        req = Request(listen_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=180) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  download failed: {e}")
        return False


def extract_clip_ffmpeg(src: Path, start_sec: float, dur_sec: float, out: Path) -> bool:
    cap = CLIP_MAX_SEC - FFMPEG_DURATION_MARGIN_SEC
    actual_dur = min(dur_sec, cap)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec), "-i", str(src),
        "-t", str(round(actual_dur, 2)),
        "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-clips", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="./clips")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    manifest_f = open(manifest_path, "a")

    rng = random.Random(args.seed)
    min_chapter_sec = CLIPS_PER_CHAPTER * CLIP_MAX_SEC + 60

    n_existing = sum(1 for _ in out_dir.glob("*.wav"))
    print(f"[downloader] {n_existing} existing clips, target={args.target_clips}")
    n = n_existing
    while n < args.target_clips:
        chosen = pick_random_chapter(rng, min_chapter_sec)
        if not chosen:
            time.sleep(2)
            continue
        book, section = chosen
        listen_url = section.get("listen_url")
        duration = playtime_sec(section)
        if not listen_url or duration < min_chapter_sec:
            continue
        section_title = (section.get("title") or "chapter")[:40].replace("/", "_")
        book_title = (book.get("title") or "book")[:40].replace("/", "_")
        print(f"[downloader] book={book_title!r} section={section_title!r} dur={duration:.0f}s")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            chapter_path = Path(tmp.name)
        try:
            if not download_chapter_mp3(listen_url, chapter_path):
                continue
            time.sleep(0.3)
            for i in range(CLIPS_PER_CHAPTER):
                if n >= args.target_clips:
                    break
                clip_dur = rng.uniform(CLIP_MIN_SEC, CLIP_MAX_SEC)
                clip_dur = min(clip_dur, CLIP_MAX_SEC - FFMPEG_DURATION_MARGIN_SEC)
                max_start = duration - clip_dur - 1
                if max_start <= 0:
                    continue
                start_sec = rng.uniform(0, max_start)
                clip_id = uuid.uuid4().hex[:16]
                out_path = out_dir / f"{clip_id}.wav"
                if not extract_clip_ffmpeg(chapter_path, start_sec, clip_dur, out_path):
                    continue
                meta = {
                    "clip_id": clip_id,
                    "wav_path": str(out_path.relative_to(out_dir)),
                    "book_id": book.get("id"),
                    "book_title": book.get("title"),
                    "section_title": section.get("title"),
                    "section_url": listen_url,
                    "start_sec": round(start_sec, 2),
                    "clip_dur": round(clip_dur, 2),
                }
                manifest_f.write(json.dumps(meta) + "\n")
                manifest_f.flush()
                n += 1
                if n % 10 == 0:
                    print(f"[downloader]   {n}/{args.target_clips} clips")
        finally:
            try:
                chapter_path.unlink()
            except OSError:
                pass

    manifest_f.close()
    print(f"[downloader] done: {n} clips in {out_dir}")


if __name__ == "__main__":
    main()
