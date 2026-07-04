#!/usr/bin/env python3
"""
batch_transcribe.py — Local batch audio/video transcription using faster-whisper.

Features:
  - CLI flags for model, device, compute type, language, task, formats, VAD, etc.
  - Output formats: txt, srt, vtt, json (any combination)
  - Recursive directory scanning, skip-existing / resume support
  - Smart WAV passthrough (skips ffmpeg re-encode when input is already 16k mono wav)
  - Robust ffmpeg/ffprobe discovery (PATH, script dir, or --ffmpeg-dir)
  - Graceful Ctrl+C handling (finishes current file, then stops)
  - Optional initial prompt / hotwords, word-level timestamps, condition_on_previous_text
  - Per-file and overall progress bars; JSON sidecar with full segment + word data

Usage examples:
  python batch_transcribe.py audio.mp3
  python batch_transcribe.py ./lectures --recursive --formats txt,srt,vtt
  python batch_transcribe.py ./meetings -m large-v3 -d cuda -l en --skip-existing
  python batch_transcribe.py clip.mp4 --word-timestamps --formats json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else _DummyBar()

    class _DummyBar:
        def update(self, *a, **k): pass
        def close(self): pass


LOG = logging.getLogger("batch_transcribe")

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mp3", ".wav", ".m4a", ".flac",
    ".ogg", ".opus", ".webm", ".mkv", ".avi", ".mov", ".aac", ".wma",
}
ALL_FORMATS = ("txt", "srt", "vtt", "json")


# --------------------------------------------------------------------------- #
# Windows CUDA DLL fix (harmless no-op on Linux/macOS)
# --------------------------------------------------------------------------- #
def fix_cuda_dll_path() -> None:
    if os.name != "nt":
        return
    appdata = os.environ.get("APPDATA", "")
    python_version = f"Python{sys.version_info.major}{sys.version_info.minor}"
    candidates = [
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
        ("nvidia", "curand", "bin"),
    ]
    for parts in candidates:
        path = os.path.join(appdata, "Python", python_version, "site-packages", *parts)
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)  # type: ignore[attr-defined]
            except Exception:
                pass


fix_cuda_dll_path()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    input_path: Path
    output_dir: Optional[Path]
    model: str
    device: str
    compute_type: str
    language: Optional[str]
    task: str
    formats: List[str]
    beam_size: int
    vad: bool
    vad_min_silence_ms: int
    initial_prompt: Optional[str]
    hotwords: Optional[str]
    word_timestamps: bool
    condition_on_previous_text: bool
    recursive: bool
    skip_existing: bool
    dry_run: bool
    download_root: Optional[str]
    ffmpeg_dir: Optional[str]
    temperature: float


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe discovery
# --------------------------------------------------------------------------- #
def _find_tool(name: str, ffmpeg_dir: Optional[str]) -> Optional[str]:
    exe = name + (".exe" if os.name == "nt" else "")
    if ffmpeg_dir:
        candidate = os.path.join(ffmpeg_dir, exe)
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    local = os.path.join(os.getcwd(), exe)
    if os.path.exists(local):
        return local
    return None


def get_audio_duration(file_path: str, ffprobe: str) -> float:
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def is_whisper_ready_wav(file_path: str) -> bool:
    """True if file is already a 16kHz mono WAV (no re-encode needed)."""
    if Path(file_path).suffix.lower() != ".wav":
        return False
    try:
        with wave.open(file_path, "rb") as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1
    except Exception:
        return False


def convert_to_wav(input_file: str, ffmpeg: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name
    cmd = [ffmpeg, "-y", "-i", input_file, "-ar", "16000", "-ac", "1", temp_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return temp_wav


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_timestamp_srt(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    return format_timestamp_srt(seconds).replace(",", ".")


def segment_to_dict(seg) -> dict:
    d = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
    words = getattr(seg, "words", None)
    if words:
        d["words"] = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": getattr(w, "probability", None)}
            for w in words
        ]
    return d


def write_outputs(base_path: Path, formats: Sequence[str], text: str, segments: list, info) -> List[Path]:
    written = []
    if "txt" in formats:
        p = base_path.with_suffix(".txt")
        p.write_text(text, encoding="utf-8")
        written.append(p)

    if "srt" in formats:
        p = base_path.with_suffix(".srt")
        with open(p, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")
        written.append(p)

    if "vtt" in formats:
        p = base_path.with_suffix(".vtt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                f.write(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")
        written.append(p)

    if "json" in formats:
        p = base_path.with_suffix(".json")
        payload = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": getattr(info, "duration", None),
            "segments": [segment_to_dict(s) for s in segments],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p)

    return written


# --------------------------------------------------------------------------- #
# Core transcription
# --------------------------------------------------------------------------- #
def transcribe(model, audio_path: str, cfg: Config, ffprobe: Optional[str]):
    duration = get_audio_duration(audio_path, ffprobe) if ffprobe else 0.0

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad,
        vad_parameters=dict(min_silence_duration_ms=cfg.vad_min_silence_ms) if cfg.vad else None,
        language=cfg.language,
        task=cfg.task,
        initial_prompt=cfg.initial_prompt,
        hotwords=cfg.hotwords,
        word_timestamps=cfg.word_timestamps,
        condition_on_previous_text=cfg.condition_on_previous_text,
        temperature=cfg.temperature,
    )

    LOG.info("Detected language: %s (p=%.2f)", info.language, info.language_probability)

    pbar = tqdm(total=duration if duration > 0 else None, unit="sec", desc="Progress", leave=False)
    last_time = 0.0
    clean_lines = []
    collected = []

    for seg in segments_gen:
        text = seg.text.strip()
        if duration > 0:
            inc = seg.end - last_time
            if inc > 0:
                pbar.update(inc)
                last_time = seg.end
        else:
            pbar.update(1)
        clean_lines.append(text)
        collected.append(seg)

    pbar.close()
    return "\n".join(clean_lines), collected, info


def outputs_exist(base_path: Path, formats: Sequence[str]) -> bool:
    return all(base_path.with_suffix(f".{fmt}").exists() for fmt in formats)


def process_file(model, input_file: Path, cfg: Config, ffmpeg: Optional[str], ffprobe: Optional[str]) -> bool:
    ext = input_file.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False

    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        base_path = cfg.output_dir / input_file.stem
    else:
        base_path = input_file.with_suffix("")

    if cfg.skip_existing and outputs_exist(base_path, cfg.formats):
        LOG.info("Skipping (outputs exist): %s", input_file.name)
        return True

    if cfg.dry_run:
        LOG.info("[dry-run] Would transcribe: %s -> %s.{%s}", input_file, base_path, ",".join(cfg.formats))
        return True

    wav_file = None
    temp_created = False
    try:
        LOG.info("Processing: %s", input_file.name)

        if is_whisper_ready_wav(str(input_file)):
            wav_file = str(input_file)
        else:
            if not ffmpeg:
                raise FileNotFoundError(
                    "ffmpeg not found. Install ffmpeg, add it to PATH, or pass --ffmpeg-dir."
                )
            wav_file = convert_to_wav(str(input_file), ffmpeg)
            temp_created = True

        text, segments, info = transcribe(model, wav_file, cfg, ffprobe)
        written = write_outputs(base_path, cfg.formats, text, segments, info)

        LOG.info("Done: %s", ", ".join(p.name for p in written))
        return True

    except Exception as e:
        LOG.error("Failed on %s: %s", input_file, e)
        return False

    finally:
        if temp_created and wav_file and os.path.exists(wav_file):
            os.remove(wav_file)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(cfg: Config):
    from faster_whisper import WhisperModel

    model_source = cfg.model
    device = cfg.device
    compute_type = cfg.compute_type

    def _try(dev: str, ctype: str):
        LOG.info("Loading model '%s' on %s (%s)...", model_source, dev, ctype)
        return WhisperModel(model_source, device=dev, compute_type=ctype, download_root=cfg.download_root)

    if device == "auto":
        try:
            model = _try("cuda", compute_type if compute_type != "auto" else "float16")
            LOG.info("Using GPU (CUDA)")
            return model
        except Exception as e:
            LOG.warning("GPU unavailable (%s); falling back to CPU.", e)
            return _try("cpu", compute_type if compute_type != "auto" else "int8")

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        model = _try(device, compute_type)
        LOG.info("Using device: %s", device)
        return model
    except Exception as e:
        if device == "cuda":
            LOG.warning("GPU failed (%s); falling back to CPU.", e)
            return _try("cpu", "int8")
        raise


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description="Batch transcribe audio/video files locally with faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Audio/video file or a folder containing them")
    p.add_argument("-o", "--output-dir", default=None, help="Write outputs here instead of next to source files")
    p.add_argument("-m", "--model", default="medium",
                   help="Model size (tiny/base/small/medium/large-v3) or a local model path")
    p.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("-c", "--compute-type", default="auto",
                   help="e.g. float16, int8, int8_float16, float32, or 'auto'")
    p.add_argument("-l", "--language", default=None, help="Force language code (e.g. en, zh). Default: auto-detect")
    p.add_argument("-t", "--task", default="transcribe", choices=["transcribe", "translate"],
                   help="'translate' outputs English regardless of source language")
    p.add_argument("-f", "--formats", default="txt,srt",
                   help=f"Comma-separated output formats: {','.join(ALL_FORMATS)}")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-vad", action="store_true", help="Disable voice activity detection filtering")
    p.add_argument("--vad-min-silence-ms", type=int, default=500)
    p.add_argument("--initial-prompt", default=None, help="Bias transcription style/vocabulary")
    p.add_argument("--hotwords", default=None, help="Words/phrases to boost recognition of")
    p.add_argument("--word-timestamps", action="store_true", help="Include per-word timestamps (needed for word-level JSON)")
    p.add_argument("--no-condition-on-previous-text", action="store_true",
                   help="Disable conditioning on previous text (reduces repetition loops on noisy audio)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders when input is a directory")
    p.add_argument("--skip-existing", action="store_true", help="Skip files whose outputs already exist")
    p.add_argument("--dry-run", action="store_true", help="List what would be processed without doing it")
    p.add_argument("--download-root", default=None, help="Custom directory for cached/downloaded models")
    p.add_argument("--ffmpeg-dir", default=None, help="Directory containing ffmpeg/ffprobe binaries")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (debug) logging")

    args = p.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    invalid = [f for f in formats if f not in ALL_FORMATS]
    if invalid:
        p.error(f"Unknown format(s): {invalid}. Choose from {ALL_FORMATS}")
    if not formats:
        p.error("At least one output format is required.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    return Config(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        task=args.task,
        formats=formats,
        beam_size=args.beam_size,
        vad=not args.no_vad,
        vad_min_silence_ms=args.vad_min_silence_ms,
        initial_prompt=args.initial_prompt,
        hotwords=args.hotwords,
        word_timestamps=args.word_timestamps,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        download_root=args.download_root,
        ffmpeg_dir=args.ffmpeg_dir,
        temperature=args.temperature,
    )


def collect_files(cfg: Config) -> List[Path]:
    if cfg.input_path.is_file():
        return [cfg.input_path]
    if cfg.input_path.is_dir():
        pattern_iter = cfg.input_path.rglob("*") if cfg.recursive else cfg.input_path.glob("*")
        return sorted(f for f in pattern_iter if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
    return []


def main(argv=None) -> int:
    cfg = parse_args(argv)

    if not cfg.input_path.exists():
        LOG.error("Path not found: %s", cfg.input_path)
        return 1

    files = collect_files(cfg)
    if not files:
        LOG.error("No supported audio/video files found at: %s", cfg.input_path)
        return 1

    ffmpeg = _find_tool("ffmpeg", cfg.ffmpeg_dir)
    ffprobe = _find_tool("ffprobe", cfg.ffmpeg_dir)
    if not ffmpeg:
        LOG.warning("ffmpeg not found on PATH — only pre-formatted 16kHz mono WAV files can be processed as-is.")
    if not ffprobe:
        LOG.warning("ffprobe not found — progress bars will be indeterminate.")

    if cfg.dry_run:
        LOG.info("Dry run: %d file(s) would be processed.", len(files))
        for f in files:
            LOG.info("  - %s", f)
        return 0

    model = load_model(cfg)

    LOG.info("Found %d file(s) to process.", len(files))
    ok, failed = 0, 0
    try:
        for f in tqdm(files, desc="Overall Progress", unit="file"):
            if process_file(model, f, cfg, ffmpeg, ffprobe):
                ok += 1
            else:
                failed += 1
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user — stopping after current file.")

    LOG.info("Finished. Success: %d, Failed: %d, Total: %d", ok, failed, len(files))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())