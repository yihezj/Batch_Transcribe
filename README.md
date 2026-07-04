# batch_transcribe.py

A local, offline batch transcription tool for audio and video files, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2
reimplementation of OpenAI Whisper). Point it at a single file or a whole
folder and get clean text, subtitles, and/or structured JSON — no cloud API,
no per-minute billing.

## Features

- **GPU-accelerated** (CUDA) with automatic fallback to CPU
- **Batch or single-file** processing, with optional recursive folder scanning
- **Multiple output formats** in one pass: `txt`, `srt`, `vtt`, `json`
- **Smart WAV passthrough** — skips the ffmpeg re-encode step if a file is
  already 16 kHz mono WAV
- **Resumable batches** via `--skip-existing`
- **Accuracy controls**: language forcing, translation, initial prompts,
  hotwords, beam size, temperature, VAD tuning
- **Word-level timestamps** (optional) for precise subtitle/karaoke-style sync
- **Dry-run mode** to preview a batch before committing GPU time
- **Graceful Ctrl+C handling** — finishes the current file, then stops cleanly

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) (for any input that isn't
  already 16 kHz mono WAV)
- A CUDA-capable GPU + drivers for GPU acceleration (optional — CPU works too,
  just slower)

### Python packages

```bash
pip install faster-whisper tqdm
```

`tqdm` is optional (used for progress bars); the script degrades gracefully
without it.

### ffmpeg setup

The script looks for `ffmpeg`/`ffprobe` in, in order:

1. A directory you pass via `--ffmpeg-dir`
2. Your system `PATH`
3. The current working directory (e.g. `ffmpeg.exe` sitting next to the script
   on Windows)

If none are found, only pre-formatted 16 kHz mono WAV files can be processed
(everything else requires ffmpeg to convert/extract audio).

**Windows quick install:** download a static build from
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or
[BtbN](https://github.com/BtbN/FFmpeg-Builds/releases), unzip, and either add
the `bin` folder to your `PATH` or pass `--ffmpeg-dir "C:\path\to\ffmpeg\bin"`.

**macOS:** `brew install ffmpeg`
**Linux:** `sudo apt install ffmpeg` (or your distro's equivalent)

### GPU acceleration (optional)

For CUDA support, faster-whisper needs the matching NVIDIA cuBLAS/cuDNN
libraries. The easiest path is:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

On Windows, the script automatically adds the relevant `site-packages/nvidia/*/bin`
folders to the DLL search path at startup, so a `pip install` into your
user site-packages is usually enough — no manual `PATH` editing required.

If CUDA isn't available or fails to initialize, the script automatically
falls back to CPU (`int8` compute type) with a warning.

## Quick Start

```bash
# Transcribe a single file (txt + srt, auto language/device)
python batch_transcribe.py lecture.mp4

# Transcribe every supported file in a folder
python batch_transcribe.py ./recordings

# Recurse into subfolders, output JSON + VTT only
python batch_transcribe.py ./recordings --recursive --formats json,vtt

# Force English, use a bigger/more accurate model, write elsewhere
python batch_transcribe.py meeting.wav -m large-v3 -l en -o ./transcripts

# Preview what would run without actually transcribing
python batch_transcribe.py ./recordings --dry-run
```

## Supported Input Formats

`.mp4  .mp3  .wav  .m4a  .flac  .ogg  .opus  .webm  .mkv  .avi  .mov  .aac  .wma`

Anything else is silently skipped when scanning a folder.

## Output Formats

| Format | Extension | Contents |
|---|---|---|
| Plain text | `.txt` | Concatenated transcript, no timestamps |
| SubRip subtitles | `.srt` | Numbered cues with `HH:MM:SS,mmm` timestamps |
| WebVTT subtitles | `.vtt` | Same as SRT, web-standard timestamp format |
| JSON | `.json` | Detected language + probability, duration, and every segment (start/end/text), plus per-word timestamps if `--word-timestamps` is set |

Select one or more with `--formats`, comma-separated, e.g.
`--formats txt,srt,vtt,json`. Default: `txt,srt`.

Output files are named after the input file (same stem, new extension) and
are written next to the source file unless `--output-dir` is given.

## Command-Line Reference

```
python batch_transcribe.py <input> [options]
```

| Flag | Default | Description |
|---|---|---|
| `input` | — | Audio/video file, or a folder containing them |
| `-o`, `--output-dir` | *(next to source)* | Directory to write outputs into |
| `-m`, `--model` | `medium` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`, ...) or a local model path |
| `-d`, `--device` | `auto` | `auto`, `cuda`, or `cpu`. `auto` tries CUDA first, falls back to CPU |
| `-c`, `--compute-type` | `auto` | e.g. `float16`, `int8`, `int8_float16`, `float32`. `auto` picks `float16` (GPU) or `int8` (CPU) |
| `-l`, `--language` | *(auto-detect)* | Force a language code, e.g. `en`, `zh`, `ja` |
| `-t`, `--task` | `transcribe` | `transcribe` (native language) or `translate` (always outputs English) |
| `-f`, `--formats` | `txt,srt` | Comma-separated: `txt`, `srt`, `vtt`, `json` |
| `--beam-size` | `5` | Beam search width (higher = slower, sometimes more accurate) |
| `--temperature` | `0.0` | Sampling temperature; raise slightly if output gets stuck in repetition loops |
| `--no-vad` | *(VAD on)* | Disable voice activity detection filtering |
| `--vad-min-silence-ms` | `500` | Minimum silence (ms) for VAD to split segments |
| `--initial-prompt` | *(none)* | Text to bias style/vocabulary (e.g. proper nouns, punctuation style) |
| `--hotwords` | *(none)* | Words/phrases to boost recognition of |
| `--word-timestamps` | off | Include per-word timing (needed for word-level JSON) |
| `--no-condition-on-previous-text` | *(on by default)* | Disable conditioning on prior text; can reduce repetition loops on noisy/long audio |
| `--recursive` | off | Recurse into subfolders when `input` is a directory |
| `--skip-existing` | off | Skip a file if all its requested outputs already exist |
| `--dry-run` | off | List files that would be processed without transcribing |
| `--download-root` | *(faster-whisper default cache)* | Custom directory for downloaded/cached models |
| `--ffmpeg-dir` | *(PATH/cwd)* | Directory containing `ffmpeg`/`ffprobe` binaries |
| `-v`, `--verbose` | off | Enable debug-level logging |

## Choosing a Model

| Model | Relative speed | Accuracy | VRAM (approx, float16) |
|---|---|---|---|
| `tiny` | fastest | lowest | ~1 GB |
| `base` | very fast | low | ~1 GB |
| `small` | fast | good | ~2 GB |
| `medium` | moderate | very good | ~5 GB |
| `large-v3` | slowest | best | ~10 GB |

If you're transcribing quickly for note-taking, `small` or `medium` is
usually the sweet spot. For final deliverables, subtitles, or non-English
audio, `large-v3` gives the best results if your GPU (or patience, on CPU)
can handle it.

## Recipes

**Resumable overnight batch job on a folder of lecture recordings:**

```bash
python batch_transcribe.py ./lectures --recursive --skip-existing \
  -m large-v3 --formats txt,srt,json -v
```

**Fast draft transcripts on CPU only (no GPU available):**

```bash
python batch_transcribe.py ./voice_memos -d cpu -m small --formats txt
```

**Subtitling a video in a language you know, with domain vocabulary:**

```bash
python batch_transcribe.py talk.mp4 -l en --formats srt,vtt \
  --initial-prompt "Topics include CRISPR, RNA-seq, and PEO glass transition."
```

**Translating non-English audio straight to English text:**

```bash
python batch_transcribe.py interview_zh.mp3 -t translate --formats txt
```

**Karaoke-style word-by-word timing:**

```bash
python batch_transcribe.py song.wav --word-timestamps --formats json
```

## How It Works

1. Scans the input path (single file, or folder — recursively if
   `--recursive`) for supported extensions.
2. Loads the faster-whisper model once, on GPU if available, otherwise CPU.
3. For each file:
   - If it's already a 16 kHz mono WAV, uses it directly.
   - Otherwise, converts it to a temporary 16 kHz mono WAV via ffmpeg.
   - Runs transcription with your chosen accuracy/decoding settings.
   - Writes each requested output format next to the source (or to
     `--output-dir`), named after the original file.
   - Cleans up any temporary WAV file it created.
4. Reports a final summary (success/failure counts).

## Troubleshooting

**`ffmpeg not found`**
Install ffmpeg and ensure it's on your `PATH`, or pass `--ffmpeg-dir`
pointing at the folder containing `ffmpeg`/`ffprobe`.

**GPU falls back to CPU unexpectedly**
Check the warning message printed just before the fallback — it includes the
underlying CUDA/cuDNN error. Common causes: missing `nvidia-cublas-cu12` /
`nvidia-cudnn-cu12` packages, or a driver/CUDA version mismatch.

**Progress bar shows no total / jumps oddly**
This happens when `ffprobe` can't be found or can't read the file's
duration — the bar falls back to counting segments instead of seconds.

**Transcription repeats the same phrase over and over**
Try `--no-condition-on-previous-text`, or raise `--temperature` slightly
(e.g. `0.2`), which are the classic fixes for Whisper's repetition-loop
failure mode on long silences or noisy audio.

**Non-WAV files being re-encoded even though they sound fine**
That's expected — only WAV files already at 16 kHz mono skip the ffmpeg
step; everything else (MP3, MP4, etc.) is always converted for compatibility.

## Notes & Limitations

- Processing is sequential (one file at a time) since a single loaded model
  instance is reused across the batch; this keeps VRAM usage predictable but
  means files aren't transcribed in parallel.
- `--skip-existing` checks for the presence of output files, not whether
  they were generated with the same settings — changing `--formats` or
  `--model` after a partial run won't force a re-transcription of files that
  already have *some* matching outputs.
- Video files are transcribed using their audio track only; no visual
  processing occurs.