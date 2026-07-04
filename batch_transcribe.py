import os
import sys
import subprocess
import tempfile

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0] if args else None

def fix_cuda_dll_path():
    appdata = os.environ.get('APPDATA', '')
    python_version = f"Python{sys.version_info.major}{sys.version_info.minor}"

    potential_paths = [
        os.path.join(appdata, 'Python', python_version, 'site-packages', 'nvidia', 'cublas', 'bin'),
        os.path.join(appdata, 'Python', python_version, 'site-packages', 'nvidia', 'cudnn', 'bin'),
        os.path.join(appdata, 'Python', python_version, 'site-packages', 'nvidia', 'cuda_nvrtc', 'bin'),
        os.path.join(appdata, 'Python', python_version, 'site-packages', 'nvidia', 'curand', 'bin'),
    ]

    for path in potential_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

fix_cuda_dll_path()

from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
LOCAL_MODEL_PATH = None
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".flac"}

def get_audio_duration(file_path):
    """Get duration using ffprobe (in seconds)."""
    ffmpeg_exe = "ffprobe"
    try:
        subprocess.run(["where", "ffprobe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        local_ffprobe = os.path.join(os.getcwd(), "ffprobe.exe")
        if os.path.exists(local_ffprobe):
            ffmpeg_exe = local_ffprobe
        else:
            return 0.0

    cmd = [
        ffmpeg_exe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return 0.0

def convert_to_wav(input_file):
    # Try to find ffmpeg
    ffmpeg_exe = "ffmpeg"
    try:
        subprocess.run(["where", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            ffmpeg_exe = local_ffmpeg
        else:
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg or place ffmpeg.exe in the script directory.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name

    cmd = [ffmpeg_exe, "-y", "-i", input_file, "-ar", "16000", "-ac", "1", temp_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return temp_wav

def format_timestamp(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe(model, audio_path):
    print("🎙️ Transcribing with real-time progress...")

    duration = get_audio_duration(audio_path)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    print(f"🌐 Language: {info.language} ({info.language_probability:.2f})")

    clean_lines = []
    collected_segments = []

    # TRUE PROGRESS BAR (based on audio time)
    pbar = tqdm(total=duration, unit="sec", desc="Progress")

    last_time = 0

    for seg in segments:
        text = seg.text.strip()

        # Update progress bar based on segment end time
        progress_increment = seg.end - last_time
        if progress_increment > 0:
            pbar.update(progress_increment)
            last_time = seg.end

        # print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {text}", flush=True)

        clean_lines.append(text)
        collected_segments.append(seg)

    pbar.close()

    return "\n".join(clean_lines), collected_segments

def process_file(model, input_file):
    ext = os.path.splitext(input_file)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return

    output_txt = os.path.splitext(input_file)[0] + ".txt"
    output_srt = os.path.splitext(input_file)[0] + ".srt"

    wav_file = None

    try:
        print(f"\n📂 Processing: {os.path.basename(input_file)}")

        wav_file = convert_to_wav(input_file)

        text, segments = transcribe(model, wav_file)

        # TXT: CLEAN TEXT ONLY
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)

        # SRT: KEEP TIMESTAMPS
        with open(output_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")

        print(f"✅ Done: {os.path.basename(output_txt)} & {os.path.basename(output_srt)}")

    except Exception as e:
        print(f"❌ Error: {input_file}\n{e}")

    finally:
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_transcribe.py <file_or_folder>")
        sys.exit(1)

    target_path = sys.argv[1]
    model_source = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH else MODEL_SIZE

    print(f"🧠 Loading model: {model_source}")

    try:
        model = WhisperModel(model_source, compute_type="float16", device="cuda")
        print("🚀 Using GPU (CUDA)")
    except Exception as e:
        print(f"⚠️ GPU failed: {e}")
        print("➡️ Falling back to CPU...")
        model = WhisperModel(model_source, compute_type="int8", device="cpu")

    if os.path.isdir(target_path):
        files = [
            os.path.join(target_path, f)
            for f in os.listdir(target_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]

        print(f"📁 Found {len(files)} files")

        for file in tqdm(files, desc="Overall Progress", unit="file"):
            process_file(model, file)

    elif os.path.isfile(target_path):
        process_file(model, target_path)
    else:
        print(f"❌ Path not found: {target_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
