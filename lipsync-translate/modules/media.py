"""FFmpeg helpers: extract, strip, mix, mux."""
import subprocess, json

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r

def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(r.stdout)["format"]["duration"])

def extract_audio(video, out_wav, sr=16000, channels=1):
    run(["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", str(sr), "-ac", str(channels), out_wav])

def to_cfr_silent(video, out_mp4, fps=25):
    """Remove audio and force constant frame rate (avoids drift in lip sync)."""
    run(["ffmpeg", "-y", "-i", video, "-an", "-r", str(fps),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", out_mp4])

def mix_audio(voice_wav, bg_wav, out_wav):
    run(["ffmpeg", "-y", "-i", voice_wav, "-i", bg_wav, "-filter_complex",
         "[0:a][1:a]amix=inputs=2:duration=first:normalize=0", out_wav])

def mux(video, audio, out_mp4):
    run(["ffmpeg", "-y", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4])

def to_wav(src, out_wav, sr, channels=1):
    run(["ffmpeg", "-y", "-i", src, "-ar", str(sr), "-ac", str(channels), out_wav])
