"""Optional: split voice and background music with Demucs (pip install demucs)."""
import os, subprocess, sys, glob

def separate(wav_path, out_dir):
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                    "-o", out_dir, wav_path], check=True)
    name = os.path.splitext(os.path.basename(wav_path))[0]
    vocals = glob.glob(os.path.join(out_dir, "*", name, "vocals.wav"))[0]
    background = glob.glob(os.path.join(out_dir, "*", name, "no_vocals.wav"))[0]
    return vocals, background
