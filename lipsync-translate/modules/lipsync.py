"""Lip sync with Wav2Lip (called as an external script)."""
import os, subprocess
import config

def apply(silent_video, audio_wav, out_mp4):
    if not os.path.exists(config.WAV2LIP_CKPT):
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {config.WAV2LIP_CKPT}")
    cmd = [config.WAV2LIP_PYTHON, "inference.py",
           "--checkpoint_path", os.path.abspath(config.WAV2LIP_CKPT),
           "--face", os.path.abspath(silent_video),
           "--audio", os.path.abspath(audio_wav),
           "--outfile", os.path.abspath(out_mp4)]
    # Optional flags if the result looks bad: "--pads", "0", "10", "0", "0"  and  "--nosmooth"
    subprocess.run(cmd, cwd=config.WAV2LIP_DIR, check=True)
    return out_mp4
