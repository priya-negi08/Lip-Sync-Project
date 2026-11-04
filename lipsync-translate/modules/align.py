"""Fit each spoken segment into its time slot and build the final audio timeline."""
import os
import numpy as np
import soundfile as sf
import config
from modules import media

def _atempo_chain(factor):
    """ffmpeg atempo accepts 0.5-2.0 per filter; chain if needed."""
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0"); factor /= 2.0
    parts.append(f"atempo={factor:.4f}")
    return ",".join(parts)

def fit_segments(segments, out_dir, total_dur, max_speedup=1.4, borrow=0.5):
    os.makedirs(out_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        orig = seg["end"] - seg["start"]
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else total_dur
        # Allow borrowing a little silence after the segment, but never run into the next one
        slot = max(orig, min(orig + borrow, next_start - seg["start"]))
        src = seg["tts_wav"]
        out = os.path.join(out_dir, f"fit_{seg['id']:04d}.wav")
        d = media.duration(src)
        ratio = d / slot
        seg["ratio"] = round(ratio, 2)
        if ratio > 1.05:
            factor = min(ratio, max_speedup)
            media.run(["ffmpeg", "-y", "-i", src, "-filter:a", _atempo_chain(factor),
                       "-ar", str(config.TTS_SR), "-ac", "1", out])
        else:
            media.to_wav(src, out, config.TTS_SR)
        seg["fit_wav"] = out
        seg["slot"] = slot
    return segments

def build_timeline(segments, total_dur, out_wav):
    sr = config.TTS_SR
    buf = np.zeros(int(total_dur * sr) + sr, dtype=np.float32)
    fade = int(0.03 * sr)
    for seg in segments:
        y, _ = sf.read(seg["fit_wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        max_len = int(seg["slot"] * sr)
        if len(y) > max_len:                       # still too long -> trim with a short fade-out
            y = y[:max_len].copy()
            n = min(fade, len(y))
            y[-n:] *= np.linspace(1, 0, n)
        a = int(seg["start"] * sr)
        buf[a:a + len(y)] += y
    buf = buf[: int(total_dur * sr)]               # exactly the video length
    peak = np.max(np.abs(buf)) or 1.0
    if peak > 0.99:
        buf = buf / peak * 0.95
    sf.write(out_wav, buf, sr)
    return out_wav
