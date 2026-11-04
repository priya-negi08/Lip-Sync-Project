"""Text -> speech. XTTS-v2 (voice cloning) with gTTS fallback."""
import os
import numpy as np
import soundfile as sf
import config
from modules import media

_xtts = None

def build_reference(wav_path, segments, out_path, max_seconds=15):
    """Concatenate the longest speech segments into a short voice sample for cloning."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    chosen, total = [], 0.0
    for s in sorted(segments, key=lambda x: x["end"] - x["start"], reverse=True):
        chunk = audio[int(s["start"] * sr): int(s["end"] * sr)]
        if len(chunk) == 0:
            continue
        chosen.append(chunk)
        total += len(chunk) / sr
        if total >= max_seconds:
            break
    ref = np.concatenate(chosen) if chosen else audio[: int(max_seconds * sr)]
    sf.write(out_path, ref, sr)
    return out_path

def _get_xtts():
    global _xtts
    if _xtts is None:
        from TTS.api import TTS
        _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(config.DEVICE)
    return _xtts

def synthesize(segments, ref_wav, tgt_code, out_dir, clone=True):
    os.makedirs(out_dir, exist_ok=True)
    xtts_lang = config.BY_CODE[tgt_code][2]
    for seg in segments:
        out = os.path.join(out_dir, f"seg_{seg['id']:04d}.wav")
        text = seg["text_tgt"]
        if clone:
            _get_xtts().tts_to_file(text=text, speaker_wav=ref_wav,
                                    language=xtts_lang, file_path=out)
        else:
            from gtts import gTTS
            mp3 = out.replace(".wav", ".mp3")
            gTTS(text=text, lang=tgt_code if tgt_code != "zh" else "zh-CN").save(mp3)
            media.to_wav(mp3, out, config.TTS_SR)
        seg["tts_wav"] = out
    return segments
