"""Speech -> text with timestamps (faster-whisper)."""
from faster_whisper import WhisperModel
import config

_model = None

def transcribe(wav_path, language=None):
    global _model
    if _model is None:
        compute = "float16" if config.DEVICE == "cuda" else "int8"
        _model = WhisperModel(config.WHISPER_SIZE, device=config.DEVICE, compute_type=compute)
    segs, info = _model.transcribe(wav_path, language=language, vad_filter=True)
    out = []
    for i, s in enumerate(segs):
        text = s.text.strip()
        if text:
            out.append({"id": i, "start": float(s.start), "end": float(s.end), "text_src": text})
    return out, info.language
