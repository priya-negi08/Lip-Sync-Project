"""Central settings. Edit paths here."""
import os, sys, torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper model size: tiny / base / small / medium / large-v3
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small")

# Translation model (NLLB-200 distilled, ~600M params)
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# Audio rate used for TTS output and the final timeline
TTS_SR = 24000

# Wav2Lip lives in its own folder and (usually) its own Python env,
# because it needs old library versions. Point these to your setup.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.getenv("WAV2LIP_DIR", os.path.join(BASE_DIR, "Wav2Lip"))
WAV2LIP_CKPT = os.getenv("WAV2LIP_CKPT", os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"))
WAV2LIP_PYTHON = os.getenv("WAV2LIP_PYTHON", sys.executable)  # interpreter of the Wav2Lip env

# Language name -> (our code, NLLB code, XTTS code)
LANGS = {
    "English":    ("en", "eng_Latn", "en"),
    "Hindi":      ("hi", "hin_Deva", "hi"),
    "Spanish":    ("es", "spa_Latn", "es"),
    "French":     ("fr", "fra_Latn", "fr"),
    "German":     ("de", "deu_Latn", "de"),
    "Italian":    ("it", "ita_Latn", "it"),
    "Portuguese": ("pt", "por_Latn", "pt"),
    "Russian":    ("ru", "rus_Cyrl", "ru"),
    "Arabic":     ("ar", "arb_Arab", "ar"),
    "Japanese":   ("ja", "jpn_Jpan", "ja"),
    "Korean":     ("ko", "kor_Hang", "ko"),
    "Chinese":    ("zh", "zho_Hans", "zh-cn"),
    "Turkish":    ("tr", "tur_Latn", "tr"),
}
BY_CODE = {v[0]: v for v in LANGS.values()}
