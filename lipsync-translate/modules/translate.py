"""Text translation with NLLB-200."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config

_tok = _model = None
_loaded_src = None

def _load(src_nllb):
    global _tok, _model, _loaded_src
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL).to(config.DEVICE)
        _model.eval()
    if _tok is None or _loaded_src != src_nllb:
        _tok = AutoTokenizer.from_pretrained(config.NLLB_MODEL, src_lang=src_nllb)
        _loaded_src = src_nllb

def translate_segments(segments, src_code, tgt_code):
    if src_code not in config.BY_CODE:
        raise ValueError(f"Source language '{src_code}' not supported in config.LANGS")
    src_nllb = config.BY_CODE[src_code][1]
    tgt_nllb = config.BY_CODE[tgt_code][1]
    _load(src_nllb)
    for seg in segments:
        if src_code == tgt_code:
            seg["text_tgt"] = seg["text_src"]
            continue
        inputs = _tok(seg["text_src"], return_tensors="pt").to(config.DEVICE)
        # Soft length limit: translation should not be much longer than the audio slot
        dur = max(seg["end"] - seg["start"], 0.5)
        max_len = int(min(256, max(24, dur * 12)))
        with torch.no_grad():
            out = _model.generate(**inputs,
                                  forced_bos_token_id=_tok.convert_tokens_to_ids(tgt_nllb),
                                  max_length=max_len, num_beams=4)
        seg["text_tgt"] = _tok.batch_decode(out, skip_special_tokens=True)[0]
    return segments
