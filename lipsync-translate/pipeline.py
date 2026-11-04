"""Orchestrates the whole pipeline.

CLI:  python pipeline.py input.mp4 hi            (Hindi, with lip sync)
      python pipeline.py input.mp4 es --no-lipsync
"""
import os, json, uuid, shutil, argparse
import config
from modules import media, asr, translate, tts, align, lipsync

def run(video_path, tgt_code, src_code=None, do_lipsync=True, use_demucs=False,
        clone_voice=True, progress=None):
    def step(frac, msg):
        print(f"[{int(frac*100):3d}%] {msg}")
        if progress:
            progress(frac, desc=msg)

    job = os.path.join(config.BASE_DIR, "work", uuid.uuid4().hex[:8])
    os.makedirs(job, exist_ok=True)
    total = media.duration(video_path)

    step(0.05, "Extracting audio and video")
    audio16 = os.path.join(job, "audio16k.wav")
    audio24 = os.path.join(job, "audio24k.wav")
    silent = os.path.join(job, "silent.mp4")
    media.extract_audio(video_path, audio16, sr=16000)
    media.extract_audio(video_path, audio24, sr=24000)
    media.to_cfr_silent(video_path, silent)

    voice16, voice24, background = audio16, audio24, None
    if use_demucs:
        from modules import separation
        step(0.10, "Separating voice and music (Demucs)")
        vocals, background = separation.separate(audio24, os.path.join(job, "demucs"))
        voice24 = vocals
        voice16 = os.path.join(job, "vocals16k.wav")
        media.to_wav(vocals, voice16, 16000)

    step(0.20, "Speech recognition (Whisper)")
    segments, detected = asr.transcribe(voice16, language=src_code)
    src_code = src_code or detected
    if not segments:
        raise RuntimeError("No speech found in the video.")

    step(0.35, "Translating")
    segments = translate.translate_segments(segments, src_code, tgt_code)

    step(0.45, "Preparing voice sample")
    ref = tts.build_reference(voice24, segments, os.path.join(job, "ref.wav"))

    step(0.50, "Generating new speech")
    segments = tts.synthesize(segments, ref, tgt_code, os.path.join(job, "tts"), clone=clone_voice)

    step(0.70, "Fitting speech to timing")
    segments = align.fit_segments(segments, os.path.join(job, "fit"), total)
    dub = align.build_timeline(segments, total, os.path.join(job, "dub.wav"))

    final_audio = dub
    if background:
        final_audio = os.path.join(job, "dub_mix.wav")
        media.mix_audio(dub, background, final_audio)

    if do_lipsync:
        step(0.80, "Lip sync (Wav2Lip)")
        synced = lipsync.apply(silent, dub, os.path.join(job, "lipsynced.mp4"))
        video_for_mux = synced
    else:
        video_for_mux = silent

    step(0.95, "Creating final video")
    out = os.path.join(job, "final.mp4")
    media.mux(video_for_mux, final_audio, out)

    with open(os.path.join(job, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    step(1.0, "Done")
    return out, segments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("target", help="language code, e.g. hi, es, fr")
    ap.add_argument("--src", default=None)
    ap.add_argument("--no-lipsync", action="store_true")
    ap.add_argument("--demucs", action="store_true")
    ap.add_argument("--no-clone", action="store_true", help="use gTTS instead of XTTS")
    a = ap.parse_args()
    path, _ = run(a.video, a.target, a.src, not a.no_lipsync, a.demucs, not a.no_clone)
    print("Output:", path)
