# AI Lip Sync and Video Translation (starter project)

Pipeline: FFmpeg -> (Demucs) -> Whisper -> NLLB translation -> XTTS voice clone
-> time fitting -> Wav2Lip -> FFmpeg mux. UI: Gradio.

## 1. Install
1. Install Python 3.10/3.11, FFmpeg (on PATH), and an NVIDIA driver.
2. Install PyTorch with CUDA from https://pytorch.org
3. `pip install -r requirements.txt`

## 2. Set up Wav2Lip (separate environment recommended)
Wav2Lip needs old library versions, so give it its own env.
1. `git clone https://github.com/Rudrabha/Wav2Lip` inside this folder.
2. Create a new env for it and install its requirements.
3. Download the checkpoint `wav2lip_gan.pth` into `Wav2Lip/checkpoints/`
   and the face detector `s3fd.pth` into
   `Wav2Lip/face_detection/detection/sfd/` (links are in the Wav2Lip README).
4. Tell this project which Python to use for Wav2Lip:
   - Windows (cmd):  `set WAV2LIP_PYTHON=C:\path\to\wav2lip_env\python.exe`
   - Linux/WSL:      `export WAV2LIP_PYTHON=/path/to/wav2lip_env/bin/python`

## 3. Run
- Web UI:  `python app.py`  then open the shown local URL.
- CLI:     `python pipeline.py input.mp4 hi`
- No GPU / no Wav2Lip yet: `python pipeline.py input.mp4 hi --no-lipsync --no-clone`
  (audio dub only, simple voice).

## 4. Build order
1. `--no-lipsync --no-clone` works -> FFmpeg, Whisper, NLLB, timing are fine.
2. Remove `--no-clone` -> XTTS voice cloning.
3. Remove `--no-lipsync` -> Wav2Lip.
4. Add `--demucs` to keep background music.

Outputs are saved in `work/<job id>/` (final.mp4 and segments.json).

## Notes
- Use only videos you own or have permission to edit; label results as AI-dubbed.
- XTTS-v2 and Wav2Lip have non-commercial licenses. Check before commercial use.
- First run downloads models (several GB).
