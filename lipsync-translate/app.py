"""Gradio web UI:  python app.py"""
import gradio as gr
import config, pipeline

def process(video, language, do_lipsync, use_demucs, clone, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Please upload a video.")
    code = config.LANGS[language][0]
    out, segments = pipeline.run(video, code, None, do_lipsync, use_demucs, clone, progress)
    table = [[round(s["start"], 1), round(s["end"], 1), s["text_src"], s["text_tgt"]] for s in segments]
    return out, table

with gr.Blocks(title="AI Video Translator") as demo:
    gr.Markdown("# AI Lip Sync and Video Translation")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Input video (try 10-30 seconds first)")
            lang = gr.Dropdown(list(config.LANGS.keys()), value="Hindi", label="Translate to")
            lip = gr.Checkbox(True, label="Lip sync (needs Wav2Lip + GPU)")
            dem = gr.Checkbox(False, label="Keep background music (Demucs)")
            clo = gr.Checkbox(True, label="Clone original voice (XTTS). Off = simple gTTS voice")
            btn = gr.Button("Translate", variant="primary")
        with gr.Column():
            out = gr.Video(label="Result")
            tbl = gr.Dataframe(headers=["start", "end", "original", "translated"], wrap=True)
    btn.click(process, [vid, lang, lip, dem, clo], [out, tbl])

if __name__ == "__main__":
    demo.queue().launch()
