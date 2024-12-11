__all__ = ["app"]

import gradio as gr
import spaces
from binoculars import Binoculars
import gc
import torch
import time

BINO = None
TOKENIZER = None
MINIMUM_TOKENS = 64

def load_model():
    global BINO, TOKENIZER
    if BINO is None:
        BINO = Binoculars()
        TOKENIZER = BINO.tokenizer
        return "Model loaded onto GPU."
    return "Model already loaded."

def unload_model():
    global BINO, TOKENIZER
    if BINO is not None:
        del BINO
        gc.collect()
        torch.cuda.empty_cache()

        BINO = None
        TOKENIZER = None
        return "Model unloaded from GPU."
    return "Model already unloaded."


def count_tokens(text):
    if TOKENIZER is None:
        raise ValueError("Model is not loaded. Please load the model first.")
    return len(TOKENIZER(text).input_ids)

load_model()

@spaces.GPU
def run_detector(input_str):
    if BINO is None:
        gr.Error("Model is not loaded. Please load the model first.")
        return ""
    if count_tokens(input_str) < MINIMUM_TOKENS:
        gr.Warning(f"Too short length. Need minimum {MINIMUM_TOKENS} tokens to run Binoculars.")
        return ""
    
    start_time = time.time()  # Start the timer
    pred, score = BINO.predict(input_str, return_score=True)
    elapsed_time = time.time() - start_time
    
    return f"{score}", f"{pred}", f"{elapsed_time} seconds", f"{count_tokens(input_str)}"

def set_threshold(threshold):
    if BINO is None:
        gr.Error("Model is not loaded. Please load the model first.")
        return threshold
    try:
        threshold = float(threshold)
        BINO.set_threshold(threshold)
    except ValueError:
        BINO.set_threshold()
        gr.Error("Invalid threshold value. Please enter a numeric value.")
    
    return f"{BINO.get_threshold()}"


css = """
.green { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ccffcc; border-radius:0.5rem;}
.red { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ffad99; border-radius:0.5rem;}
.hyperlinks {
  display: flex;
  align-items: center;
  align-content: center;
  padding-top: 12px;
  justify-content: flex-end;
  margin: 0 10px; /* Adjust the margin as needed */
  text-decoration: none;
  color: #000; /* Set the desired text color */
}
"""

capybara_problem = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

with gr.Blocks(css=css,
               theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as app:

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<p><h1> Binoculars (zero-shot llm-text detector) with CPU inference</h1>")
        with gr.Column(scale=1):
            gr.HTML("""
            <p>
            <a href="https://arxiv.org/abs/2401.12070" target="_blank">code</a>
                    
            <a href="https://arxiv.org/abs/2401.12070" target="_blank">Original paper</a>
                
            <a href="https://github.com/AHans30/Binoculars" target="_blank">Original code</a>
            """, elem_classes="hyperlinks")
    with gr.Row():
        input_box = gr.Textbox(value=capybara_problem, placeholder="Enter text here", lines=8, label="Input Text")
    with gr.Row():
        threshold_box = gr.Textbox(value="", placeholder="Enter threshold here", lines=1, label="Threshold")
        change_threshold_button = gr.Button("Change threshold")
    with gr.Row():
        submit_button = gr.Button("Run", variant="primary")
        clear_button = gr.ClearButton()
    with gr.Row():
        load_button = gr.Button("Load Model")
        unload_button = gr.Button("Unload Model")
    with gr.Row():
        output_score = gr.Textbox(label="Score", value="")
        output_label = gr.Textbox(label="Label", value="")
        output_token_count = gr.Textbox(label="Token count", value="")
        output_time = gr.Textbox(label="Time elapsed", value="")
    with gr.Row():
        output_text = gr.Textbox(label="Info", value="")

    submit_button.click(run_detector, inputs=input_box, outputs=[output_score, output_label, output_time, output_token_count])
    clear_button.click(lambda: ("", "", "", "", "", ""), outputs=[input_box, output_text, output_score, output_label, output_time, output_token_count])
    load_button.click(load_model, outputs=output_text)
    unload_button.click(unload_model, outputs=output_text)
    change_threshold_button.click(set_threshold, inputs=threshold_box, outputs=threshold_box)
