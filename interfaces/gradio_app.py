__all__ = ["app"]

import gradio as gr
import spaces
from config import MODEL_MINIMUM_TOKENS, MAX_FILE_SIZE_BYTES
from interfaces.utils import bino_predict, count_tokens as count_tokens, extract_pdf_content
from interfaces.bino_singleton import BINO, TOKENIZER
from binoculars.detector import BINOCULARS_THRESHOLD


@spaces.GPU
def handle_submit_text(threshold, input_text, pdf_file):
    try:
        threshold = float(threshold)
    except ValueError:
        raise gr.Error("Invalid threshold value. Please enter a numeric value.")
    
    content = input_text
    if pdf_file is not None:
        content = extract_pdf_content(pdf_file.name)
    
    content, score, threshold, pred_class, pred_label, total_gpu_time, total_token_count, content_length, chunk_count = bino_predict(BINO, content, threshold=threshold)

    return content[0], f"{score[0]}", pred_label[0], f"{total_gpu_time} seconds", total_token_count, content_length[0], chunk_count[0]


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

/* Needed because mount_gradio_app version 4.x does not support show_api attributes */
footer > button:first-of-type {
    display: none !important;
}
"""

capybara_problem = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

with gr.Blocks(css=css,
               theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])
               ) as gradio_app:

    with gr.Row():
        with gr.Column(scale=3):
                with gr.Row():
                    gr.HTML("<h1> Binoculars (zero-shot llm-text detector) with CPU inference</h1>")
                with gr.Row():
                    gr.HTML("""<h3> Keep in mind that the same model is shared across all requests (GUI/API).
                                    Your request are queued and process later. Consequently waiting time does not reflect the actual
                                    processing time, which elapsed time parameter return for every request is.
                                </h3>
                            """)
        with gr.Column(scale=1):
                gr.HTML("""
                    <p>
                        <a href="https://huggingface.co/spaces/ben-weber/Binoculars-CPU" target="_blank">HuggingFace space 🤗</a>

                        <a href="https://github.com/weberBen/Binoculars-cpu" target="_blank">Github 🐙</a>
                                
                        <a href="https://arxiv.org/abs/2401.12070" target="_blank">Original paper</a>
                            
                        <a href="https://github.com/AHans30/Binoculars" target="_blank">Original code</a>
                    </p>
                    """,elem_classes="hyperlinks"
                )
    
    with gr.Row():
        gr.HTML(f"<h5> Observer model: {BINO.observer_model_name} <h5>")
        gr.HTML(f"<h5> Performer model: {BINO.performer_model_name} <h5>")
    
    with gr.Row():
        input_box = gr.Textbox(value=capybara_problem, placeholder="Enter text here", lines=8, label="Input Text")
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single")
    with gr.Row():
        threshold_box = gr.Textbox(value=f"{BINOCULARS_THRESHOLD}", placeholder="Enter threshold here", lines=1, label="Detection threshold")
        reset_threshold_button = gr.Button("Reset threshold")
    with gr.Row():
        submit_button = gr.Button("Run", variant="primary")
        clear_button = gr.ClearButton()
    with gr.Row():
        output_score = gr.Textbox(label="Score", value="")
        output_label = gr.Textbox(label="Label", value="")
        output_token_count = gr.Textbox(label="Token count", value="")
        output_content_length = gr.Textbox(label="Content length (char)", value="")
        output_chunk_count = gr.Textbox(label="Chunk count", value="")
        output_time = gr.Textbox(label="GPU time elapsed", value="")

    with gr.Row():
        gr.HTML("""<p><h2> See <a href="/docs" target="_blank">API doc</a> 🚀 </h2></p>""")


    clear_button.click(lambda: ("", "", "", "", "", "", "", None),
                       outputs=[input_box, output_score, output_label, output_time, output_token_count,
                                output_content_length, output_chunk_count, pdf_input]
                    )
    
    reset_threshold_button.click(lambda: (BINOCULARS_THRESHOLD), outputs=threshold_box)

    submit_button.click(
        lambda threshold_input, input_text, pdf_file: (*handle_submit_text(threshold_input, input_text, pdf_file), None),
        inputs=[threshold_box, input_box, pdf_input],
        outputs=[input_box, output_score, output_label, output_time,
                 output_token_count, output_content_length,
                 output_chunk_count, pdf_input
                ]
    )

def run_gradio(show_api=False, debug=True, share=False):
    # IMPORTANT : show_api layout is override in css
    gradio_app.launch(show_api=show_api, debug=debug, share=share, max_file_size=MAX_FILE_SIZE_BYTES)

if __name__ == "__main__":
    # Launch the Gradio interface
    run_gradio()