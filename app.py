from demo.demo import app
from demo.utils import MAX_FILE_SIZE

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(show_api=False, debug=True, share=False, max_file_size=MAX_FILE_SIZE)
