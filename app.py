from demo.demo import app

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(show_api=True, debug=True, share=False, max_file_size=1000000)
