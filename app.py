
from fastapi import Header, Request
import gradio as gr
import uvicorn
from fastapi.responses import RedirectResponse
import os

from interfaces.utils import MAX_FILE_SIZE
from interfaces.gradio_app import gradio_app
from interfaces.fastapi_app import fastapi_app, run_fastapi

APP_URL = "app"

@fastapi_app.get("/")
async def root(accept: str = Header(None)):
    # Check if the request accepts HTML (browser request)
    if accept and "text/html" in accept.lower():
        return RedirectResponse(url=f"/{APP_URL}")
    
    # Return JSON for API requests
    return {
        "message": "API is running",
        "endpoints": {
            "app": f"/{APP_URL}",
            "endpoints": '/docs'
        }
    }

gradio_app.queue()

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(fastapi_app, gradio_app, path=f"/{APP_URL}", max_file_size=MAX_FILE_SIZE)

# Run the app
if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = float(os.getenv("SERVER_PORT", "8080"))
    
    run_fastapi(host=host, port=port)
    