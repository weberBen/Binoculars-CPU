
from fastapi import Header, Request
import gradio as gr
from fastapi.responses import RedirectResponse

from interfaces.utils import MAX_FILE_SIZE
from interfaces.gradio_app import gradio_app, run_gradio
from interfaces.fastapi_app import fastapi_app, run_fastapi

APP_URL = "app"

@fastapi_app.get("/")
async def root(request: Request, accept: str = Header(default=None)):
    # Get the base URL from the request object
    base_url = str(request.base_url).rstrip('/').replace("http://", "https://")

    # Redirct text/html request to the current app url in HuggingFace cause requests
    # to be blocked. Thus display information JSON as route to manually redirect user
    # to the appropriate URL

    return {
        "message": "App is running",
        "endpoints": {
            "app": f"{base_url}/{APP_URL}",
            "api_docs": f"{base_url}/docs",
            "home": base_url,
        },
    }


app = gr.mount_gradio_app(fastapi_app, gradio_app, path=f"/{APP_URL}", max_file_size=MAX_FILE_SIZE)

# USAGE : uvicorn app:app --host 0.0.0.0 --port 7860