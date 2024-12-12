from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security.api_key import APIKeyHeader
from typing import Union, Optional, List
import io
import uvicorn
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import os

from interfaces.utils import bino_predict, count_tokens as bino_count_tokens, extract_pdf_content, MINIMUM_TOKENS, MAX_FILE_SIZE
from interfaces.bino_singleton import BINO, TOKENIZER
from interfaces.fastapi_utils import Token, api_key_header, AUTHORIZED_API_KEYS, create_access_token, validate_token, FASTAPI_NO_LONG_RUNNING_TASK, NO_LONG_RUNNING_TASK_MESSAGE

BASE_API_URL = "/api/v1"


fastapi_app = FastAPI(
    name="Binoculars (zero-shot llm-text detector) with CPU inference",
    description="""Keep in mind that the same model is shared across all requests (GUI/API).
    Your request are queued and process later. Consequently waiting time does not reflect the actual
    processing time, which elapsed time parameter return for every request is.
    """
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Route to exchange API key for bearer token
@fastapi_app.post(
    f"{BASE_API_URL}/auth/token", 
    response_model=Token,
    summary="Exchange API Key for Bearer Token",
    description="""Provide your API key to receive a bearer token that can be used to authenticate other endpoints.
    Add you API key to the header `X-API-Key`.
    Default api key is `my_api_key_1`
    """,
    tags=["Authentication"],
)
async def get_token(api_key: str = Security(api_key_header)):
    if api_key not in AUTHORIZED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    access_token = create_access_token(api_key)
    return {"access_token": access_token, "token_type": "bearer"}


# Route for prediction
@fastapi_app.post(
    f"{BASE_API_URL}/predict/",
    summary="AI/Human Content Detection",
    tags=["Prediction"],
)
async def process_content(
    content: Optional[str] = Form(
        None,
        description="Text content to analyze",
    ),
    file: Optional[UploadFile] = File(
        None,
        description=f"PDF file to analyze (max {MAX_FILE_SIZE} Bytes)",
    ),
    api_key: str = Depends(validate_token)
):
  
    global BINO, TOKENIZER

    if FASTAPI_NO_LONG_RUNNING_TASK:
        return {
            "message": NO_LONG_RUNNING_TASK_MESSAGE,
            "score": 0.8846334218978882,
            "class": 0,
            "label": "Most likely AI-generated",
            "total_elapsed_time": 23.35552716255188,
            "total_token_count": 134,
            "content_length": 661,
            "chunk_count": 1,
        }

    if not content and not file:
        raise HTTPException(status_code=400, detail="Either text or file must be provided.")

    if content:
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Text content exceeds maximum size limit.")

    elif file:
        try:
            if file.content_type == "application/pdf":
                # Extract text from the PDF file
                content = extract_pdf_content(file.file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type.")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
    else:
      raise HTTPException(status_code=400, detail="Invalid input.")
    
    if bino_count_tokens(TOKENIZER, content) < MINIMUM_TOKENS:
        raise HTTPException(status_code=400, detail=f"Too short length. Need minimum {MINIMUM_TOKENS} tokens to run.")
    
    content, score, pred_class, pred_label, total_elapsed_time, total_token_count, content_length, chunk_count = bino_predict(BINO, content)

    return {
        "score": score,
        "class": pred_class,
        "label": pred_label,
        "total_elapsed_time": total_elapsed_time,
        "total_token_count": total_token_count,
        "content_length": content_length,
        "chunk_count": chunk_count,
      }

def run_fastapi(port=8080, host='0.0.0.0'):
    uvicorn.run(fastapi_app, port=port, host=host)

if __name__ == '__main__':
    run_fastapi()