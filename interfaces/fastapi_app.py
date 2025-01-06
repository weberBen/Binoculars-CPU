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
from starlette.concurrency import run_in_threadpool

from config import MODEL_MINIMUM_TOKENS, MAX_FILE_SIZE_BYTES, API_AUTHORIZED_KEYS, FLATTEN_BATCH
from interfaces.utils import bino_predict, count_tokens as bino_count_tokens, extract_pdf_content
from interfaces.bino_singleton import BINO, TOKENIZER
from interfaces.fastapi_utils import Token, api_key_header, create_access_token, validate_token
from interfaces.fastapi_model_type import PredictionResponse, SinglePredictionResponse, ModelInfo

BASE_API_URL = "/api/v1"

fastapi_app = FastAPI(
    title="Binoculars (zero-shot llm-text detector) with CPU inference",
    description="""Keep in mind that the same model is shared across all requests (GUI/API).
    Your request are queued and process later. Consequently waiting time does not reflect the actual
    processing time, which elapsed time parameter return for every request is.
    """,
    contact={
        "name": "Github",
        "url": "https://github.com/weberBen/Binoculars-cpu"
    }
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
    if api_key not in API_AUTHORIZED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    access_token = create_access_token(api_key)
    return {"access_token": access_token, "token_type": "bearer"}


# Route for prediction
@fastapi_app.post(f"{BASE_API_URL}/predict", include_in_schema=False) # Redirect route without trailing slash
@fastapi_app.post(
    f"{BASE_API_URL}/predict/",
    summary="AI/Human Content Detection",
    tags=["Prediction"],
    response_model=PredictionResponse,
)
async def process_content(
    contents: Optional[List[str]] = Form(
        None,
        description="Array of text content to analyze",
    ),
    files: Optional[List[UploadFile]] = File(
        None,
        description=f"Array of PDF files to analyze (max {MAX_FILE_SIZE_BYTES} Bytes each)",
    ),
    threshold: Optional[float] = Form(
        None,
        description="Threshold detection AI/Human",
    ),
    api_key: str = Depends(validate_token)
):
    global BINO, TOKENIZER

    request_received_at = datetime.now(timezone.utc)
    
    # Validate input
    if not contents and not files:
        raise HTTPException(
            status_code=400, 
            detail="Either text array or file array must be provided."
        )

    # Validate threshold
    if threshold is not None:
        try:
            threshold = float(threshold)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid threshold value"
            )

    documents = []
    
    # Process text contents
    if contents:
        for content in contents:
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Text content exceeds maximum size limit: {content[:50]}..."
                )
            documents.append(content)

    # Process files
    elif files:
        for file in files:
            try:
                if file.content_type == "application/pdf":
                    content = extract_pdf_content(file.file)
                    documents.append(content)
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type for file: {file.filename}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
    else:
        raise HTTPException(
            status_code=400, 
            detail="No valid content to process"
        )
    
    # long running task
    documents, score_list, threshold, pred_class_list, pred_label_list, total_gpu_time, total_token_count, document_length_list, document_chunks_count_list = await run_in_threadpool(bino_predict, BINO, documents, threshold=threshold)
    
    results = []
    for idx in range(len(documents)):
        result = SinglePredictionResponse(
            score = score_list[idx],
            label_class = pred_class_list[idx],
            label = pred_label_list[idx],
            content_length = document_length_list[idx],
            chunk_count = document_chunks_count_list[idx],
        )

        results.append(result)
    
    return PredictionResponse(
        model = ModelInfo(
            threshold = threshold,
            observer_model = BINO.observer_model_name,
            performer_model = BINO.performer_model_name,
        ),
        total_gpu_time = total_gpu_time,
        total_token_count = total_token_count,
        request_received_at = request_received_at,
        request_response_at = datetime.now(timezone.utc),
        results = results,
    )


def run_fastapi(port=8080, host='0.0.0.0'):
    uvicorn.run(fastapi_app, port=port, host=host)

if __name__ == '__main__':
    run_fastapi()