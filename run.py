from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Union
import io
import uvicorn
from binoculars import Binoculars
import torch

from demo.utils import bino_predict, count_tokens as bino_count_tokens, extract_pdf_content, MINIMUM_TOKENS, MAX_FILE_SIZE

BINO = Binoculars()
TOKENIZER = BINO.tokenizer

app = FastAPI()

@app.post("/predict/")
async def process_content(
    content: Union[str, None] = Form(None),
    file: Union[UploadFile, None] = None
):
  
    global BINO, TOKENIZER

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

    


if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')