
import time
from PyPDF2 import PdfReader

TEXT_SPLIT_CHAR = 10000
MAX_FILE_SIZE = 100000
MINIMUM_TOKENS = 64

def count_tokens(tokenizer, text):
    return len(tokenizer(text).input_ids)

def extract_pdf_content(file_data):
    """
    Extract text content from the uploaded PDF file, ignoring images and non-text elements.
    Return an array of strings, each with a maximum length of 5000 characters.
    """
    reader = PdfReader(file_data)
    content = ""

    # Iterate through each page and extract text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            content += text + "\n"
    return content

def split_text(input_text):
    result = []
    # Split content into chunks of 5000 characters
    while input_text:
        chunk = input_text[:TEXT_SPLIT_CHAR]
        result.append(chunk)
        input_text = input_text[TEXT_SPLIT_CHAR:]
    
    return result

def run_bino(bino, content, threshold=None):
  start_time = time.time()  # Start the timer
  pred_class, pred_label, score, threshold = bino.predict(content, return_fields=["class", "label", "score", "threshold"], threshold=threshold)
  elapsed_time = time.time() - start_time
  
  return score, threshold, pred_class, pred_label, elapsed_time, count_tokens(bino.tokenizer, content)

def bino_predict(bino, content, threshold=None):
    # Attempt to encode the string to bytes and then decode back to a string
    content = content.encode('utf-8').decode('utf-8')

    if len(content) > MAX_FILE_SIZE:
        raise Exception(f"Input text over limit of {MAX_FILE_SIZE} Bytes")

    total_score = 0.0
    total_elapsed_time = 0
    total_token_count = 0

    content_list = split_text(content)
    for content in content_list:
        score, threshold, pred_class, pred_label, elapsed_time, token_count = run_bino(bino, content, threshold=threshold)

        total_score += score
        total_elapsed_time += elapsed_time
        total_token_count += token_count

    score = total_score/len(content_list)
    pred_class = bino.score_to_class(score, threshold=threshold)
    pred_label = bino.score_to_label(score, threshold=threshold)
    content = " ".join(content_list)

    return content, score, threshold, pred_class, pred_label, total_elapsed_time, total_token_count, len(content), len(content_list)
