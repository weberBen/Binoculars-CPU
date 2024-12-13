
import time
import numpy as np
from PyPDF2 import PdfReader
from config import MODEL_CHUNK_SIZE, MODEL_BATCH_SIZE, MAX_FILE_SIZE_BYTES, MODEL_MINIMUM_TOKENS, FLATTEN_BATCH
from typing import List, Tuple, Union

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


def split_text(input_text: str, chunk_size: int = MODEL_CHUNK_SIZE):
    """
    Split text into chunks using NumPy for improved performance.
    
    Args:
        input_text (str): The input text to be split
        chunk_size (int): Size of each chunk in characters (default: 5000)
    
    Returns:
        list: List of text chunks
    """
    
    # Convert string to numpy array of characters
    text_array = np.array(list(input_text))
    
    # Create array of indices for splitting
    indices = np.arange(0, len(text_array), chunk_size)
    
    # Split text into chunks using array operations
    chunks = [
        ''.join(text_array[i:i + chunk_size])
        for i in indices
    ]
    
    return chunks


def text_to_chunks(documents: Union[list[str], str], chunk_size: int = MODEL_CHUNK_SIZE, batch_size: int = MODEL_BATCH_SIZE):
    """
    Convert documents into optimized chunks for model processing while respecting GPU memory constraints.

    This function processes documents into chunks that can be efficiently handled by the model, taking into
    account GPU VRAM limitations. It performs two levels of chunking:
    1. Splits documents into chunks based on model's maximum input size
    2. Groups compatible chunks into "super chunks" to optimize batch processing

    Parameters
    ----------
    documents : Union[list[str], str]
        Input text document(s). Can be either a single document string or a list of document strings.
    chunk_size : int, optional
        Maximum size of individual chunks in characters (default: MODEL_CHUNK_SIZE)
    batch_size : int, optional
        Number of chunks that can be processed together (default: MODEL_BATCH_SIZE)

    Returns
    -------
    tuple
        Contains three elements:
        - list[str]: Flattened list of all chunks across all documents
        - list[tuple[int, int]]: Document chunk indices indicating which chunks belong to which document
            Format: [(start_idx, end_idx), ...] where each tuple represents the chunk range for a document
        - list[tuple[int, int]]: Super chunk indices for batch processing
            Format: [(start_idx, end_idx), ...] where each tuple represents chunks that can be processed together

    Example
    -------
    Given:
    - chunk_size = 200 characters
    - batch_size = 1
    - batch_chunk_size = 200 * 1 = 200 characters total per batch

    For documents with the following chunk distribution:
    [
        doc1: [chunk1(200), chunk2(111)],  # Document 1 has 2 chunks of length 200, 111
        doc2: [chunk3(80)],                # Document 2 has 1 chunk of length 80
        doc3: [chunk4(96)],                # Document 3 has 1 chunk of length 96
        doc4: [chunk5(100)],               # Document 4 has 1 chunk of length 100
        doc5: [chunk6(169)],               # Document 5 has 1 chunk of length 169
        doc6: [chunk7(31)]                 # Document 6 has 1 chunk of length 31
    ]

    The function returns:
    1. Flattened chunks list
    2. Document chunk indices: [(0,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
        - (0,2) means document 1 contains chunks[0:2]
        - (2,3) means document 2 contains chunks[2:3]
        And so on...
    3. Super chunk indices: [(0,1), (1,3), (3,5), (5,7)]
        - (0,1) processes the first chunk of document 1
        - (1,3) processes the second chunk of document 1 and document 2
        And so on...

    Notes
    -----
    - Chunks within super chunks must have a combined size < batch_chunk_size
    - Super chunks are optimized to maximize GPU memory usage while staying within limits
    - The last super chunk may be smaller than others to accommodate remaining chunks
    """

    model_chunk_size = chunk_size
    batch_chunk_size = model_chunk_size * batch_size

    documents = [documents] if isinstance(documents, str) else documents

    chunk_list = []
    document_chunk_indices = []
    super_chunk_indices = []

    chunk_size = 0
    super_chunk_length = 0

    for idx, document in enumerate(documents):
        chunks = split_text(document, chunk_size=model_chunk_size)

        for chunk in chunks:
            chunk_size += len(chunk)
            super_chunk_length += 1

            if chunk_size > batch_chunk_size:
                last_elem_idx = super_chunk_indices[-1][-1] if len(super_chunk_indices) > 0 else 0
                super_chunk_indices.append((last_elem_idx, last_elem_idx + (super_chunk_length - 1)))
                super_chunk_length = 1
                chunk_size = len(chunk)
            
            chunk_list.append(chunk)
        
        last_elem_idx = document_chunk_indices[-1][-1] if len(document_chunk_indices) > 0 else 0
        document_chunk_indices.append((last_elem_idx, last_elem_idx + len(chunks)))

    if super_chunk_length > 0:
        last_elem_idx = super_chunk_indices[-1][-1] if len(super_chunk_indices) > 0 else 0
        super_chunk_indices.append((last_elem_idx, last_elem_idx + super_chunk_length))
    
    return chunk_list, document_chunk_indices, super_chunk_indices


def run_bino(bino, batch: list[str], threshold=None):
  start_time = time.time()  # Start the timer
  pred_class, pred_label, np_score, threshold = bino.predict(batch, return_fields=["class", "label", "score", "threshold"], threshold=threshold)
  gpu_time = time.time() - start_time
  
  return np_score, threshold, pred_class, pred_label, gpu_time

def bino_predict(bino, documents: Union[list[str], str], threshold=None):
    documents = [documents] if isinstance(documents, str) else documents
    document_length_list = []
    total_token_count = 0

    for document in documents:
        # Attempt to encode the string to bytes and then decode back to a string
        try:
            document.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError as e:
           raise UnicodeDecodeError(f"Invalid document encoding: {e}")

        document_length = len(document)
        document_length_list.append(document_length)
        if document_length > MAX_FILE_SIZE_BYTES:
            raise Exception(f"Document over limit of {MAX_FILE_SIZE_BYTES} Bytes")
        
        token_count = count_tokens(bino.tokenizer, document)
        total_token_count += token_count
        if token_count < MODEL_MINIMUM_TOKENS:
            raise Exception(f"Too short length. Need minimum {MODEL_MINIMUM_TOKENS} tokens to run.")

    total_gpu_time = 0
    score_list = []
    document_scores = []
    document_pred_classes = []
    document_pred_labels = []

    if FLATTEN_BATCH:
        # Flatten chunks across all documents to optimize performance
        # May introduce score variation on individual sample because of batch operation
        chunks, document_chunk_indices, super_chunk_indices = text_to_chunks(documents)

        for (i, j) in super_chunk_indices:
            batch = chunks[i:j]
            np_score, threshold, pred_class, pred_label, gpu_time = run_bino(bino, batch, threshold=threshold)
            total_gpu_time += gpu_time
            score_list.extend(np_score.flatten().tolist())

        for (i, j) in document_chunk_indices:
            document_chunk_scores = np.array(score_list[i:j])

            score = document_chunk_scores.mean()

            document_scores.append(score)
            document_pred_classes.append(bino.score_to_class(score, threshold=threshold))
            document_pred_labels.append(bino.score_to_label(score, threshold=threshold))
        
        document_chunks_count_list = [j-i for (i, j) in document_chunk_indices]
    else:
        # Processing document one by one without chunks flatten across documents
        # May reduce performance
        document_chunks_count_list = []

        for document in documents:
            document_chunks = split_text(document)
            batches = [document_chunks[i:i + MODEL_BATCH_SIZE] for i in range(0, len(document_chunks), MODEL_BATCH_SIZE)]

            score_list = []
            for batch in batches:
                np_score, threshold, pred_class, pred_label, gpu_time = run_bino(bino, batch, threshold=threshold)
                total_gpu_time += gpu_time
                score_list.extend(np_score.flatten().tolist())
            
            score_list = np.array(score_list)
            score = score_list.mean()

            document_scores.append(score)
            document_pred_classes.append(bino.score_to_class(score, threshold=threshold))
            document_pred_labels.append(bino.score_to_label(score, threshold=threshold))
            document_chunks_count_list.append(len(document_chunks))

    return documents, document_scores, threshold, document_pred_classes, document_pred_labels, total_gpu_time, total_token_count, document_length_list, document_chunks_count_list
