import os

HUGGINGFACE_CONFIG = {
  "TOKEN": os.getenv("HF_TOKEN", None)
}

# get threshold from env var
BINOCULARS_THRESHOLD =  os.getenv("BINOCULARS_THRESHOLD", "")
BINOCULARS_THRESHOLD = float(BINOCULARS_THRESHOLD)
assert(type(BINOCULARS_THRESHOLD) is float)

BINOCULARS_OBSERVER_MODEL_NAME = os.getenv("BINOCULARS_OBSERVER_MODEL_NAME", "").strip()
assert(len(BINOCULARS_OBSERVER_MODEL_NAME) > 0)

BINOCULARS_PERFORMER_MODEL_NAME = os.getenv("BINOCULARS_PERFORMER_MODEL_NAME", "").strip()
assert(len(BINOCULARS_PERFORMER_MODEL_NAME) > 0)

BINOCULARS_FORCE_TO_CPU = os.getenv("BINOCULARS_FORCE_TO_CPU", "False").lower() in ("true", "1", "yes")

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
assert(len(API_SECRET_KEY.strip()) > 0), "Invalid secret api key"

API_ENCRYPT_ALGORITHM = "HS256"
API_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("API_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

def parse_authorized_keys():
    keys = os.getenv("API_AUTHORIZED_API_KEYS", "").split("|")
    keys = [x for x in keys if x.strip() != '']

    return keys

API_AUTHORIZED_API_KEYS = parse_authorized_keys()

MODEL_CHUNK_SIZE = os.getenv("MODEL_CHUNK_SIZE", "10000")
MODEL_CHUNK_SIZE = int(MODEL_CHUNK_SIZE)
assert(type(MODEL_CHUNK_SIZE) is int)

MODEL_BATCH_SIZE = os.getenv("MODEL_BATCH_SIZE", "1")
MODEL_BATCH_SIZE = int(MODEL_BATCH_SIZE)
assert(type(MODEL_BATCH_SIZE) is int)

MODEL_MINIMUM_TOKENS = os.getenv("MODEL_MINIMUM_TOKENS", "64")
MODEL_MINIMUM_TOKENS = int(MODEL_MINIMUM_TOKENS)
assert(type(MODEL_MINIMUM_TOKENS) is int)

MAX_FILE_SIZE_BYTES=os.getenv("MAX_FILE_SIZE_BYTES", "1000000")
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_BYTES)
assert(type(MAX_FILE_SIZE_BYTES) is int)