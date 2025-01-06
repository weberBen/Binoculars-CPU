from env_utils import cast_string, cast_int, cast_float, cast_list, cast_bool


#%%

HUGGINGFACE_CONFIG = {
  "TOKEN": cast_string("HF_TOKEN", default="", empty_to_none=True)
}

BINOCULARS_THRESHOLD = cast_float("BINOCULARS_THRESHOLD")

BINOCULARS_OBSERVER_MODEL_NAME = cast_string("BINOCULARS_OBSERVER_MODEL_NAME", require=True)

BINOCULARS_PERFORMER_MODEL_NAME = cast_string("BINOCULARS_PERFORMER_MODEL_NAME", require=True)

BINOCULARS_FORCE_TO_CPU = cast_bool("BINOCULARS_FORCE_TO_CPU", default="False")

API_SECRET_KEY = cast_string("API_SECRET_KEY", require=True)

API_ENCRYPT_ALGORITHM = "HS256"

API_ACCESS_TOKEN_EXPIRE_MINUTES = cast_int("API_ACCESS_TOKEN_EXPIRE_MINUTES", default="60") 

API_AUTHORIZED_KEYS = cast_list("API_AUTHORIZED_KEYS")

MODEL_CHUNK_SIZE = cast_int("MODEL_CHUNK_SIZE", default="10000000")

MODEL_BATCH_SIZE = cast_int("MODEL_BATCH_SIZE", default="1")

MODEL_MINIMUM_TOKENS = cast_int("MODEL_MINIMUM_TOKENS", default="64")

MAX_FILE_SIZE_BYTES = cast_int("MAX_FILE_SIZE_BYTES", default="1000000")

FLATTEN_BATCH = cast_bool("FLATTEN_BATCH", default="true")