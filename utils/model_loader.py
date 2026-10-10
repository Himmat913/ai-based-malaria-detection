import os
from functools import lru_cache

import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = "himmat123/malaria-model"
FILENAME = "malaria_model.h5"

# Optional: point MALARIA_MODEL_PATH at a local copy of malaria_model.h5 to
# skip the Hugging Face download (offline use). When unset, behaviour is
# unchanged: the file is fetched from REPO_ID on first use and cached.
LOCAL_MODEL_ENV = "MALARIA_MODEL_PATH"


@lru_cache(maxsize=1)
def get_model_path() -> str:
    local_path = os.environ.get(LOCAL_MODEL_ENV, "").strip()
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                f"{LOCAL_MODEL_ENV} is set to '{local_path}', but no file exists there."
            )
        return local_path
    return hf_hub_download(repo_id=REPO_ID, filename=FILENAME)


@lru_cache(maxsize=1)
def load_malaria_model():
    model_path = get_model_path()
    return tf.keras.models.load_model(model_path, compile=False)
