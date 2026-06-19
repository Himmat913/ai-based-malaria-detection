from functools import lru_cache

import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = "himmat123/malaria-model"
FILENAME = "malaria_model.h5"


@lru_cache(maxsize=1)
def get_model_path() -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=FILENAME)


@lru_cache(maxsize=1)
def load_malaria_model():
    model_path = get_model_path()
    return tf.keras.models.load_model(model_path, compile=False)