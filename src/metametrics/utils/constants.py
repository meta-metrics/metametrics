import numpy as np
import os

VERSION = "0.0.1"

DATASET_RANDOM_SEED = 1
MODEL_RANDOM_SEED = 1

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

# Default column names
TEXT_SRC = "text_src"
TEXT_HYP = "text_hyp"
TEXT_REF = "text_ref"
IMG_SRC = "img_src"
CHOSEN = "chosen"
REJECTED = "rejected"
TARGET = "target_score"

HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = None
