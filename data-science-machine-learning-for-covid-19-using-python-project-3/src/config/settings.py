from pathlib import Path

# =========================================================
# Project Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"


DATA_DIR = BASE_DIR / "src" / "data"
DATASET_PATH = DATA_DIR / "updated_train.csv"


# =========================================================
# NLP Settings
# =========================================================

LANGUAGE = "english"
REMOVE_PUNCTUATION = True

# =========================================================
# Word2Vec Settings
# =========================================================

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_WORKERS = 4

# =========================================================
# Tokenizer / Sequences
# =========================================================

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

# =========================================================
# LSTM Model Settings
# =========================================================

LSTM_UNITS = 128
EMBEDDING_TRAINABLE = False

# =========================================================
# Training Settings
# =========================================================

BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2

# =========================================================
# Runtime
# =========================================================

RANDOM_STATE = 42
