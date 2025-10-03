import os
from dotenv import load_dotenv

# === LOAD ENV VARIABLES ===
load_dotenv()

# === FILE PATHS ===
INPUT_CSV = os.getenv("INPUT_CSV")

# === HYPERPARAMETERS ===
EPOCHS = 20
BATCH_SIZE = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42
