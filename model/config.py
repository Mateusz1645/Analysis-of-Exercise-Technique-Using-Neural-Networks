import os
from dotenv import load_dotenv

# === LOAD ENV VARIABLES ===
load_dotenv()

# === FILE PATHS ===
INPUT_CSV = os.getenv("INPUT_CSV")
INPUT_TEST_CSV = os.getenv("INPUT_TEST_CSV", "landmarks_test.csv")

# === HYPERPARAMETERS ===
EPOCHS = 20
BATCH_SIZE = 16
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
