import os
from dotenv import load_dotenv

# LOAD ENV VARIABLES
load_dotenv()

# FILE PATHS
INPUT_CSV = os.getenv("INPUT_CSV")

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
TEST_SIZE = 0.15
RANDOM_STATE = 42
VAL_SIZE = 0.15
