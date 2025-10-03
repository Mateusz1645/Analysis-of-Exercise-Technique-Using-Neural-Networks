import os
from dotenv import load_dotenv

# === LOAD ENV VARIABLES ===
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", "./data")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "./output/landmarks.csv")
WARNING_LOG = os.getenv("WARNING_TXT", "./output/warnings.txt")

# === PROCESSING PARAMETERS ===
TARGET_LEN = 90           # number of frames after resampling
RANDOM_ADD = 2            # number of random shift augmentations
MAX_SHIFT = 10            # maximum pixel shift for random augmentation
SHOW_PREVIEW = False      # show video preview while processing
MISSING_THRESHOLD = 10
# === LANDMARKS ===
LANDMARKS_INFO = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle"
}
SELECTED_LANDMARKS = list(LANDMARKS_INFO.keys())

LANDMARKS_INFO_MIRRORED = {
    0: "nose",
    12: "left_shoulder", 11: "right_shoulder",
    14: "left_elbow", 13: "right_elbow",
    16: "left_wrist", 15: "right_wrist",
    24: "left_hip", 23: "right_hip",
    26: "left_knee", 25: "right_knee",
    28: "left_ankle", 27: "right_ankle"
}
SELECTED_LANDMARKS_MIRRORED = list(LANDMARKS_INFO_MIRRORED.keys())

# === WARNING CHECK PARAMETERS ===
CHECK_FRAMES = 5  # number of first frames to check for abnormal positions
