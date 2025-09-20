# Table of Contents

1. [Data Processing](#data-processing)
   - [Video Loading](#video-loading)
   - [Pose Detection](#pose-detection)
   - [Landmark Extraction](#landmark-extraction)
   - [Sequence Resampling](#sequence-resampling)
   - [Data Augmentation](#data-augmentation)
   - [CSV Creation](#csv-creation)

2. [Model Training](#model-training)
   - [Training Pipeline (`main.py`)](#training-pipeline-mainpy)
   - [Utility Modules](#utility-modules)

3. [Workflow](#workflow)

4. [Requirements & Installation](#requirements--installation)

5. [Notes](#notes)

## Data Processing

The `data_processing` module handles **video preprocessing, pose extraction, augmentation, and CSV generation**.

### Pipeline Overview

1. **Video Loading**  
   - Reads videos from a folder structure:  
     ```
     BASE_DIR/
       0 correct/
       1 no_initial_extension/
       2 incomplete_upper_phase/
       ...
     ```
   - Each subfolder corresponds to a label.

2. **Pose Detection**  
   - Uses **MediaPipe Pose** to detect body landmarks for each frame.
   - Automatically handles rotated or upside-down videos.

3. **Landmark Extraction**  
   - Extracts selected landmarks (e.g., shoulders, knees, ankles) and converts them to **pixel coordinates**.
   - Checks the first few frames for abnormal positions (legs above shoulders or shoulder asymmetry) and logs warnings if needed.

4. **Sequence Resampling**  
   - Each video sequence is **resampled** to a fixed length (`TARGET_LEN`) using linear interpolation.
   - Ensures all sequences have the same number of frames for downstream analysis.

5. **Data Augmentation**  
   - **Mirror:** flips sequences horizontally.  
     - Special handling for asymmetry labels (5 ↔ 6).  
   - **Random Shift:** applies small random x/y offsets for robustness.

6. **CSV Creation**  
   - Each frame of each video (original + augmentations) is stored as a row:  
     ```
     video_id, frame, label, nose_x, nose_y, left_shoulder_x, ...
     ```
   - Video ID is incremented for augmented variants.
   - Saves to `OUTPUT_CSV`, creating directories if needed.

### Example Usage

```python
from video_processing import process_single_video
import pandas as pd

video_file = "BASE_DIR/0 correct/sample.mp4"
label = 0
video_id = 0

rows = process_single_video((video_file, label, video_id))

# Save rows to CSV
df = pd.DataFrame(rows)
df.to_csv("output/landmarks.csv", index=False)
print("CSV saved to: output/landmarks.csv")
```

## Model Training

The `model/` folder provides an LSTM-based pipeline for classification.

### Main Pipeline (`main.py`)
```python
from utils import load_data, preprocess_data, split_data
from models import build_model
from metrics import plot_history, evaluate_model
import numpy as np
from config import EPOCHS, BATCH_SIZE

# Load and preprocess data
df = load_data()
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))
model = build_model(input_shape, num_classes)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

# Visualize
plot_history(history)
evaluate_model(model, X_test, y_test, np.unique(y))
```

### Utilities

- **utils.py** – loads CSV, preprocesses sequences for LSTM, scales features, splits train/test.  
- **models.py** – defines LSTM model with masking and fully connected layers.  
- **metrics.py** – plots training history, evaluates model (confusion matrix & classification report).  
- **config.py** – contains constants like `INPUT_CSV`, `EPOCHS`, `BATCH_SIZE`, `TEST_SIZE`.  

## Example Workflow

1. Preprocess videos and generate `landmarks.csv`.  
2. Set CSV path in `config.py` (`INPUT_CSV`).  
3. Run `main.py` to train and evaluate the LSTM model.  
4. Visualize metrics and confusion matrix.  

## Requirements & Installation

```bash
pip install -r requirements.txt
```
### Typical packages include:

 - **tensorflow**
 - **mediapipe**
 - **numpy**
 - **pandas**
 - **scikit-learn**
 - **matplotlib**

## Notes:

 - All video sequences are resampled to a fixed length for consistent LSTM input.
 - Data augmentation increases dataset robustness.
 - Asymmetric landmarks are handled correctly during horizontal flipping.
