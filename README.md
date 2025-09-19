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
