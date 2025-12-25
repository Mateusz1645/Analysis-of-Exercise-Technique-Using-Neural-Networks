import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import BASE_DIR, OUTPUT_CSV, LANDMARKS_INFO, RANDOM_ADD
from video_processing import process_single_video
import absl.logging
import logging

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

absl.logging.set_verbosity(absl.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Logging configuration
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", mode="w"),
        logging.StreamHandler()
    ]
)

def gather_videos(base_dir):
    """
    Collects all video files along with their labels and IDs.
    """
    video_list = []
    video_id = 0
    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        try:
            label = int(label_name.split(" ")[0])
        except ValueError:
            continue
        for video_file in glob(os.path.join(label_path, "*.mp4")):
            video_list.append((video_file, label, video_id))
            video_id += 2 + RANDOM_ADD  # normal + mirror + augmentation
    return video_list

def save_to_csv(all_rows, output_csv):
    """
    Saves a list of dictionaries (landmark data) to a CSV file,
    calculating joint angles and adding them as additional columns.
    """
    # Base columns: video ID, frame, label + landmark coordinates
    fieldnames = ['video_id', 'frame', 'label']
    for idx in LANDMARKS_INFO:
        name = LANDMARKS_INFO[idx]
        fieldnames.extend([f"{name}_x", f"{name}_y"])
    
    # Create DataFrame from raw landmark data
    df = pd.DataFrame(all_rows, columns=fieldnames)
    
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    # Function to calculate joint angles for a single row
    def calculate_angles(row):
        angles = {}
        # Elbows
        angles['left_elbow_angle'] = calculate_angle(
            [row['left_shoulder_x'], row['left_shoulder_y']],
            [row['left_elbow_x'], row['left_elbow_y']],
            [row['left_wrist_x'], row['left_wrist_y']]
        )
        angles['right_elbow_angle'] = calculate_angle(
            [row['right_shoulder_x'], row['right_shoulder_y']],
            [row['right_elbow_x'], row['right_elbow_y']],
            [row['right_wrist_x'], row['right_wrist_y']]
        )
        # Shoulders
        angles['left_shoulder_angle'] = calculate_angle(
            [row['left_elbow_x'], row['left_elbow_y']],
            [row['left_shoulder_x'], row['left_shoulder_y']],
            [row['left_hip_x'], row['left_hip_y']]
        )
        angles['right_shoulder_angle'] = calculate_angle(
            [row['right_elbow_x'], row['right_elbow_y']],
            [row['right_shoulder_x'], row['right_shoulder_y']],
            [row['right_hip_x'], row['right_hip_y']]
        )
        # Hips
        angles['left_hip_angle'] = calculate_angle(
            [row['left_shoulder_x'], row['left_shoulder_y']],
            [row['left_hip_x'], row['left_hip_y']],
            [row['left_knee_x'], row['left_knee_y']]
        )
        angles['right_hip_angle'] = calculate_angle(
            [row['right_shoulder_x'], row['right_shoulder_y']],
            [row['right_hip_x'], row['right_hip_y']],
            [row['right_knee_x'], row['right_knee_y']]
        )
        # Knees
        angles['left_knee_angle'] = calculate_angle(
            [row['left_hip_x'], row['left_hip_y']],
            [row['left_knee_x'], row['left_knee_y']],
            [row['left_ankle_x'], row['left_ankle_y']]
        )
        angles['right_knee_angle'] = calculate_angle(
            [row['right_hip_x'], row['right_hip_y']],
            [row['right_knee_x'], row['right_knee_y']],
            [row['right_ankle_x'], row['right_ankle_y']]
        )

        angles['left_knee_ankle_dist'] = calculate_distance(
            [row['left_knee_x'], row['left_knee_y']],
            [row['left_ankle_x'], row['left_ankle_y']]
        )

        angles['right_knee_ankle_dist'] = calculate_distance(
            [row['right_knee_x'], row['right_knee_y']],
            [row['right_ankle_x'], row['right_ankle_y']]
        )

        return pd.Series(angles)
    
    # Add joint angle columns to the DataFrame
    angles_df = df.apply(calculate_angles, axis=1)
    df = pd.concat([df, angles_df], axis=1)

    # SCALE ONLY KNEE-ANKLE DISTANCES TO [0, 1]
    dist_cols = ['left_knee_ankle_dist', 'right_knee_ankle_dist']

    scaler = MinMaxScaler()
    df[dist_cols] = scaler.fit_transform(df[dist_cols])
    
    # Save the final DataFrame to CSV
    df.to_csv(output_csv, index=False)
    logging.info(f"All data saved to: {output_csv}")


def main():
    # Gather all videos from the base directory
    video_list = gather_videos(BASE_DIR)
    line = f"Found {len(video_list)} videos to process."
    logging.info(line)

    # Process all videos in parallel
    all_rows = []
    with ProcessPoolExecutor() as executor:
        for i, rows in enumerate(executor.map(process_single_video, video_list), 1):
            all_rows.extend(rows)
            line = f"Processed {i}/{len(video_list)} videos ({i / len(video_list) * 100:.1f}%)"
            logging.info(line)

    # Save the processed data to CSV
    save_to_csv(all_rows, OUTPUT_CSV)

if __name__ == "__main__":
    main()
