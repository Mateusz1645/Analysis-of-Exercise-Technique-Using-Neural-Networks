import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from config import BASE_DIR, OUTPUT_CSV, LANDMARKS_INFO, RANDOM_ADD
from video_processing import process_single_video
import absl.logging
import logging

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

absl.logging.set_verbosity(absl.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Logging configuration ===
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
    Saves a list of dictionaries (landmark data) to a CSV file.
    """
    fieldnames = ['video_id', 'frame', 'label']
    for idx in LANDMARKS_INFO:
        name = LANDMARKS_INFO[idx]
        fieldnames.extend([f"{name}_x", f"{name}_y"])
    df = pd.DataFrame(all_rows, columns=fieldnames)
    df.to_csv(output_csv, index=False)
    line = f"All data saved to: {output_csv}"
    logging.info(line)

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

def main_test():
    # Gather all videos from the base directory
    video_list = gather_videos(BASE_DIR)
    if not video_list:
        logging.error("No videos found in the base directory!")
        return

    # Take only the first video for testing
    test_video = video_list[0]
    logging.info(f"Testing with video: {test_video[0]}, label: {test_video[1]}, id: {test_video[2]}")

    # Process the single video
    all_rows = process_single_video(test_video)
    print(all_rows)
    # Save the processed data to a temporary CSV
    test_output_csv = OUTPUT_CSV.replace(".csv", "_test.csv")
    save_to_csv(all_rows, test_output_csv)
    logging.info(f"Test processing completed. Output saved to {test_output_csv}")

if __name__ == "__main__":
    main()
    # main_test()
