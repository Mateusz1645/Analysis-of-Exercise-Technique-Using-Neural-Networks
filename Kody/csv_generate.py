import cv2
import mediapipe as mp
import csv
import numpy as np
from scipy.interpolate import interp1d
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv

# === KONFIGURACJA ===
base_dir = os.getenv("BASE_DIR")
output_csv = os.getenv("OUTPUT_CSV")
target_len = 90
show_preview = False

# === LANDMARKI ===
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

# === FUNKCJE ===
def resample_sequence(sequence: np.ndarray, target_len: int = 60) -> np.ndarray:
    if sequence.shape[0] == 0:
        return np.zeros((target_len, len(SELECTED_LANDMARKS) * 2))
    old_time = np.linspace(0, 1, sequence.shape[0])
    new_time = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        interpolator = interp1d(old_time, sequence[:, i], kind='linear', fill_value="extrapolate")
        resampled[:, i] = interpolator(new_time)
    return resampled

def process_video(args):
    video_file, label, video_id = args
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Nie można otworzyć:", video_file)
        return []
    
    sequence = []
    frame_width, frame_height = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if frame_width is None:
            frame_height, frame_width = frame.shape[:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = []

            # Barki i nogi
            left_shoulder_y = landmarks[11].y * frame_height
            right_shoulder_y = landmarks[12].y * frame_height
            left_knee_y = landmarks[25].y * frame_height
            right_knee_y = landmarks[26].y * frame_height
            left_ankle_y = landmarks[27].y * frame_height
            right_ankle_y = landmarks[28].y * frame_height
            left_shoulder_x = landmarks[11].x * frame_width
            right_shoulder_x = landmarks[12].x * frame_width

            # Ostrzeżenia
            if (left_knee_y < left_shoulder_y or right_knee_y < right_shoulder_y or
                left_ankle_y < left_shoulder_y or right_ankle_y < right_shoulder_y):
                print(f"Ostrzeżenie: dziwne ułożenie nóg w wideo {video_id}, klatka {len(sequence)}")
            if right_shoulder_x <= left_shoulder_x:
                print(f"Ostrzeżenie: prawy bark nie jest po prawej stronie w wideo {video_id}, klatka {len(sequence)}")

            for idx in SELECTED_LANDMARKS:
                lm = landmarks[idx]
                cx = lm.x * frame_width
                cy = lm.y * frame_height
                frame_data.extend([cx, cy])
            sequence.append(frame_data)

        if show_preview:
            cv2.imshow("Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    pose.close()
    sequence = np.array(sequence)
    resampled = resample_sequence(sequence, target_len=target_len)

    rows = []
    # Normalne dane
    for i in range(target_len):
        row = {'video_id': video_id, 'frame': i, 'label': label}
        for j, idx in enumerate(SELECTED_LANDMARKS):
            row[f"{LANDMARKS_INFO[idx]}_x"] = resampled[i, j*2]
            row[f"{LANDMARKS_INFO[idx]}_y"] = resampled[i, j*2+1]
        rows.append(row)

    # Odbicie lustrzane
    resampled_mirror = resampled.copy()
    resampled_mirror[:, 0::2] = frame_width - resampled_mirror[:, 0::2]
    for i in range(target_len):
        row = {'video_id': video_id + 1, 'frame': i, 'label': label}
        for j, idx in enumerate(SELECTED_LANDMARKS):
            row[f"{LANDMARKS_INFO[idx]}_x"] = resampled_mirror[i, j*2]
            row[f"{LANDMARKS_INFO[idx]}_y"] = resampled_mirror[i, j*2+1]
        rows.append(row)

    return rows, 2  # zwracamy też ile ID zwiększyć (normal + mirror)

# === GROMADZENIE LISTY WIDEO ===
video_list = []
video_id = 0
for label_name in os.listdir(base_dir):
    label_path = os.path.join(base_dir, label_name)
    if not os.path.isdir(label_path):
        continue
    try:
        label = int(label_name.split(" ")[0])
    except:
        continue
    for video_file in glob(os.path.join(label_path, "*.mp4")):
        video_list.append((video_file, label, video_id))
        video_id += 2  # normal + mirror

# === PRZETWARZANIE WIELOPROCESOWE ===
all_rows = []
with Pool(processes=min(cpu_count(), len(video_list))) as pool:
    results = pool.map(process_video, video_list)
    for rows, _ in results:
        all_rows.extend(rows)

# === ZAPIS DO CSV ===
fieldnames = ['video_id', 'frame', 'label']
for idx in SELECTED_LANDMARKS:
    name = LANDMARKS_INFO[idx]
    fieldnames.append(f"{name}_x")
    fieldnames.append(f"{name}_y")

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Wszystkie dane zapisane do: {output_csv}")
