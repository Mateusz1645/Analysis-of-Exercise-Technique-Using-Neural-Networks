import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm



# === Parametry wideo ===
frame_width = 1080
frame_height = 1920
fps = 30

# === Lista punktów i połączeń ===
LANDMARKS_NAMES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}

# Odwrócona mapa ID → nazwa
ID_TO_NAME = {v: k for k, v in LANDMARKS_NAMES.items()}

# Połączenia (linia między parami punktów)
CUSTOM_CONNECTIONS = [
    (11, 12),
    (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

# === Wczytaj dane z CSV ===
df = pd.read_csv(csv_path)

# === Przygotuj VideoWriter ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# === Lista unikalnych klatek do przetworzenia ===
frames = sorted(df['frame'].unique())

# === Przetwarzanie klatek ===
for frame_num in tqdm(frames, desc="Generowanie wideo"):
    frame_data = df[df['frame'] == frame_num].iloc[0]

    # Puste czarne tło
    img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Wyodrębnij punkty
    points = {}
    for name, idx in LANDMARKS_NAMES.items():
        x_col = f'{name}_x'
        y_col = f'{name}_y'

        if pd.isna(frame_data[x_col]) or pd.isna(frame_data[y_col]):
            continue

        x = int(frame_data[x_col])
        y = int(frame_data[y_col])

        # Bezpieczne rysowanie tylko jeśli w granicach
        if 0 <= x < frame_width and 0 <= y < frame_height:
            points[idx] = (x, y)
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)

    # Rysowanie połączeń
    for start_idx, end_idx in CUSTOM_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(img, points[start_idx], points[end_idx], (255, 0, 0), 3)

    
    cv2.putText(img, f"Frame: {frame_num}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    img = cv2.flip(img, 0)
    out.write(img)

# === Zakończenie ===
out.release()
print(f"✅ Wideo z punktami zapisane pod: {output_video_path}")
