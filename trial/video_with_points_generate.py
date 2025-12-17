import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,        
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
# SELECTED_LANDMARKS = [0, 11, 13, 15, 23, 25, 27]


# CUSTOM_CONNECTIONS = [
#     (11, 13), (13, 15),  # ramię
#     (11, 23),           # bark → biodro
#     (23, 25), (25, 27)  # noga (opcjonalnie)
# ]



CUSTOM_CONNECTIONS = [
    (0, 11), (0, 12),         # głowa do barków
    (11, 12),                 # barki
    (23, 24),                 # biodra
    (11, 13), (13, 15),       # lewa ręka
    (12, 14), (14, 16),       # prawa ręka
    (11, 23), (12, 24),       # barki do bioder
    (23, 25), (25, 27),       # lewa noga
    (24, 26), (26, 28)        # prawa noga
]

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, "20250719_194118.mp4")
print(input_video_path)
output_video_path = os.path.join(script_dir, "output_3.mp4")

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Nie udało się otworzyć pliku wideo!")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 0)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        for idx in SELECTED_LANDMARKS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        for start_idx, end_idx in CUSTOM_CONNECTIONS:
            lm_start = landmarks[start_idx]
            lm_end = landmarks[end_idx]

            x1, y1 = int(lm_start.x * width), int(lm_start.y * height)
            x2, y2 = int(lm_end.x * width), int(lm_end.y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
pose.close()

print(f"Wideo zapisane pod: {output_video_path}")
