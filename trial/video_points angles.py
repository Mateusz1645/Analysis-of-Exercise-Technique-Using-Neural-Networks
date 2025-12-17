import cv2
import mediapipe as mp
import os
import math

# --- Ustawienia Mediapipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Landmarki ---
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

# --- Funkcja do rysowania łuku kąta ---
def draw_angle_arc(frame, vertex, point1, point2, radius=40, color=(0,0,255), label='α'):
    """
    Rysuje łuk przy wierzchołku vertex między point1 a point2.
    vertex, point1, point2: tuple(x, y)
    """
    angle1 = math.atan2(point1[1]-vertex[1], point1[0]-vertex[0])
    angle2 = math.atan2(point2[1]-vertex[1], point2[0]-vertex[0])
    
    start_deg = int(math.degrees(angle1))
    end_deg = int(math.degrees(angle2))
    
    if start_deg > end_deg:
        start_deg, end_deg = end_deg, start_deg
    
    cv2.ellipse(frame, (vertex[0], vertex[1]), (radius, radius), 0, start_deg, end_deg, color, 2)
    v1 = (point1[0]-vertex[0], point1[1]-vertex[1])
    v2 = (point2[0]-vertex[0], point2[1]-vertex[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle_deg = int(math.degrees(math.atan2(abs(det), dot)))

    # Rysowanie wartości kąta jako label
    cv2.putText(frame, f"{angle_deg} st.", (vertex[0]+radius//2, vertex[1]-radius//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# --- Ścieżki plików ---
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, "20250628_155721.mp4")
output_video_path = os.path.join(script_dir, "output_katy.mp4")

# --- Wczytanie wideo ---
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Nie udało się otworzyć pliku wideo!")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Przetwarzanie klatek ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.flip(frame, 0)  # jeśli potrzebujesz obrócić w pionie
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- Rysowanie punktów ---
        for idx in SELECTED_LANDMARKS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # --- Rysowanie połączeń ---
        for start_idx, end_idx in CUSTOM_CONNECTIONS:
            lm_start = landmarks[start_idx]
            lm_end = landmarks[end_idx]

            x1, y1 = int(lm_start.x * width), int(lm_start.y * height)
            x2, y2 = int(lm_end.x * width), int(lm_end.y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- Rysowanie kątów ---
        # Lewa ręka
        left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
        left_elbow = (int(landmarks[13].x * width), int(landmarks[13].y * height))
        left_wrist = (int(landmarks[15].x * width), int(landmarks[15].y * height))
        left_hip = (int(landmarks[23].x * width), int(landmarks[23].y * height))

        # Kąt w łokciu: między barkiem a nadgarstkiem
        draw_angle_arc(frame, left_elbow, left_shoulder, left_wrist, radius=40, color=(0,0,255), label='α')

        # Kąt w barku: między łokciem a biodrem
        draw_angle_arc(frame, left_shoulder, left_elbow, left_hip, radius=50, color=(0,255,0), label='β')


        # Prawa ręka
        right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
        right_elbow = (int(landmarks[14].x * width), int(landmarks[14].y * height))
        right_wrist = (int(landmarks[16].x * width), int(landmarks[16].y * height))
        right_hip = (int(landmarks[24].x * width), int(landmarks[24].y * height))

        # Kąt w łokciu
        draw_angle_arc(frame, right_elbow, right_shoulder, right_wrist, radius=40, color=(0,0,255), label='')

        # Kąt w barku: między łokciem a biodrem
        draw_angle_arc(frame, right_shoulder, right_elbow, right_hip, radius=50, color=(0,255,0), label='Kat Łokcia')

    # --- Zapis klatki ---
    out.write(frame)

# --- Zwolnienie zasobów ---
cap.release()
out.release()
pose.close()

print(f"Wideo zapisane pod: {output_video_path}")
