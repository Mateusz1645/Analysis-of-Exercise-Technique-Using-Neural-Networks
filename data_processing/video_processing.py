import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
from config import TARGET_LEN, RANDOM_ADD, MAX_SHIFT, SHOW_PREVIEW, LANDMARKS_INFO, SELECTED_LANDMARKS, MISSING_THRESHOLD, SELECTED_LANDMARKS_MIRRORED
import os
import absl.logging
import logging

absl.logging.set_verbosity(absl.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def resample_sequence(sequence: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """
    Resamples a sequence of landmark coordinates to a fixed number of frames.
    
    Args:
        sequence (np.ndarray): Original sequence of shape (num_frames, num_landmarks*2)
        target_len (int): Number of frames to resample to
    
    Returns:
        np.ndarray: Resampled sequence of shape (target_len, num_landmarks*2)
    """
    if sequence.shape[0] == 0:
        logging.warning("Empty sequence received for resampling, returning zeros.")
        return np.zeros((target_len, len(SELECTED_LANDMARKS) * 2))
    
    old_time = np.linspace(0, 1, sequence.shape[0])
    new_time = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, sequence.shape[1]))
    
    for i in range(sequence.shape[1]):
        interpolator = interp1d(old_time, sequence[:, i], kind='linear', fill_value="extrapolate")
        resampled[:, i] = interpolator(new_time)
    
    return resampled

def extract_landmarks(landmarks, frame_width: int, frame_height: int) -> list:
    """
    Extracts selected landmark coordinates from MediaPipe results.

    Args:
        landmarks: list of MediaPipe landmark objects for a frame
        frame_width: width of the frame in pixels
        frame_height: height of the frame in pixels

    Returns:
        list: x, y coordinates of selected landmarks
    """
    data = []
    for idx in SELECTED_LANDMARKS:
        lm = landmarks[idx]
        cx = lm.x
        cy = lm.y
        data.extend([cx, cy])
    return data

def check_initial_frame_problems(landmarks, frame_width: int, frame_height: int) -> tuple[int, int]:
    """
    Checks if the first frame has abnormal leg or shoulder positions.

    Args:
        landmarks: list of MediaPipe landmark objects for a frame
        frame_width: width of the frame in pixels
        frame_height: height of the frame in pixels

    Returns:
        tuple: (problem_legs_count, problem_shoulders_count)
    """
    problem_legs_count = 0
    problem_shoulders_count = 0

    required_landmarks = [11, 12, 25, 26, 27, 28]
    if all(landmarks[idx] is not None for idx in required_landmarks):
        left_shoulder_y = landmarks[11].y * frame_height
        right_shoulder_y = landmarks[12].y * frame_height
        left_knee_y = landmarks[25].y * frame_height
        right_knee_y = landmarks[26].y * frame_height
        left_ankle_y = landmarks[27].y * frame_height
        right_ankle_y = landmarks[28].y * frame_height
        left_shoulder_x = landmarks[11].x * frame_width
        right_shoulder_x = landmarks[12].x * frame_width

        # Legs must be below shoulders
        if ((left_knee_y < left_shoulder_y and right_knee_y < right_shoulder_y) or
            (left_ankle_y < left_shoulder_y and right_ankle_y < right_shoulder_y)):
            problem_legs_count += 1

        # Check shoulder positions
        if right_shoulder_x <= left_shoulder_x:
            problem_shoulders_count += 1

    return problem_legs_count, problem_shoulders_count

def augment_mirror(sequence: np.ndarray, frame_width: int) -> np.ndarray:
    """
    Mirror the sequence horizontally.

    Args:
        sequence: np.ndarray of shape (num_frames, num_landmarks*2)
        frame_width: width of the video frame

    Returns:
        np.ndarray: mirrored sequence
    """
    mirrored = sequence.copy()
    mirrored[:, 0::2] = 1 - mirrored[:, 0::2]  # mirror x-coordinates

    return mirrored

def augment_random_shift(sequence: np.ndarray, num_shifts: int = RANDOM_ADD) -> list[np.ndarray]:
    """
    Apply random x/y shifts to a sequence for data augmentation.

    Args:
        sequence: np.ndarray of shape (num_frames, num_landmarks*2)
        num_shifts: number of augmented variants to generate

    Returns:
        list of np.ndarray: each array is an augmented sequence
    """
    augmented_sequences = []
    for _ in range(num_shifts):
        dx = np.random.uniform(-MAX_SHIFT, MAX_SHIFT)
        dy = np.random.uniform(-MAX_SHIFT, MAX_SHIFT)
        augmented = sequence.copy()
        augmented[:, 0::2] += dx  # x-coordinates
        augmented[:, 1::2] += dy  # y-coordinates
        augmented_sequences.append(augmented)
    return augmented_sequences

def create_rows(sequence, video_id, label, is_mirrored=False):
    """
    Converts a numpy sequence of landmarks to list of dictionaries for CSV.
    """
    rows = []
    for i in range(sequence.shape[0]):
        row = {'video_id': video_id, 'frame': i, 'label': label}
        if is_mirrored:
            for j, idx in enumerate(SELECTED_LANDMARKS_MIRRORED):
                row[f"{LANDMARKS_INFO[idx]}_x"] = sequence[i, j*2]
                row[f"{LANDMARKS_INFO[idx]}_y"] = sequence[i, j*2+1]
            rows.append(row)
        else:
            for j, idx in enumerate(SELECTED_LANDMARKS):
                row[f"{LANDMARKS_INFO[idx]}_x"] = sequence[i, j*2]
                row[f"{LANDMARKS_INFO[idx]}_y"] = sequence[i, j*2+1]
            rows.append(row)
    return rows

def check_flip_180(landmarks) -> bool:
    """
    Checks if the person in the frame is upside down.
    Returns True if feet are above shoulders.
    """
    left_foot_y = landmarks[27].y
    right_foot_y = landmarks[28].y
    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y
    return left_foot_y < left_shoulder_y or right_foot_y < right_shoulder_y

def process_video(video_file: str, label: int, video_id: int):
    """
    Processes a single video to extract pose landmarks, resample them, 
    and apply augmentations (mirror and random shifts).
    
    Args:
        video_file (str): Path to the video file
        label (int): Original label of the video
        video_id (int): Starting ID for the video (used for augmented variants)
    
    Returns:
        list[dict]: List of dictionaries containing frame-wise landmark data
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Cannot open: {video_file}")
        return []

    sequence = []
    frame_width, frame_height = None, None
    rotate_code = None
    flip_180 = False
    first_landmarks_checked = False
    check_frames = 5
    missing_landmarks_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect orientation for vertical video
        if frame_width is None:
            h, w = frame.shape[:2]
            frame_height, frame_width = h, w
            if h < w:
                rotate_code = cv2.ROTATE_90_CLOCKWISE
                frame_width, frame_height = h, w  # swap dimensions after rotation

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # Convert frame to RGB and process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Check if the video is upside down (feet at top)
            if not first_landmarks_checked:
                flip_180 = check_flip_180(landmarks)
                first_landmarks_checked = True

            if flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_data = []

            first_frame_problems = []
            if len(sequence) < check_frames:
                legs, shoulders = check_initial_frame_problems(landmarks, frame_width, frame_height)
                first_frame_problems.append((legs, shoulders))

            # Log warnings if all first frames are abnormal
            if len(sequence) == check_frames:
                legs_total = sum(f[0] for f in first_frame_problems)
                shoulders_total = sum(f[1] for f in first_frame_problems)
                if legs_total == check_frames:
                    logging.warning(f"Warning: all first {check_frames} frames have abnormal leg position in video {video_id} -> {video_file}")
                if shoulders_total == check_frames:
                    logging.warning(f"Warning: all first {check_frames} frames have right shoulder not on the right in video {video_id} -> {video_file}")


            # Extract selected landmark coordinates
            frame_data = extract_landmarks(landmarks, frame_width, frame_height)
            sequence.append(frame_data)
        else:
            missing_landmarks_count += 1
            if missing_landmarks_count >= MISSING_THRESHOLD:
                logging.warning(f"No pose landmarks detected for {missing_landmarks_count} consecutive frames in video {video_id} -> {video_file}")

        # Optional preview
        if SHOW_PREVIEW:
            cv2.imshow("Preview", frame)
            if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    pose.close()

    # Resample to fixed length
    sequence = np.array(sequence)
    resampled = resample_sequence(sequence, target_len=TARGET_LEN)

    rows = []

    # Original data
    rows.extend(create_rows(resampled, video_id, label))

    # Mirror augmentation
    mirrored_seq = augment_mirror(resampled, frame_width)
    mirrored_label = label
    if label == 5:
        mirrored_label = 6
    elif label == 6:
        mirrored_label = 5
    rows.extend(create_rows(mirrored_seq, video_id + 1, mirrored_label, is_mirrored=True))

    # Random shift augmentations
    augmented_seqs = augment_random_shift(resampled)
    for k, aug_seq in enumerate(augmented_seqs):
        rows.extend(create_rows(aug_seq, video_id + 2 + k, label))

    return rows

def process_single_video(args):
    """
    Wrapper function for parallel processing.
    Args is a tuple: (video_file, label, video_id)
    """
    video_file, label, video_id = args
    return process_video(video_file, label, video_id)