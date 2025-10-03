import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import INPUT_CSV, TEST_SIZE, RANDOM_STATE

# === LOAD DATAFRAME FROM CSV ===
def load_data():
    return pd.read_csv(INPUT_CSV)

# === PREPROCESS DATA FOR LSTM ===
def preprocess_data(df, target_candidates=("target","label","class"), sequence_length=90):
    """
    Preprocess data for LSTM by grouping frames per video.
    
    Returns:
        X: np.array of shape (num_samples, timesteps, num_features)
        y: np.array of shape (num_samples,)
    """
    # === FIND TARGET COLUMN ===
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise KeyError(f"No target column found. Available columns: {list(df.columns)}")
    
    # === GROUP BY VIDEO ID ===
    X_list, y_list = [], []
    video_ids = df['video_id'].unique()
    
    for vid in video_ids:
        video_df = df[df['video_id'] == vid].sort_values('frame')
        features = video_df.drop(columns=['video_id', 'frame', target_col]).values
        
        if features.shape[0] == sequence_length:  # only full sequences
            X_list.append(features)
            y_list.append(video_df[target_col].iloc[0])
    
    X = np.array(X_list)  # (num_samples, timesteps, num_features)
    y = np.array(y_list)
    
    # === SCALE FEATURES ===
    num_samples, timesteps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)  # flatten all frames for scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, timesteps, num_features)  # back to 3D
    
    return X_scaled, y

# === TRAIN/TEST SPLIT ===
def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
