import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import INPUT_CSV, TEST_SIZE, RANDOM_STATE, VAL_SIZE

# === LOAD DATAFRAME FROM CSV ===
def load_data():
    return pd.read_csv(INPUT_CSV)

# === PREPROCESS DATA FOR LSTM ===
def preprocess_data(df, target_candidates=("target","label","class"), sequence_length=60):
    """
    Preprocess data for LSTM using only joint angles.
    
    Args:
        df: DataFrame containing video frames and angles
        target_candidates: possible target column names
        sequence_length: number of frames per LSTM sequence
    
    Returns:
        X: np.array of shape (num_samples, timesteps, num_features)
        y: np.array of shape (num_samples,)
    """
    # === Find the target column ===
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise KeyError(f"No target column found. Available columns: {list(df.columns)}")
    
    # === Select only columns with angles ===
    angle_cols = [col for col in df.columns if col.endswith('_angle')]
    features_df = df[angle_cols].copy()

    # === Group frames by video ===
    X_list, y_list = [], []
    video_ids = df['video_id'].unique()
    
    for vid in video_ids:
        mask = df['video_id'] == vid
        video_df = df[mask].sort_values('frame')
        features = features_df[mask].loc[video_df.index].values  # maintain frame order

        if features.shape[0] == sequence_length:  # only full sequences
            X_list.append(features)
            y_list.append(video_df[target_col].iloc[0])
    
    X = np.array(X_list)  # (num_samples, timesteps, num_features)
    y = np.array(y_list)
    
    return X, y

# === TRAIN/TEST SPLIT ===
def split_data(X, y, is_val=False, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=True):
    strat = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    if val_size and is_val:
        strat_val = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=strat_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_test, y_train, y_test
