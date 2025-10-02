import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import INPUT_CSV, TEST_SIZE, RANDOM_STATE, VAL_SIZE

# === LOAD DATAFRAME FROM CSV ===
def load_data(input=INPUT_CSV):
    return pd.read_csv(input)

# === PREPROCESS DATA FOR LSTM ===
def preprocess_data(df, target_candidates=("target","label","class"), sequence_length=90, max_x=1080, max_y=1920):
    """
    Preprocess data for LSTM by grouping frames per video.
    Scaling is done relative to maximum scene size (max_x, max_y) instead of min/max from data.
    
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

    return X, y

# === TRAIN/TEST SPLIT ===
def split_data(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=True):
    strat = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    if val_size:
        strat_val = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=strat_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_test, y_train, y_test