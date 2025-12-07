import numpy as np
import pandas as pd
from itertools import product
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Dropout, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.optimizers import Adam
from utils import load_data, preprocess_data, split_data
from config import EPOCHS, BATCH_SIZE
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

df = load_data()
X, y = preprocess_data(df)
input_shape = (X.shape[1], X.shape[2])
num_classes = len(np.unique(y))

cnn_units_list = [32, 64, 128]
kernel_size_list = [5]
lstm_units_list = [128, 256]
dense_units_list = [64, 128, 256]
dropout_list = [0.1]
K = 5

def build_model(cnn_units, kernel_size, lstm_units, dense_units, dropout):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Conv1D(filters=cnn_units, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=lstm_units, dropout=dropout, recurrent_dropout=0.1))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_combination(params):
    cnn_units, kernel_size, lstm_units, dense_units, dropout = params
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    fold_reports = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model = build_model(cnn_units, kernel_size, lstm_units, dense_units, dropout)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=[early_stopping],
            verbose=0
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        fold_reports.append(report)

    return {
        "cnn_units": cnn_units,
        "kernel_size": kernel_size,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "dropout": dropout,
        "accuracy_mean": np.mean([r["accuracy"] for r in fold_reports]),
        "macro_f1_mean": np.mean([r["macro avg"]["f1-score"] for r in fold_reports]),
        "macro_precision_mean": np.mean([r["macro avg"]["precision"] for r in fold_reports]),
        "macro_recall_mean": np.mean([r["macro avg"]["recall"] for r in fold_reports]),
        "weighted_f1_mean": np.mean([r["weighted avg"]["f1-score"] for r in fold_reports])
    }

if __name__ == "__main__":
    param_combinations = list(product(cnn_units_list, kernel_size_list, lstm_units_list, dense_units_list, dropout_list))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_combination, param_combinations))

    df_results = pd.DataFrame(results).sort_values(by="macro_f1_mean", ascending=False)
    df_results.to_csv("cnn_lstm_results.csv", index=False)
    print(df_results)
