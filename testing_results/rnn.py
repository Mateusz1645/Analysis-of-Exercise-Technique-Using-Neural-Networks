import numpy as np
import pandas as pd
from itertools import product
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam
from utils import load_data, preprocess_data, split_data
from sklearn.utils.class_weight import compute_class_weight
from config import EPOCHS, BATCH_SIZE

df = load_data()
X, y = preprocess_data(df)
input_shape = (X.shape[1], X.shape[2])
num_classes = len(np.unique(y))

rnn_units_list = [64, 128, 256]
dense_units_list = [64, 128, 256]
dropout_list = [0.1]
K = 5 

def build_model(rnn_units, dense_units, dropout):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(SimpleRNN(rnn_units, dropout=dropout))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

param_combinations = list(product(rnn_units_list, dense_units_list, dropout_list))
results = []

for rnn_units, dense_units, dropout in param_combinations:
    print(f"\n=== TESTUJE: RNN={rnn_units}, Dense={dense_units}, Dropout={dropout} ===")
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    
    fold_reports = []
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        model = build_model(rnn_units, dense_units, dropout)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=[early_stopping],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        fold_reports.append(report)
    
    accuracy = np.mean([r["accuracy"] for r in fold_reports])
    macro_f1 = np.mean([r["macro avg"]["f1-score"] for r in fold_reports])
    macro_precision = np.mean([r["macro avg"]["precision"] for r in fold_reports])
    macro_recall = np.mean([r["macro avg"]["recall"] for r in fold_reports])
    weighted_f1 = np.mean([r["weighted avg"]["f1-score"] for r in fold_reports])
    
    results.append({
        "rnn_units": rnn_units,
        "dense_units": dense_units,
        "dropout": dropout,
        "accuracy_mean": accuracy,
        "macro_f1_mean": macro_f1,
        "macro_precision_mean": macro_precision,
        "macro_recall_mean": macro_recall,
        "weighted_f1_mean": weighted_f1
    })

df_results = pd.DataFrame(results).sort_values(by="macro_f1_mean", ascending=False)
df_results.to_csv("rnn_results.csv", index=False)
print(df_results)
