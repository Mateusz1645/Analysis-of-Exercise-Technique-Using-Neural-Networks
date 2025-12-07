import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import pandas as pd
from keras_tuner import GridSearch
from utils import load_data, preprocess_data, split_data
from config import EPOCHS, BATCH_SIZE

# Load and preprocess data
df = load_data()
X, y = preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, is_val=True)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))

# LSTM model tuner
def build_lstm_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    model.add(LSTM(
        units=hp.Choice('lstm_units', [64, 128, 256]),
        return_sequences=False,
        dropout=hp.Choice('lstm_dropout', [0.1])
    ))
    
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dense_dropout', [0.1])))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN model tuner
def build_cnn_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    model.add(Conv1D(
        filters=hp.Choice('conv_filters', [32, 64, 128, 256]),
        kernel_size=hp.Choice('kernel_size', [3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dropout_rate', [0.1])))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN+LSTM model tuner
def build_cnn_lstm_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    model.add(Conv1D(
        filters=hp.Choice('conv_filters', [32, 64, 128, 256]),
        kernel_size=hp.Choice('kernel_size', [3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(LSTM(
        units=hp.Choice('lstm_units', [64, 128, 256]),
        return_sequences=False,
        dropout=hp.Choice('lstm_dropout', [0.1])
    ))
    
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dropout_rate', [0.1])))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# RNN model tuner
def build_rnn_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    model.add(SimpleRNN(
        units=hp.Choice('rnn_units', [64, 128, 256]),
        return_sequences=False,
        dropout=hp.Choice('rnn_dropout', [0.1])
    ))
    
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dense_dropout', [0.1])))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of models to test
models_to_test = [
    # ('LSTM', build_lstm_model),
    # ('RNN', build_rnn_model),
    # ('CNN', build_cnn_model),
    ('CNN+LSTM', build_cnn_lstm_model)
]

# Run tuner for each model
for model_name, build_fn in models_to_test:
    print(f"=== Running tuner for {model_name} ===")

    all_trials_df = [] 

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    tuner = GridSearch(
        build_fn,
        objective='val_accuracy',
        max_trials=None,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name=f'{model_name}_grid'
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    best_models = tuner.get_best_models(num_models=len(tuner.oracle.trials))
    best_trials = tuner.oracle.get_best_trials(num_trials=len(best_models))

    for i, model in enumerate(best_models):

        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=1)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        report_df = pd.DataFrame(report_dict).T

        hp_df = pd.DataFrame([best_trials[i].hyperparameters.values])

        report_df['model'] = model_name
        report_df['trial'] = i + 1

        for col, val in hp_df.iloc[0].items():
            report_df[col] = val

        all_trials_df.append(report_df)

    final_df = pd.concat(all_trials_df)

    final_df.to_csv(f"{model_name}_tuning_report_df.csv", index=True)

    print(f"Raport zapisany: {model_name}_tuning_report_df.csv")
