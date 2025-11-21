import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_tuner import GridSearch
from utils import load_data, preprocess_data, split_data
from config import EPOCHS, BATCH_SIZE

# Load and preprocess data
df = load_data()
X, y = preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, is_val=True)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))

# Function to save results to a file
def save_results(filename, model_name, best_hp, test_loss, test_acc):
    with open(filename, 'a') as f:
        f.write(f"=== Model: {model_name} ===\n")
        f.write("Best hyperparameters:\n")
        for param, value in best_hp.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n\n")

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
    ('LSTM', build_lstm_model),
    ('RNN', build_rnn_model),
    ('CNN', build_cnn_model),
    ('CNN+LSTM', build_cnn_lstm_model)
]

# Clear previous results file
with open('tuner_results.txt', 'w') as f:
    f.write("Hyperparameter Tuning Results\n\n")

# Run tuner for each model
for model_name, build_fn in models_to_test:
    print(f"=== Running tuner for {model_name} ===")
    
    # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',       
        patience=10,              
        restore_best_weights=True 
    )
    
    tuner = GridSearch(
        build_fn,
        objective='val_accuracy',
        max_trials=None,
        executions_per_trial=3,
        directory='tuner_dir',
        project_name=f'{model_name}_grid'
    )
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,          
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stop]   
    )
    
    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0].values
    loss, acc = best_model.evaluate(X_test, y_test)
    
    save_results('tuner_results.txt', model_name, best_hp, loss, acc)

print("All models tested. Result are in tuner_results.txt")
