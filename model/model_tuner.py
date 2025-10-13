import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import GridSearch
from utils import load_data, preprocess_data, split_data
from config import EPOCHS, BATCH_SIZE

# === Load and preprocess ===
df = load_data()
X, y = preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, is_val=True)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))

# === Build model function for Keras Tuner ===
def build_model_tuner(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    # LSTM layer
    model.add(LSTM(
        units=hp.Choice('lstm_units', [64, 128, 256]),
        return_sequences=False,
        dropout=hp.Choice('lstm_dropout', [0.1])
    ))
    
    # Dense layer
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dense_dropout', [0.1])))

    # Output
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Setup Random Search Tuner ===
tuner = GridSearch(
    build_model_tuner,
    objective='val_accuracy',
    max_trials=None,  # wszystkie kombinacje
    executions_per_trial=3,
    directory='tuner_dir',
    project_name='lstm_video_grid'
)

# === Run the search ===
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# === Get the best model and hyperparameters ===
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print best hyperparameters
print("Best hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# === Evaluate on test set ===
loss, acc = best_model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")