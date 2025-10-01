from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout

# === MODEL DEFINITION ===
def build_model(input_shape, num_classes):
    model = Sequential([
        # Mask missing values with 0.0
        Masking(mask_value=0.0, input_shape=input_shape),
        
        # LSTM layer
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        # Fully connected layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # === MODEL COMPILATION ===
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
