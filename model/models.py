from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout

# === MODEL DEFINITION WITH DROPOUT ===
def build_model(input_shape, num_classes, lstm_units=64, dense_units=256, dropout_rate=0.1):
    model = Sequential([
        # Mask missing values with 0.0
        # Masking(mask_value=0.0, input_shape=input_shape),
        
        # LSTM layer with dropout
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),  # Dropout after LSTM
        
        # Fully connected layers
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),  # Dropout after Dense
        Dense(num_classes, activation='softmax')
    ])

    # === MODEL COMPILATION ===
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

