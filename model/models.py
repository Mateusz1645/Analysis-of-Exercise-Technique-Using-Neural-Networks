from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Conv1D, MaxPooling1D, Flatten


def build_model(input_shape, num_classes, model_type='cnn_lstm'):
    """
    model_type: 'lstm', 'rnn', 'cnn', 'cnn_lstm'
    """
    model = Sequential()
    
    if model_type == 'lstm':
        # LSTM – najlepsze parametry
        model.add(LSTM(128, return_sequences=False, dropout=0.1, input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        
    elif model_type == 'rnn':
        # RNN – najlepsze parametry
        model.add(SimpleRNN(64, return_sequences=False, dropout=0.1, input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
    elif model_type == 'cnn':
        # CNN – najlepsze parametry
        model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        
    elif model_type == 'cnn_lstm':
        # CNN+LSTM – najlepsze parametry
        model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128, return_sequences=False, dropout=0.1))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
    
    # Warstwa wyjściowa wspólna dla wszystkich modeli
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

