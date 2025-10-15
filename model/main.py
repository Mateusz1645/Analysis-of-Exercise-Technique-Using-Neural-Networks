import numpy as np
from config import EPOCHS, BATCH_SIZE
from models import build_model
from metrics import plot_history, evaluate_model
from utils import load_data, preprocess_data, split_data
from tensorflow.keras.callbacks import EarlyStopping

# === PIPELINE ENTRY POINT ===
def main():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y))
    model = build_model(input_shape, num_classes)

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',    # monitorujemy stratę na zbiorze walidacyjnym
        patience=10,            # liczba epok bez poprawy przed zatrzymaniem
        restore_best_weights=True  # przywracamy wagi z najlepszej epoki
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping]  # dodajemy callback
    )

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    # Plots and reports
    plot_history(history)
    evaluate_model(model, X_test, y_test, y_labels=np.unique(y))

if __name__ == "__main__":
    main()
