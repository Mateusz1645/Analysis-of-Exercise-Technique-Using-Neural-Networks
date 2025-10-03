import numpy as np
from utils import load_data, preprocess_data

def check_preprocessed_data(sequence_length=90):
    # Wczytanie danych
    df = load_data()
    print("=== Informacje ogólne o DataFrame ===")
    print(df.info())

    # Preprocessing
    X, y = preprocess_data(df, sequence_length=sequence_length)

    print("\n=== Kształt po preprocessingu ===")
    print(f"X: {X.shape} (num_samples, timesteps, num_features)")
    print(f"y: {y.shape} (num_samples,)")

    # Sprawdzenie unikalnych klas i liczności
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== Liczba próbek na klasę po preprocessingu ===")
    for u, c in zip(unique, counts):
        print(f"Klasa {u}: {c} próbek")

    # Sprawdzenie zakresu wartości w X (MinMaxScaler)
    print("\n=== Zakres wartości w X po skalowaniu ===")
    print(f"Min: {X.min():.4f}, Max: {X.max():.4f}")

if __name__ == "__main__":
    check_preprocessed_data(sequence_length=90)
