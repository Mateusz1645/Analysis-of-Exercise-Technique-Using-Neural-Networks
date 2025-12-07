import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# TRAINING HISTORY PLOTS
def plot_history(history):
    plt.figure(figsize=(12,5))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# CONFUSION MATRIX & REPORT
def evaluate_model(model, X_test, y_test, model_type, y_labels=['Prawidłowe', 'Brak wyprostu na początku', 'Niepełna górna faza', 'Brak wyprostu na koncu', 'Zgięte kolana', 'Bardziej do lewej', 'Bardziej do prawej']):
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")

    ax.set_xlabel("Przewidywana klasa", fontsize=12)
    ax.set_ylabel("Prawdziwa klasa", fontsize=12)
    ax.set_title(f"Macierz pomyłek - {model_type.upper()}", fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.subplots_adjust(bottom=0.25, left=0.2, top=0.9, right=0.95)
    plt.show()

    print(classification_report(y_test, y_pred, target_names=y_labels))
