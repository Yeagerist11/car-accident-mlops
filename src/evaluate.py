
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from data_loader import load_datasets

MODEL_PATH = Path("models/vehicle_damage_finetuned.keras")
REPORT_DIR = Path("reports")

CLASS_NAMES = ["Minor", "Moderate", "Severe"]


def evaluate_model():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation images
    _, validation_dataset = load_datasets()

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    actual_labels = []
    predicted_labels = []

    print("\nEvaluating model...\n")

    # Predict each validation batch
    for images, labels in validation_dataset:
        predictions = model.predict(images, verbose=0)

        predicted_labels.extend(
            np.argmax(predictions, axis=1)
        )

        actual_labels.extend(labels.numpy())

    # Classification report
    print("Classification Report:")
    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    # Save classification report
    with open(REPORT_DIR / "classification_report.txt", "w") as file:
        file.write(report)

    # Confusion matrix
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Vehicle Damage Classification")
    plt.tight_layout()

    plt.savefig(REPORT_DIR / "confusion_matrix.png")
    plt.show()

    print("\nConfusion matrix saved in reports/")
    print("Classification report saved in reports/")


if __name__ == "__main__":
    evaluate_model()
