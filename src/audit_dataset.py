
from pathlib import Path

import csv
import numpy as np
import tensorflow as tf

from data_loader import load_datasets

MODEL_PATH = Path("models/vehicle_damage_mobilenetv2.keras")
REPORT_PATH = Path("reports/dataset_audit.csv")

CLASS_NAMES = ["Minor", "Moderate", "Severe"]


def audit_dataset():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _, validation_dataset = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    rows = []

    image_paths = validation_dataset.file_paths
    image_index = 0

    for images, labels in validation_dataset:
        probabilities = model.predict(images, verbose=0)

        predicted_labels = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        for actual, predicted, confidence in zip(
            labels.numpy(),
            predicted_labels,
            confidences
        ):
            image_path = image_paths[image_index]
            image_index += 1

            rows.append({
                "image_path": image_path,
                "actual_class": CLASS_NAMES[actual],
                "predicted_class": CLASS_NAMES[predicted],
                "confidence": round(float(confidence), 4),
                "correct": actual == predicted
            })

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "actual_class",
                "predicted_class",
                "confidence",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    incorrect = sum(not row["correct"] for row in rows)

    print("\nDataset audit completed!")
    print("Total images:", len(rows))
    print("Correct predictions:", len(rows) - incorrect)
    print("Incorrect predictions:", incorrect)
    print("Report saved at:", REPORT_PATH)


if __name__ == "__main__":
    audit_dataset()
