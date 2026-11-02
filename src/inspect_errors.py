
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf

from data_loader import load_datasets

MODEL_PATH = Path("models/vehicle_damage_mobilenetv2.keras")
OUTPUT_DIR = Path("reports/moderate_review")

CLASS_NAMES = ["Minor", "Moderate", "Severe"]


def inspect_errors():
    _, validation_dataset = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    file_paths = validation_dataset.file_paths
    image_index = 0

    for images, labels in validation_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        for actual, predicted in zip(labels.numpy(), predicted_labels):
            image_path = Path(file_paths[image_index])
            image_index += 1

            # Inspect only actual Moderate images
            if actual != 1:
                continue

            if predicted == 1:
                folder = OUTPUT_DIR / "correct"
            else:
                predicted_class = CLASS_NAMES[predicted]
                folder = OUTPUT_DIR / f"misclassified_as_{predicted_class.lower()}"

            folder.mkdir(parents=True, exist_ok=True)

            destination = folder / image_path.name
            shutil.copy2(image_path, destination)

    print("\nModerate image inspection completed!")
    print(f"Images saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    inspect_errors()
