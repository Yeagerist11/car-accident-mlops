
from pathlib import Path
import tensorflow as tf

# Dataset location
import os

DATASET_DIR = Path(
    os.getenv("DATASET_DIR", "data/processed")
)

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Class labels
CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]


def load_datasets():
    training_path = DATASET_DIR / "training"
    validation_path = DATASET_DIR / "validation"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        training_path,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_path,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    return train_dataset, validation_dataset


if __name__ == "__main__":
    train, validation = load_datasets()

    print("Class names:", train.class_names)
    print("Training batches:", len(train))
    print("Validation batches:", len(validation))
