
from pathlib import Path
import tensorflow as tf

from src.data_loader import load_datasets
from src.model import build_model

# Training settings
EPOCHS = 10
MODEL_DIR = Path("models")
MODEL_PATH = Path("models/vehicle_damage_augmented.keras")
def train_model():
    # Create models folder if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load training and validation images
    train_dataset, validation_dataset = load_datasets()

    # Build MobileNetV2 model
    model = build_model()

    # Stop training if validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nStarting model training...\n")

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # Save trained model
    model.save(MODEL_PATH)

    print("\nTraining completed!")
    print(f"Model saved at: {MODEL_PATH}")

    return history


if __name__ == "__main__":
    train_model()
