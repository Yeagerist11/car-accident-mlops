
from pathlib import Path
import tensorflow as tf
import mlflow

from src.data_loader import load_datasets
from src.model import build_model

# Training settings
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
MODEL_PATH = Path("models/vehicle_damage_augmented.keras")

# Use the same MLflow database as your local experiments
mlflow.set_tracking_uri(
    "sqlite:///D:/car-accident-mlops/mlflow.db"
)
mlflow.set_experiment("Vehicle_Damage_Classification")


def train_model():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset = load_datasets()

    with mlflow.start_run(run_name="Jenkins_MobileNetV2_Training"):

        # Record training configuration
        mlflow.log_params({
            "model": "MobileNetV2",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "image_size": "224x224",
            "batch_size": 32,
            "optimizer": "Adam",
            "data_augmentation": True
        })

        model = build_model()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        print("\nStarting model training with MLflow tracking...\n")

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=[early_stopping]
        )

        # Record best training metrics
        best_epoch = min(
            range(len(history.history["val_loss"])),
            key=lambda i: history.history["val_loss"][i]
        )

        mlflow.log_metrics({
            "best_train_accuracy": float(
                history.history["accuracy"][best_epoch]
            ),
            "best_validation_accuracy": float(
                history.history["val_accuracy"][best_epoch]
            ),
            "best_validation_loss": float(
                history.history["val_loss"][best_epoch]
            ),
            "epochs_completed": len(history.history["loss"])
        })

        # Save trained model
        model.save(MODEL_PATH)

        # Record model as an MLflow artifact
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model"
        )

        print("\nTraining completed!")
        print(f"Model saved at: {MODEL_PATH}")
        print("Metrics and model artifact logged to MLflow.")

        return history


if __name__ == "__main__":
    train_model()
