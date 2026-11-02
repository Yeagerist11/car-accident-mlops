
from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow

from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_datasets

# Model paths
INPUT_MODEL = Path("models/vehicle_damage_augmented.keras")
OUTPUT_MODEL = Path("models/vehicle_damage_finetuned_mlflow.keras")

# MLflow experiment
mlflow.set_experiment("Vehicle_Damage_Classification")

CLASS_NAMES = ["Minor", "Moderate", "Severe"]


def train_with_mlflow():

    train_dataset, validation_dataset = load_datasets()

    with mlflow.start_run(run_name="FineTuning_Automatic_Tracking"):

        # Record training parameters
        mlflow.log_params({
            "model": "MobileNetV2",
            "input_model": str(INPUT_MODEL),
            "epochs": 5,
            "learning_rate": 0.00001,
            "unfrozen_layers": 30,
            "data_augmentation": True
        })

        # Load the augmented model
        model = tf.keras.models.load_model(INPUT_MODEL)

        base_model = model.get_layer("mobilenetv2_1.00_224")
        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.00001
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )

        print("\nTraining with automatic MLflow tracking...\n")

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=5,
            callbacks=[early_stopping]
        )

        # Record best training metrics
        mlflow.log_metrics({
            "best_val_accuracy": float(
                max(history.history["val_accuracy"])
            ),
            "best_val_loss": float(
                min(history.history["val_loss"])
            )
        })

        # Evaluate the trained model
        actual_labels = []
        predicted_labels = []

        for images, labels in validation_dataset:
            predictions = model.predict(images, verbose=0)

            predicted_labels.extend(
                np.argmax(predictions, axis=1)
            )
            actual_labels.extend(labels.numpy())

        accuracy = accuracy_score(
            actual_labels, predicted_labels
        )

        report = classification_report(
            actual_labels,
            predicted_labels,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        # Record evaluation metrics
        mlflow.log_metric("evaluation_accuracy", float(accuracy))

        for class_name in CLASS_NAMES:
            mlflow.log_metric(
                f"{class_name.lower()}_precision",
                report[class_name]["precision"]
            )
            mlflow.log_metric(
                f"{class_name.lower()}_recall",
                report[class_name]["recall"]
            )
            mlflow.log_metric(
                f"{class_name.lower()}_f1",
                report[class_name]["f1-score"]
            )

        mlflow.log_metric(
            "macro_f1",
            report["macro avg"]["f1-score"]
        )

        # Save model separately
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        model.save(OUTPUT_MODEL)

        # Attach model file to this MLflow run
        mlflow.log_artifact(
            str(OUTPUT_MODEL),
            artifact_path="model"
        )

        print("\nClassification Report:")
        print(classification_report(
            actual_labels,
            predicted_labels,
            target_names=CLASS_NAMES,
            zero_division=0
        ))

        print("\nAutomatic MLflow tracking completed!")
        print("Model saved at:", OUTPUT_MODEL)


if __name__ == "__main__":
    train_with_mlflow()
