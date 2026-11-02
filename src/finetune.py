
from pathlib import Path
import tensorflow as tf
from data_loader import load_datasets

MODEL_PATH = Path("models/vehicle_damage_augmented.keras")
OUTPUT_PATH = Path("models/vehicle_damage_finetuned.keras")

def finetune_model():

    # Load datasets
    train_dataset, validation_dataset = load_datasets()

    # Load the already trained augmented model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Find MobileNetV2 inside our model
    base_model = model.get_layer("mobilenetv2_1.00_224")

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze earlier layers; fine-tune only the final 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep Batch Normalization layers frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Recompile using a very small learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nStarting fine-tuning...\n")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=5,
        callbacks=[early_stopping]
    )

    # Save as a separate model
    model.save(OUTPUT_PATH)

    print("\nFine-tuning completed!")
    print("Model saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    finetune_model()
