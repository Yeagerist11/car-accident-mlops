
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3

def build_model():

    # Data augmentation
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Load MobileNetV2 with pretrained ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        # Apply augmentation during training
        data_augmentation,

        # Normalize pixels from 0-255 to -1 to 1
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.3),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
