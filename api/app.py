
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Vehicle Damage Classifier",
    page_icon="🚗",
    layout="centered"
)

# Model configuration


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/vehicle_damage_finetuned_mlflow.keras"
)

CLASS_NAMES = ["Minor", "Moderate", "Severe"]

IMAGE_SIZE = (224, 224)


# Load the trained model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Predict vehicle damage
def predict_damage(image, model):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence, predictions[0]


# Application interface
st.title("🚗 Vehicle Damage Severity Classifier")

st.write(
    "Upload a vehicle damage image to classify "
    "the visible damage as Minor, Moderate, or Severe."
)

st.info(
    "This application provides an AI-based visual assessment. "
    "It does not determine official accident severity or repair costs."
)

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded vehicle image",
        use_container_width=True
    )

    if st.button("Predict Damage", type="primary"):

        with st.spinner("Analyzing vehicle image..."):

            try:
                model = load_model()

                predicted_class, confidence, scores = predict_damage(
                    image, model
                )

                st.subheader("Prediction Result")

                st.success(f"Predicted Damage: {predicted_class}")

                st.metric(
                    "Model confidence (uncalibrated)",
                    f"{confidence:.2f}%"
                )

                st.subheader("Class Scores")

                for class_name, score in zip(CLASS_NAMES, scores):
                    st.write(f"{class_name}: {score * 100:.2f}%")
                    st.progress(float(score))

                st.warning(
                    "This is an experimental model. "
                    "Borderline images should be reviewed by a person."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
