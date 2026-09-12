import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DR-AI",
    page_icon="👁️",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("diabetic_retinopathy_model.keras")


model = load_model()


# Class names must match the training class indices
class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("👁️ DR-AI")
st.subheader("AI-Powered Eye Screening")

st.write(
    "Upload a retinal image for preliminary AI-based screening."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS UPLOADED IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    # --------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------

    width, height = image.size

    st.write("### Image Information")
    st.write(f"Width: {width} pixels")
    st.write(f"Height: {height} pixels")


    # --------------------------------------------------
    # BASIC IMAGE QUALITY CHECK
    # --------------------------------------------------

    if width < 300 or height < 300:

        st.warning(
            "⚠️ Image quality may be insufficient. "
            "Please upload a higher-resolution retinal image."
        )

    else:

        st.success("✅ Image quality check passed.")


    # --------------------------------------------------
    # SCREENING
    # --------------------------------------------------

    st.subheader("Screening")

    if st.button("🔍 Analyze Image"):

        st.write("Analyzing image...")


        # --------------------------------------------------
        # PREPARE IMAGE FOR MODEL
        # --------------------------------------------------

        img = image.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)


        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])

        confidence = float(
            np.max(prediction[0]) * 100
        )

        predicted_class = class_names[predicted_index]


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.success("Analysis completed.")

        st.subheader("Screening Result")

        st.write(
            f"**Predicted condition:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.info(
            "⚠️ This AI result is for preliminary screening "
            "and educational purposes only. It is not a "
            "medical diagnosis. Please consult a qualified "
            "eye-care professional for proper evaluation."
        )
