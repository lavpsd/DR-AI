import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load trained diabetic retinopathy model
model = tf.keras.models.load_model("diabetic_retinopathy_model.keras")

class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

# Page configuration
st.set_page_config(
    page_title="DR-AI",
    page_icon="👁️",
    layout="centered"
)

# Title
st.title("👁️ DR-AI")
st.subheader("AI-Powered Eye Screening")

st.write(
    "Upload a retinal image for preliminary AI-based screening."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")

    # Image information
    width, height = image.size

    st.write("### Image Information")
    st.write(f"Width: {width} pixels")
    st.write(f"Height: {height} pixels")

    # Basic quality check
    if width < 300 or height < 300:

        st.warning(
            "⚠️ Image quality may be insufficient. "
            "Please upload a higher-resolution retinal image."
        )

    else:

        st.success("✅ Image quality check passed.")

        # Screening
        st.subheader("Screening")

        if st.button("🔍 Analyze Image"):

           st.write("Analyzing image...")

# Prepare image for the model
img = image.resize((224, 224))
img_array = np.array(img)

# Make sure image has 3 color channels
if img_array.shape[-1] == 4:
    img_array = img_array[:, :, :3]

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction[0])
confidence = float(np.max(prediction[0])) * 100
predicted_class = class_names[predicted_index]

st.success("Analysis completed.")

st.subheader("Screening Result")

st.write("**Predicted condition:**", predicted_class)
st.write("**Confidence:**", f"{confidence:.2f}%")
            )
