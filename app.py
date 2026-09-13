import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="DR-AI | Retinal Screening",
    page_icon="👁️",
    layout="wide"
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "diabetic_retinopathy_model.keras"
    )


model = load_model()


# =========================================================
# CLASS NAMES
# IMPORTANT: THIS ORDER MATCHES THE MODEL
# =========================================================

class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# =========================================================
# HEADER
# =========================================================

st.title("👁️ DR-AI")

st.subheader(
    "AI-Powered Diabetic Retinopathy Screening"
)

st.write(
    "An educational deep-learning prototype for retinal image classification."
)

st.divider()


# =========================================================
# ABOUT
# =========================================================

st.header("🩺 About DR-AI")

st.write(
    """
    DR-AI is an AI-based prototype designed to analyze retinal images
    and classify them into five diabetic retinopathy categories.

    The system uses transfer learning with a MobileNetV2-based
    deep learning model.
    """
)


# =========================================================
# FEATURE CARDS
# =========================================================

st.subheader("✨ What DR-AI Does")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.info(
        """
        **📷 Image Input**

        Upload a retinal image or capture one using a supported camera.
        """
    )

with feature2:
    st.info(
        """
        **🤖 AI Analysis**

        The trained deep learning model analyzes the retinal image.
        """
    )

with feature3:
    st.info(
        """
        **📊 Screening Result**

        The application displays the predicted category,
        confidence and probability distribution.
        """
    )


# =========================================================
# IMAGE INPUT
# =========================================================

st.divider()

st.header("🔬 Start Screening")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("📁 Upload Retinal Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with input_col2:
    st.subheader("📷 Camera Capture")

    camera_image = st.camera_input(
        "Take a photo",
        label_visibility="collapsed"
    )


# =========================================================
# SELECT IMAGE
# =========================================================

image_file = None

if uploaded_image is not None:
    image_file = uploaded_image

elif camera_image is not None:
    image_file = camera_image


# =========================================================
# IMAGE DISPLAY
# =========================================================

if image_file is not None:

    image = Image.open(image_file)

    original_width, original_height = image.size
    original_format = image.format

    image = image.convert("RGB")

    st.divider()

    st.header("🖼️ Retinal Image")

    image_col, details_col = st.columns([2, 1])

    with image_col:

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

    with details_col:

        st.subheader("📋 Image Details")

        st.metric(
            "Width",
            f"{original_width}px"
        )

        st.metric(
            "Height",
            f"{original_height}px"
        )

        format_name = original_format

        if format_name is None:
            format_name = "Image"

        st.metric(
            "Format",
            format_name
        )


        # =================================================
        # QUALITY CHECK
        # =================================================

        st.subheader("🔎 Input Quality Check")

        resolution_ok = (
            original_width >= 224
            and original_height >= 224
        )

        if resolution_ok:
            st.success("✅ Resolution check passed")
        else:
            st.warning(
                "⚠️ Resolution is below the model input size"
            )

        if original_format in ["JPEG", "PNG"]:
            st.success("✅ Supported image format")
        else:
            st.info(
                "ℹ️ Image will be converted to RGB"
            )

        if resolution_ok:
            st.success(
                "🟢 Image is ready for AI preprocessing"
            )
        else:
            st.warning(
                "🟡 Consider using a higher-resolution image"
            )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.divider()

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    if analyze:

        with st.spinner(
            "🤖 DR-AI is analyzing the retinal image..."
        ):

            # Resize image
            processed_image = image.resize(
                (224, 224)
            )

            # Convert to NumPy array
            img_array = np.array(
                processed_image,
                dtype=np.float32
            )

            # Normalize pixel values
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Get prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )

            # Find highest probability
            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                predictions[0][predicted_index] * 100
            )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header("📊 AI Screening Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.subheader("🎯 Predicted Category")

            st.success(
                predicted_class
            )

        with result_col2:

            st.subheader("🤖 Model Confidence")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.caption(
            "The confidence value represents the model's "
            "predicted probability for the selected category. "
            "It does not guarantee prediction accuracy."
        )


        # =================================================
        # PROBABILITY DISTRIBUTION
        # =================================================

        st.subheader("📈 Prediction Probabilities")

        for i in range(len(class_names)):

            class_name = class_names[i]

            probability = float(
                predictions[0][i] * 100
            )

            st.write(
                f"**{class_name}** — {probability:.2f}%"
            )

            st.progress(
                min(probability / 100, 1.0)
            )


# =========================================================
# HOW DR-AI WORKS
# =========================================================

st.divider()

st.header("⚙️ How DR-AI Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.info(
        """
        **1️⃣ Image Input**

        A retinal image is uploaded or captured.
        """
    )

with step2:
    st.info(
        """
        **2️⃣ Preprocessing**

        The image is converted to RGB,
        resized to 224 × 224 pixels
        and normalized.
        """
    )

with step3:
    st.info(
        """
        **3️⃣ AI Model**

        MobileNetV2 extracts visual features
        and performs classification.
        """
    )

with step4:
    st.info(
        """
        **4️⃣ Result**

        The predicted category and
        probabilities are displayed.
        """
    )


# =========================================================
# FIVE POSSIBLE CLASSES
# =========================================================

st.divider()

st.header("🧠 Five Possible Classes")

class1, class2, class3, class4, class5 = st.columns(5)

with class1:
    st.info(
        """
        **No DR**

        No diabetic retinopathy category.
        """
    )

with class2:
    st.info(
        """
        **Mild**

        Mild retinopathy category.
        """
    )

with class3:
    st.info(
        """
        **Moderate**

        Moderate retinopathy category.
        """
    )

with class4:
    st.info(
        """
        **Severe**

        Severe retinopathy category.
        """
    )

with class5:
    st.info(
        """
        **Proliferative DR**

        Proliferative retinopathy category.
        """
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.header("🤖 Model Information")

model1, model2, model3, model4 = st.columns(4)

with model1:
    st.metric(
        "Architecture",
        "MobileNetV2"
    )

with model2:
    st.metric(
        "Input Size",
        "224 × 224"
    )

with model3:
    st.metric(
        "Classes",
        "5"
    )

with model4:
    st.metric(
        "Method",
        "Transfer Learning"
    )

st.write(
    """
    DR-AI uses a pre-trained MobileNetV2 network as the
    feature-extraction backbone, followed by a classification
    head trained for the five retinal categories.
    """
)


# =========================================================
# LIMITATIONS
# =========================================================

st.divider()

st.header("⚠️ Limitations")

limit1, limit2 = st.columns(2)

with limit1:

    st.subheader("Technical Limitations")

    st.write(
        """
        - Performance depends on image quality.
        - Images that differ from the training data may
          produce less reliable predictions.
        - Different cameras and image-processing methods
          can affect performance.
        - The model can produce incorrect predictions.
        """
    )

with limit2:

    st.subheader("Important Interpretation")

    st.write(
        """
        - Confidence does not mean medical certainty.
        - A high-confidence prediction can still be incorrect.
        - This prototype has not been clinically validated.
        - Results should not replace professional eye examination.
        """
    )


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    DR-AI is an educational and preliminary screening prototype.
    It is NOT a medical diagnostic device and should not be used
    to make medical decisions.

    Please consult a qualified eye-care professional for proper
    evaluation.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "👁️ DR-AI • AI-Based Retinal Image Screening Prototype"
)

st.caption(
    "Built for educational and research purposes • 2026"
)
