import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DR-AI | Retinal Screening",
    page_icon="👁️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 55px 35px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f3443, #1b6b7a);
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 52px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 20px;
    margin: 5px;
}

.hero-small {
    color: #d8eef2;
    font-size: 15px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e3e8ee;
    min-height: 160px;
}

.info-card h3 {
    color: #123b4a;
}

.result-card {
    background-color: white;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #e3e8ee;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 45px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "diabetic_retinopathy_model.keras"
    )


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>👁️ DR-AI</h1>

<p><b>AI-Powered Diabetic Retinopathy Screening</b></p>

<div class="hero-small">
Analyze retinal images using a deep learning image classification model.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# ABOUT DR-AI
# =========================================================

st.markdown(
    '<div class="section-title">🩺 About DR-AI</div>',
    unsafe_allow_html=True
)

st.write("""
DR-AI is an AI-based prototype designed to analyze retinal images
and classify them into five diabetic retinopathy categories.

The system uses transfer learning with a MobileNetV2-based
deep learning model.
""")


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="info-card">

    <h3>📷 Image Analysis</h3>

    <p>
    Upload or capture a retinal image for AI-based analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="info-card">

    <h3>🤖 AI Classification</h3>

    <p>
    A MobileNetV2-based model analyzes visual patterns
    in the retinal image.
    </p>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="info-card">

    <h3>📊 Screening Result</h3>

    <p>
    The system displays the predicted category
    and model confidence.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# IMAGE INPUT
# =========================================================

st.markdown(
    '<div class="section-title">🔬 Start Screening</div>',
    unsafe_allow_html=True
)


input_col1, input_col2 = st.columns(2)


with input_col1:

    st.markdown("### 📁 Upload Retinal Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


with input_col2:

    st.markdown("### 📷 Camera Capture")

    camera_image = st.camera_input(
        "Capture retinal image",
        label_visibility="collapsed"
    )


# =========================================================
# SELECT IMAGE
# =========================================================

image_file = uploaded_image

if image_file is None:
    image_file = camera_image


# =========================================================
# IMAGE DISPLAY
# =========================================================

if image_file is not None:

    image = Image.open(image_file)

    original_width, original_height = image.size
    original_format = image.format

    # Convert image to RGB
    image = image.convert("RGB")

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🖼️ Retinal Image</div>',
        unsafe_allow_html=True
    )


    display_col1, display_col2 = st.columns([2, 1])


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with display_col1:

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )


    # -----------------------------------------------------
    # IMAGE DETAILS
    # -----------------------------------------------------

    with display_col2:

        st.markdown("### 📋 Image Details")

        st.metric(
            "Width",
            f"{original_width}px"
        )

        st.metric(
            "Height",
            f"{original_height}px"
        )

        st.metric(
            "Format",
            original_format if original_format else "Image"
        )


        # -------------------------------------------------
        # QUALITY CHECK
        # -------------------------------------------------

        st.markdown("### 🔎 Input Quality Check")


        if original_width >= 224 and original_height >= 224:

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


        if original_width >= 224 and original_height >= 224:

            st.info(
                "🟢 Image is ready for AI preprocessing"
            )

        else:

            st.warning(
                "🟡 Image may require higher resolution"
            )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.markdown("---")


    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # AI PREDICTION
    # =====================================================

    if analyze:

        with st.spinner(
            "🤖 AI is analyzing the retinal image..."
        ):

            # Resize image
            processed_image = image.resize(
                (224, 224)
            )


            # Convert image to NumPy array
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


            # Model prediction
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

        st.markdown(
            '<div class="section-title">📊 AI Screening Result</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.markdown("### Predicted Category")

            st.subheader(
                predicted_class
            )


        with result_col2:

            st.markdown("### Model Confidence")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # PROBABILITIES
        # =================================================

        st.markdown(
            "### 📈 Prediction Probabilities"
        )


        for i, class_name in enumerate(class_names):

            probability = float(
                predictions[0][i] * 100
            )


            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )


            st.progress(
                min(probability / 100, 1.0)
            )


# =========================================================
# HOW DR-AI WORKS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">⚙️ How DR-AI Works</div>',
    unsafe_allow_html=True
)

st.write("""
**1. Image Input**

A retinal image is uploaded or captured.

**2. Preprocessing**

The image is converted to RGB, resized to
224 × 224 pixels and normalized.

**3. Feature Extraction**

MobileNetV2 extracts useful visual features
from the retinal image.

**4. Classification**

A trained neural-network classifier uses those
features to predict one of five categories.

**5. Result**

DR-AI displays the predicted category and
the model's confidence.
""")


# =========================================================
# FIVE POSSIBLE CLASSES
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Possible Classes</div>',
    unsafe_allow_html=True
)


class_col1, class_col2 = st.columns(2)


with class_col1:

    st.write("• **No DR**")

    st.write("• **Mild**")

    st.write("• **Moderate**")


with class_col2:

    st.write("• **Severe**")

    st.write("• **Proliferative DR**")


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.info("""
    **Architecture**

    MobileNetV2
    """)


with model_col2:

    st.info("""
    **Technique**

    Transfer Learning
    """)


with model_col3:

    st.info("""
    **Input Size**

    224 × 224 pixels
    """)


# =========================================================
# LIMITATIONS
# =========================================================

st.markdown(
    '<div class="section-title">⚠️ Limitations</div>',
    unsafe_allow_html=True
)

st.write("""
- Model performance depends on image quality and similarity
  to the training data.
- The model can produce incorrect predictions.
- Confidence scores do not guarantee prediction accuracy.
- Images from different cameras or datasets may produce
  different results.
- The model has not been clinically validated.
""")


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.warning("""
⚠️ **Medical Disclaimer**

DR-AI is an educational and preliminary screening prototype.
It is **not a medical diagnostic device** and should not be
used to make medical decisions.

Please consult a qualified eye-care professional for
proper evaluation.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

<b>DR-AI</b> • AI-Based Retinal Image Screening Prototype

<br><br>

Built for educational and research purposes

</div>
""", unsafe_allow_html=True)
