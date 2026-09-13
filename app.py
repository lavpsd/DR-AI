import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DR-AI | AI Retinal Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------- GENERAL ---------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}


/* ---------- HERO ---------- */

.hero {
    padding: 55px 35px;
    border-radius: 24px;
    background: linear-gradient(135deg, #0b3442, #176b7c);
    color: white;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 58px;
    margin: 0;
    font-weight: 800;
}

.hero-main {
    font-size: 23px;
    font-weight: 600;
    margin-top: 10px;
}

.hero-sub {
    font-size: 16px;
    margin-top: 12px;
    opacity: 0.9;
}


/* ---------- SECTION TITLES ---------- */

.section-title {
    font-size: 29px;
    font-weight: 750;
    margin-top: 32px;
    margin-bottom: 18px;
}


/* ---------- FEATURE CARDS ---------- */

.feature-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 175px;
    background: rgba(128,128,128,0.06);
}

.feature-card h3 {
    margin-top: 0;
    font-size: 21px;
}


/* ---------- RESULT CARD ---------- */

.result-box {
    padding: 25px;
    border-radius: 18px;
    border: 2px solid rgba(23,107,124,0.35);
    background: rgba(23,107,124,0.08);
    margin-top: 10px;
}

.result-label {
    font-size: 15px;
    opacity: 0.75;
    margin-bottom: 5px;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
}


/* ---------- CLASS CARDS ---------- */

.class-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 120px;
    background: rgba(128,128,128,0.05);
}

.class-card h4 {
    margin-top: 0;
    font-size: 19px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    opacity: 0.7;
    margin-top: 50px;
    padding: 25px;
    border-top: 1px solid rgba(128,128,128,0.25);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "diabetic_retinopathy_model.keras"
    )


model = load_model()


# =========================================================
# CLASS NAMES
# IMPORTANT: KEEP THIS ORDER
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

st.title("👁️ DR-AI")

st.subheader(
    "AI-Powered Diabetic Retinopathy Screening"
)

st.write(
    "An educational deep-learning prototype for retinal image classification."
)

st.markdown("---")


# =========================================================
# ABOUT DR-AI
# =========================================================

st.markdown(
    '<div class="section-title">🩺 About DR-AI</div>',
    unsafe_allow_html=True
)

st.write("""
**DR-AI** is an AI-based prototype that analyzes retinal images
and classifies them into five diabetic retinopathy categories.

The system uses **transfer learning** with a MobileNetV2-based
deep learning model trained on retinal image data.
""")


# =========================================================
# FEATURE CARDS
# =========================================================

feature1, feature2, feature3 = st.columns(3)


with feature1:

    st.markdown("""
    <div class="feature-card">

    <h3>📷 Image Input</h3>

    <p>
    Upload a retinal image from your device or capture an image
    using a supported camera.
    </p>

    </div>
    """, unsafe_allow_html=True)


with feature2:

    st.markdown("""
    <div class="feature-card">

    <h3>🤖 AI Analysis</h3>

    <p>
    The trained MobileNetV2-based model processes the image
    and identifies the most probable category.
    </p>

    </div>
    """, unsafe_allow_html=True)


with feature3:

    st.markdown("""
    <div class="feature-card">

    <h3>📊 Screening Result</h3>

    <p>
    The application displays the predicted category,
    confidence and probability distribution.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# START SCREENING
# =========================================================

st.markdown(
    '<div class="section-title">🔬 Start Screening</div>',
    unsafe_allow_html=True
)


input_col1, input_col2 = st.columns(2)


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

with input_col1:

    st.markdown("### 📁 Upload Retinal Image")

    uploaded_image = st.file_uploader(
        "Choose a retinal image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

with input_col2:

    st.markdown("### 📷 Camera Capture")

    camera_image = st.camera_input(
        "Capture retinal image",
        label_visibility="collapsed"
    )


# =========================================================
# CHOOSE IMAGE SOURCE
# =========================================================

image_file = uploaded_image

if image_file is None:

    image_file = camera_image


# =========================================================
# IMAGE PROCESSING
# =========================================================

if image_file is not None:

    # Open original image
    image = Image.open(image_file)

    # Save original information
    original_width, original_height = image.size
    original_format = image.format

    # Convert to RGB
    image = image.convert("RGB")


    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🖼️ Retinal Image</div>',
        unsafe_allow_html=True
    )


    image_col, details_col = st.columns([2.2, 1])


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with image_col:

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )


    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    with details_col:

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


        # =================================================
        # QUALITY CHECK
        # =================================================

        st.markdown("### 🔎 Input Quality Check")


        # Resolution
        if original_width >= 224 and original_height >= 224:

            st.success("✅ Resolution check passed")

        else:

            st.warning(
                "⚠️ Resolution is below model input size"
            )


        # Format
        if original_format in ["JPEG", "PNG"]:

            st.success("✅ Supported image format")

        else:

            st.info(
                "ℹ️ Image will be converted to RGB"
            )


        # Overall preprocessing status
        if original_width >= 224 and original_height >= 224:

            st.info(
                "🟢 Image is ready for AI preprocessing"
            )

        else:

            st.warning(
                "🟡 Consider using a higher-resolution image"
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
    # AI ANALYSIS
    # =====================================================

    if analyze:

        with st.spinner(
            "🤖 DR-AI is analyzing the retinal image..."
        ):

            # ---------------------------------------------
            # RESIZE
            # ---------------------------------------------

            processed_image = image.resize(
                (224, 224)
            )


            # ---------------------------------------------
            # NUMPY ARRAY
            # ---------------------------------------------

            img_array = np.array(
                processed_image,
                dtype=np.float32
            )


            # ---------------------------------------------
            # NORMALIZATION
            # ---------------------------------------------

            img_array = img_array / 255.0


            # ---------------------------------------------
            # BATCH DIMENSION
            # ---------------------------------------------

            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            # ---------------------------------------------
            # MODEL PREDICTION
            # ---------------------------------------------

            predictions = model.predict(
                img_array,
                verbose=0
            )


            # ---------------------------------------------
            # FIND PREDICTED CLASS
            # ---------------------------------------------

            predicted_index = int(
                np.argmax(predictions[0])
            )


            predicted_class = class_names[
                predicted_index
            ]


            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            confidence = float(
                predictions[0][predicted_index] * 100
            )


        # =================================================
        # RESULT SECTION
        # =================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📊 AI Screening Result</div>',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns(2)


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with result_col1:

            st.markdown(
                f"""
                <div class="result-box">

                    <div class="result-label">
                        🎯 Predicted Category
                    </div>

                    <div class="result-value">
                        {predicted_class}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        with result_col2:

            st.markdown(
                f"""
                <div class="result-box">

                    <div class="result-label">
                        🤖 Model Confidence
                    </div>

                    <div class="result-value">
                        {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.caption(
            "The confidence value represents the model's predicted "
            "probability for the selected category. It does not "
            "guarantee prediction accuracy."
        )


        # =================================================
        # PROBABILITY DISTRIBUTION
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


how_col1, how_col2, how_col3, how_col4 = st.columns(4)


with how_col1:

    st.markdown("""
    <div class="feature-card">

    <h3>1️⃣ Input</h3>

    <p>
    A retinal image is uploaded or captured.
    </p>

    </div>
    """, unsafe_allow_html=True)


with how_col2:

    st.markdown("""
    <div class="feature-card">

    <h3>2️⃣ Preprocessing</h3>

    <p>
    The image is converted to RGB,
    resized to 224 × 224 and normalized.
    </p>

    </div>
    """, unsafe_allow_html=True)


with how_col3:

    st.markdown("""
    <div class="feature-card">

    <h3>3️⃣ AI Model</h3>

    <p>
    MobileNetV2 extracts visual features
    and performs classification.
    </p>

    </div>
    """, unsafe_allow_html=True)


with how_col4:

    st.markdown("""
    <div class="feature-card">

    <h3>4️⃣ Result</h3>

    <p>
    The system displays the predicted category
    and probability distribution.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FIVE POSSIBLE CLASSES
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 Five Possible Classes</div>',
    unsafe_allow_html=True
)


class1, class2, class3, class4, class5 = st.columns(5)


with class1:

    st.markdown("""
    <div class="class-card">

    <h4>🟢 No DR</h4>

    <p>
    No diabetic retinopathy category.
    </p>

    </div>
    """, unsafe_allow_html=True)


with class2:

    st.markdown("""
    <div class="class-card">

    <h4>🟡 Mild</h4>

    <p>
    Mild retinopathy category.
    </p>

    </div>
    """, unsafe_allow_html=True)


with class3:

    st.markdown("""
    <div class="class-card">

    <h4>🟠 Moderate</h4>

    <p>
    Moderate retinopathy category.
    </p>

    </div>
    """, unsafe_allow_html=True)


with class4:

    st.markdown("""
    <div class="class-card">

    <h4>🔴 Severe</h4>

    <p>
    Severe retinopathy category.
    </p>

    </div>
    """, unsafe_allow_html=True)


with class5:

    st.markdown("""
    <div class="class-card">

    <h4>🔴 Proliferative DR</h4>

    <p>
    Proliferative retinopathy category.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)


model_col1, model_col2, model_col3, model_col4 = st.columns(4)


with model_col1:

    st.metric(
        "Architecture",
        "MobileNetV2"
    )


with model_col2:

    st.metric(
        "Input Size",
        "224 × 224"
    )


with model_col3:

    st.metric(
        "Classes",
        "5"
    )


with model_col4:

    st.metric(
        "Technique",
        "Transfer Learning"
    )


st.write("""
The model uses a **pre-trained MobileNetV2 network** as the
feature extraction backbone and a classification head trained
for the five retinal categories used in this prototype.
""")


# =========================================================
# LIMITATIONS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">⚠️ Limitations</div>',
    unsafe_allow_html=True
)


lim_col1, lim_col2 = st.columns(2)


with lim_col1:

    st.write("""
    **Technical limitations**

    - Performance depends on image quality.
    - Images that differ significantly from the training data
      may produce less reliable predictions.
    - Different cameras and image-processing methods can
      affect model performance.
    - The model can make incorrect predictions.
    """)


with lim_col2:

    st.write("""
    **Important interpretation**

    - Confidence does not mean medical certainty.
    - A high-confidence prediction can still be incorrect.
    - This prototype has not been clinically validated.
    - Results should not be used as a substitute for
      professional eye examination.
    """)


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.warning("""
⚠️ **Medical Disclaimer**

DR-AI is an educational and preliminary screening prototype.
It is **not a medical diagnostic device** and should not be
used to make medical decisions.

For medical concerns or suspected diabetic retinopathy,
consult a qualified eye-care professional.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

<h3>👁️ DR-AI</h3>

AI-Based Retinal Image Screening Prototype

<br>

Built for educational and research purposes

<br><br>

© 2026 DR-AI

</div>
""", unsafe_allow_html=True)
