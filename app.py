import streamlit as st

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="DR-AI | Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 60px 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f3443, #1b6b7a);
    color: white;
    text-align: center;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 52px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px;
    margin-bottom: 5px;
}

.subtitle {
    color: #d9f3f7;
    font-size: 16px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.info-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e5e9ef;
    height: 100%;
}

.info-card h3 {
    color: #123b4a;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">

<h1>👁️ DR-AI</h1>

<p><b>AI-Powered Diabetic Retinopathy Screening</b></p>

<div class="subtitle">
Analyze retinal images using an AI-based image classification model.
</div>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# INTRODUCTION
# -----------------------------
st.markdown(
    '<div class="section-title">🩺 About DR-AI</div>',
    unsafe_allow_html=True
)

st.write(
    """
    DR-AI is an AI-based prototype designed to analyze retinal images
    and classify them into different diabetic retinopathy categories.

    The system uses deep learning and transfer learning to identify
    patterns in retinal images.
    """
)


# -----------------------------
# THREE FEATURE CARDS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
    <h3>📷 Image Analysis</h3>
    <p>
    Upload a retinal image and prepare it for AI-based analysis.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
    <h3>🤖 AI Classification</h3>
    <p>
    A MobileNetV2-based deep learning model analyzes the image
    and predicts a retinal condition category.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
    <h3>📊 Screening Result</h3>
    <p>
    The system displays the predicted category along with
    the model's confidence score.
    </p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# CALL TO ACTION
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🔬 Start Screening</div>',
    unsafe_allow_html=True
)

st.info(
    "The image upload and camera capture functionality will appear here."
)


# -----------------------------
# DISCLAIMER
# -----------------------------
st.warning(
    """
    ⚠️ **Important:** DR-AI is a prototype intended for educational
    and preliminary screening purposes. It is not a medical diagnostic
    device. Results should not replace evaluation by a qualified
    eye-care professional.
    """
)


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
DR-AI • AI-Based Retinal Image Screening Prototype
</div>
""", unsafe_allow_html=True)
