import streamlit as st
from PIL import Image

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

       st.subheader("Screening")

if st.button("🔍 Analyze Image"):

    st.write("Analyzing image...")

    st.success("Analysis completed.")

    st.subheader("Screening Result")

    st.info(
        "AI model will be connected here."
    )
else:
    st.success("✅ Image quality check passed.")

    st.subheader("Screening")

    if st.button("🔍 Analyze Image"):

        st.write("Analyzing image...")

        st.success("Analysis completed.")

        st.subheader("Screening Result")

        st.info(
            "AI model will be connected here."
        )
