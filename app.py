import streamlit as st

# Page configuration
st.set_page_config(
    page_title="DR-AI",
    page_icon="👁️",
    layout="centered"
)

# Title
st.title("👁️ DR-AI")
st.subheader("AI-Powered Eye Screening Prototype")

st.write(
    "Upload an eye image to perform preliminary diabetic retinopathy screening."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["jpg", "jpeg", "png"]
)

# Show uploaded image
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Eye Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")

    # Temporary result
    st.subheader("Screening Result")
    st.info("AI analysis will be connected here.")
