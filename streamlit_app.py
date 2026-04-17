import streamlit as st
import cv2
import numpy as np
import tempfile

from app.main import process_image, process_video

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Skin Enhancement", layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🎥 Skin Tone Enhancement System")
st.markdown("Enhance skin tones in images and videos intelligently")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🎨 Controls")

brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.2)
smoothness = st.sidebar.slider("Smoothness", 0, 10, 3)

mode = st.sidebar.selectbox(
    "Enhancement Mode",
    ["Natural", "Warm", "Bright"]
)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "png", "jpeg", "mp4"]
)

# ------------------ MAIN LOGIC ------------------
if uploaded_file is not None:

    # -------- IMAGE --------
    if uploaded_file.type.startswith("image"):

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        output, mask = process_image(
            image,
            brightness=brightness,
            smoothness=smoothness,
            mode=mode
        )

        # -------- COMPARISON VIEW --------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Original")
            st.image(image, channels="BGR")

        with col2:
            st.subheader("✨ Enhanced")
            st.image(output, channels="BGR")

        # -------- MASK --------
        st.subheader("🧠 Skin Mask")
        st.image(mask, clamp=True)

        # -------- DOWNLOAD --------
        st.download_button(
            "⬇ Download Image",
            data=cv2.imencode('.png', output)[1].tobytes(),
            file_name="enhanced.png",
            mime="image/png"
        )

    # -------- VIDEO --------
    elif uploaded_file.type == "video/mp4":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.subheader("🎬 Original Video")
        st.video(tfile.name)

        if st.button("🚀 Process Video"):

            output_path = "outputs/output.mp4"

            process_video(tfile.name, output_path)

            st.success("Processing Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.video(tfile.name)

            with col2:
                st.subheader("Enhanced")
                st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇ Download Video",
                    f,
                    file_name="enhanced.mp4"
                )