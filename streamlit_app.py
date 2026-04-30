import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import imageio

from app.main import process_image, process_video


# ---------------- PURE PYTHON VIDEO CONVERTER ----------------
def convert_video_py(input_path, output_path):
    reader = imageio.get_reader(input_path)

    try:
        fps = reader.get_meta_data().get('fps', 20)
    except:
        fps = 20

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264'
    )

    for frame in reader:
        writer.append_data(frame)

    writer.close()


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skin Tone Enhancement System",
    layout="wide"
)


# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1,h2,h3 {
    color:white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.title("🎥 Skin Tone Enhancement System")
st.markdown(
    "Enhance detected skin regions in images and videos intelligently."
)


# ---------------- SIDEBAR (UPDATED CONTROLS) ----------------
st.sidebar.header("🎨 Enhancement Controls")

brightness = st.sidebar.slider("Brightness", -50, 50, 10)
exposure = st.sidebar.slider("Exposure", 0.8, 1.5, 1.1)
saturation = st.sidebar.slider("Saturation", -30, 50, 20)
smoothness = st.sidebar.slider("Smoothness", 1, 10, 5)
contrast = st.sidebar.slider("Contrast", 0.8, 1.5, 1.2)
even_tone = st.sidebar.slider("Even Skin Tone", 0.0, 0.7, 0.3)

show_mask = st.sidebar.checkbox("Show Skin Mask", value=True)


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)


# ==================================================
# MAIN
# ==================================================
if uploaded_file is not None:

    # ==============================================
    # IMAGE
    # ==============================================
    if uploaded_file.type.startswith("image"):

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, 1)

        # 🔥 UPDATED FUNCTION CALL
        output, mask = process_image(
            image,
            brightness=brightness,
            exposure=exposure,
            saturation=saturation,
            smoothness=smoothness,
            contrast=contrast,
            even_tone_strength=even_tone
        )

        st.subheader("🖼 Before vs After")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original")
            st.image(image, channels="BGR")

        with col2:
            st.markdown("### Enhanced")
            st.image(output, channels="BGR")

        if show_mask:
            st.subheader("🧠 Skin Mask")
            st.image(mask, clamp=True)

        st.download_button(
            "⬇ Download Enhanced Image",
            data=cv2.imencode(".png", output)[1].tobytes(),
            file_name="enhanced.png",
            mime="image/png"
        )

    # ==============================================
    # VIDEO
    # ==============================================
    elif uploaded_file.type == "video/mp4":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        if st.button("🚀 Process Video"):

            os.makedirs("app/outputs/videos", exist_ok=True)

            raw_output = "app/outputs/videos/raw.mp4"
            final_output = "app/outputs/videos/output.mp4"

            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value):
                    progress_bar.progress(value)
                    status_text.text(f"Processing: {int(value*100)}%")

                # 🔥 UPDATED FUNCTION CALL
                process_video(
                    tfile.name,
                    raw_output,
                    brightness=brightness,
                    exposure=exposure,
                    saturation=saturation,
                    smoothness=smoothness,
                    contrast=contrast,
                    even_tone_strength=even_tone
                )

                convert_video_py(raw_output, final_output)

                status_text.text("Processing Complete ✅")
                st.success("Video enhancement complete!")

                st.subheader("🎬 Before vs After Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Original")
                    with open(tfile.name, "rb") as v:
                        st.video(v.read())

                with col2:
                    st.markdown("### Enhanced")
                    with open(final_output, "rb") as v:
                        st.video(v.read())

                with open(final_output, "rb") as f:
                    st.download_button(
                        "⬇ Download Enhanced Video",
                        data=f,
                        file_name="enhanced.mp4"
                    )

            except Exception as e:
                st.error(f"Error processing video: {e}")
                