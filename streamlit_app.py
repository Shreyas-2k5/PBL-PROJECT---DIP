import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from app.main import process_image, process_video


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


# ---------------- SIDEBAR ----------------
st.sidebar.header("🎨 Enhancement Controls")

brightness = st.sidebar.slider(
    "Brightness",
    0.5,
    2.0,
    1.2
)

smoothness = st.sidebar.slider(
    "Smoothness",
    0,
    10,
    3
)

mode = st.sidebar.selectbox(
    "Enhancement Mode",
    ["Natural", "Warm", "Bright"]
)


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

        output, mask = process_image(
            image,
            brightness=brightness,
            smoothness=smoothness,
            mode=mode
        )

        st.subheader("🖼 Before vs After")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original")
            st.image(image, channels="BGR")

        with col2:
            st.markdown("### Enhanced")
            st.image(output, channels="BGR")

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

        # Initial preview
        st.subheader("🎥 Video Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Before")
            with open(tfile.name, "rb") as v:
                st.video(v.read())

        with col2:
            st.markdown("### After")
            st.info(
                "Enhanced preview will appear after processing."
            )

        if st.button("🚀 Process Video"):

            os.makedirs(
                "app/outputs/videos",
                exist_ok=True
            )

            output_path = "app/outputs/videos/output.mp4"

            try:

                # -------- Progress Bar --------
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value):
                    progress_bar.progress(value)
                    status_text.text(
                        f"Processing: {int(value*100)}%"
                    )

                # -------- Process Video --------
                process_video(
                    tfile.name,
                    output_path,
                    brightness=brightness,
                    smoothness=smoothness,
                    mode=mode,
                    progress_callback=update_progress
                )

                status_text.text(
                    "Processing Complete ✅"
                )

                st.success(
                    "Video enhancement complete!"
                )

                # -------- Before / After --------
                st.subheader(
                    "🎬 Before vs After Comparison"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Original")
                    with open(tfile.name, "rb") as v:
                        st.video(v.read())

                with col2:
                    st.markdown("### Enhanced")
                    with open(output_path, "rb") as v:
                        st.video(v.read())

                # -------- Download --------
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Download Enhanced Video",
                        data=f,
                        file_name="enhanced.mp4"
                    )

            except Exception as e:
                st.error(
                    f"Error processing video: {e}"
                )