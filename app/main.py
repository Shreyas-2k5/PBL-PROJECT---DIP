import cv2
import numpy as np
from fastapi import FastAPI

from app.processing.skin_segmentation import get_skin_mask
from app.processing.noise_refinement import refine_mask
from app.processing.enhancement import enhance_skin


app = FastAPI(title="DIP Skin Processing API")


# ------------------ HEALTH CHECK ------------------
@app.get("/")
def test():
    return {"message": "working"}


# ------------------ IMAGE PIPELINE ------------------
def process_image(
    image,
    brightness=0,
    exposure=1.0,
    saturation=0,
    smoothness=5,
    contrast=1.0,
    even_tone_strength=0.0
):
    mask = get_skin_mask(image)
    mask = refine_mask(mask)

    output = enhance_skin(
        image,
        mask,
        brightness=brightness,
        exposure=exposure,
        saturation=saturation,
        smoothness=smoothness,
        contrast=contrast,
        even_tone_strength=even_tone_strength
    )

    return output, mask


# ------------------ VIDEO PIPELINE ------------------
def process_video(
    input_path,
    output_path,
    brightness=0,
    exposure=1.0,
    saturation=0,
    smoothness=5,
    contrast=1.0,
    even_tone_strength=0.0
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Error initializing video writer")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _ = process_image(
            frame,
            brightness=brightness,
            exposure=exposure,
            saturation=saturation,
            smoothness=smoothness,
            contrast=contrast,
            even_tone_strength=even_tone_strength
        )

        out.write(processed_frame)

    cap.release()
    out.release()

    return output_path
    