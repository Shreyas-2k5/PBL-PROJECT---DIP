import cv2
import os

from app.processing.face_detection import detect_faces
from app.processing.skin_segmentation import get_skin_mask
from app.processing.noise_refinement import refine_mask
from app.processing.enhancement import enhance_skin


def process_video(video_path, output_path="app/outputs/videos/output.mp4"):

    # ✅ Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("❌ Error: Could not open video.")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 20  # fallback

    # ✅ Use stable codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("❌ Error initializing video writer")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        final_frame = frame.copy()

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]

            mask = get_skin_mask(face_region)
            refined_mask = refine_mask(mask)

            enhanced_face = enhance_skin(face_region, refined_mask)

            final_frame[y:y+h, x:x+w] = enhanced_face

        # ✅ Write processed frame
        out.write(final_frame)

    cap.release()
    out.release()

    return output_path