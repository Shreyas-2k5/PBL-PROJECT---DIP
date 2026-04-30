import cv2
import os
import numpy as np

from app.processing.face_detection import detect_faces
from app.processing.skin_segmentation import get_skin_mask
from app.processing.noise_refinement import refine_mask
from app.processing.enhancement import enhance_skin


def process_video(
    video_path,
    output_path="app/outputs/videos/raw.mp4",

    # 🔥 NEW CONTROLS
    brightness=0,
    exposure=1.0,
    saturation=0,
    smoothness=5,
    contrast=1.0,
    even_tone_strength=0.0,

    progress_callback=None,
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---------------- PERFORMANCE SETTINGS ----------------
    target_fps = 20
    frame_interval = max(int(fps / target_fps), 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (width, height)
    )

    if not out.isOpened():
        raise ValueError("VideoWriter failed to open")

    frame_count = 0
    previous_faces = []
    last_processed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---------------- PROGRESS ----------------
        if progress_callback and total_frames > 0:
            progress_callback(frame_count / total_frames)

        # ---------------- FRAME SKIPPING ----------------
        if frame_count % frame_interval != 0:
            frame_to_write = (
                last_processed_frame if last_processed_frame is not None else frame
            )

            frame_to_write = cv2.resize(frame_to_write, (width, height))
            frame_to_write = np.clip(frame_to_write, 0, 255).astype(np.uint8)

            if len(frame_to_write.shape) == 2:
                frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_GRAY2BGR)

            out.write(frame_to_write)
            continue

        # ---------------- FACE DETECTION (REDUCED) ----------------
        small = cv2.resize(frame, (640, 360))

        if frame_count == 1 or frame_count % (frame_interval * 5) == 0:
            small_faces = detect_faces(small)

            previous_faces = []
            sx = width / 640
            sy = height / 360

            for (x, y, w, h) in small_faces:
                previous_faces.append(
                    (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                )

        # ---------------- PROCESS FRAME ----------------
        final_frame = frame.copy()

        for (x, y, w, h) in previous_faces:
            face_region = frame[y:y+h, x:x+w]

            if face_region.size == 0:
                continue

            # Skin mask
            mask = get_skin_mask(face_region)
            refined_mask = refine_mask(mask)

            # 🔥 APPLY FULL ENHANCEMENT PIPELINE
            enhanced_face = enhance_skin(
                face_region,
                refined_mask,
                brightness=brightness,
                exposure=exposure,
                saturation=saturation,
                smoothness=smoothness,
                contrast=contrast,
                even_tone_strength=even_tone_strength
            )

            # Safety fixes
            enhanced_face = np.clip(enhanced_face, 0, 255).astype(np.uint8)

            if len(enhanced_face.shape) == 2:
                enhanced_face = cv2.cvtColor(enhanced_face, cv2.COLOR_GRAY2BGR)

            enhanced_face = cv2.resize(enhanced_face, (w, h))

            final_frame[y:y+h, x:x+w] = enhanced_face

        # ---------------- FINAL SAFETY ----------------
        final_frame = cv2.resize(final_frame, (width, height))
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

        if len(final_frame.shape) == 2:
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_GRAY2BGR)

        last_processed_frame = final_frame.copy()
        out.write(final_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ---------------- OUTPUT VALIDATION ----------------
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ValueError("Output video is empty or corrupted")

    return output_path
    