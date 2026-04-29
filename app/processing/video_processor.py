import cv2
import os

from app.processing.face_detection import detect_faces
from app.processing.skin_segmentation import get_skin_mask
from app.processing.noise_refinement import refine_mask
from app.processing.enhancement import enhance_skin


def process_video(
    video_path,
    output_path="app/outputs/videos/output.mp4",
    brightness=1.0,
    smoothness=3,
    mode="Natural",
   
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Could not open video"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps == 0:
        fps = 20

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Reduce heavy processing load
    target_fps = 20

    frame_interval = max(
        int(fps / target_fps),
        1
    )

    fourcc = cv2.VideoWriter_fourcc(
        *'mp4v'
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        raise ValueError(
            "VideoWriter failed to open"
        )

    frame_count = 0
    previous_faces = []

    # Used to avoid flicker
    last_processed_frame = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Progress callback
        if progress_callback and total_frames > 0:
            progress_callback(
                frame_count / total_frames
            )

        # Reuse last enhanced frame for skipped frames
        if frame_count % frame_interval != 0:

            if last_processed_frame is not None:
                out.write(
                    last_processed_frame
                )

            else:
                out.write(
                    frame
                )

            continue

        print(
            f"Processing frame {frame_count}"
        )

        # Resize for faster face detection
        small = cv2.resize(
            frame,
            (640, 360)
        )

        # Detect on first frame and periodically
        if (
            frame_count == 1 or
            frame_count % (
                frame_interval * 5
            ) == 0
        ):

            small_faces = detect_faces(
                small
            )

            previous_faces = []

            sx = width / 640
            sy = height / 360

            for (
                x, y, w, h
            ) in small_faces:

                previous_faces.append(
                    (
                        int(x * sx),
                        int(y * sy),
                        int(w * sx),
                        int(h * sy)
                    )
                )

        faces = previous_faces

        final_frame = frame.copy()

        for (
            x, y, w, h
        ) in faces:

            face_region = frame[
                y:y+h,
                x:x+w
            ]

            if face_region.size == 0:
                continue

            # Skin mask
            mask = get_skin_mask(
                face_region
            )

            # Refine mask
            refined_mask = refine_mask(
                mask
            )

            # Enhance only skin
            enhanced_face = enhance_skin(
                face_region,
                refined_mask
            )

            final_frame[
                y:y+h,
                x:x+w
            ] = enhanced_face

        # Store for skipped frames
        last_processed_frame = (
            final_frame.copy()
        )

        out.write(
            final_frame
        )

    cap.release()
    out.release()

    return output_path