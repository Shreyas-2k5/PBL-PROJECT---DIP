import cv2

from app.processing.face_detection import detect_faces
from app.processing.skin_segmentation import get_skin_mask
from app.processing.noise_refinement import refine_mask
from app.processing.enhancement import enhance_skin


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

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

        cv2.imshow("Original", frame)
        cv2.imshow("Processed", final_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()