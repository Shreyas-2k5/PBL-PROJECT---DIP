import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

face_cascade = None  # global variable


def get_face_cascade():
    global face_cascade

    if face_cascade is None:
        print("Loading cascade from:", CASCADE_PATH)

        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        if face_cascade.empty():
            raise IOError(f"❌ Failed to load Haar Cascade: {CASCADE_PATH}")

    return face_cascade


def detect_faces(frame):
    cascade = get_face_cascade()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    return faces