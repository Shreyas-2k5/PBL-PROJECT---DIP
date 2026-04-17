from fastapi import FastAPI
import cv2
import numpy as np

app = FastAPI(title="DIP Skin Processing API")


# ------------------ HEALTH CHECK ------------------
@app.get("/")
def test():
    return {"message": "working"}


# ------------------ SKIN MASK ------------------
def get_skin_mask(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Slight blur (reduced for better edges)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask


# ------------------ ENHANCEMENT ------------------
def enhance_skin(image, mask, brightness=1.0, smoothness=3, mode="Natural"):
    # Safe brightness scaling
    img = np.clip(image.astype(np.float32) * brightness, 0, 255)

    # Tone adjustment
    if mode == "Warm":
        img[:, :, 2] *= 1.1
    elif mode == "Bright":
        img *= 1.2

    # Smooth skin
    k = max(1, smoothness * 2 + 1)
    smooth = cv2.GaussianBlur(img, (k, k), 0)

    # Normalize mask properly
    mask = mask.astype(np.float32) / 255.0
    mask_3 = cv2.merge([mask, mask, mask])

    result = img * (1 - mask_3) + smooth * mask_3

    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------ IMAGE PIPELINE ------------------
def process_image(image, brightness=1.0, smoothness=3, mode="Natural"):
    mask = get_skin_mask(image)
    output = enhance_skin(image, mask, brightness, smoothness, mode)
    return output, mask


# ------------------ VIDEO PIPELINE ------------------
def process_video(
    input_path,
    output_path,
    brightness=1.0,
    smoothness=3,
    mode="Natural"
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # FPS safety
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
            smoothness=smoothness,
            mode=mode
        )

        out.write(processed_frame)

    cap.release()
    out.release()

    return output_path