import cv2

def enhance_skin(frame, mask):
    enhanced = frame.copy()

    enhanced[mask > 0] = cv2.convertScaleAbs(
        enhanced[mask > 0],
        alpha=1.1,   # brightness/contrast
        beta=10
    )

    return enhanced
