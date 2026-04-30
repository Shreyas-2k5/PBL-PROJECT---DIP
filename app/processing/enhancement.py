import cv2
import numpy as np


# ------------------ BRIGHTNESS ------------------
def adjust_brightness(img, value=0):
    if value == 0:
        return img

    return cv2.convertScaleAbs(img, alpha=1, beta=value)


# ------------------ EXPOSURE ------------------
def adjust_exposure(img, value=1.0):
    if value == 1.0:
        return img

    return cv2.convertScaleAbs(img, alpha=value, beta=0)


# ------------------ SATURATION ------------------
def adjust_saturation(img, value=0):
    if value == 0:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.add(s, value)
    s = np.clip(s, 0, 255)

    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------ SMOOTHNESS ------------------
def smooth_skin(img, strength=5):
    if strength <= 1:
        return img

    return cv2.bilateralFilter(
        img,
        d=9,
        sigmaColor=strength * 10,
        sigmaSpace=75
    )


# ------------------ CONTRAST ------------------
def adjust_contrast(img, value=1.0):
    if value == 1.0:
        return img

    return cv2.convertScaleAbs(img, alpha=value, beta=0)


# ------------------ EVEN SKIN TONE ------------------
def even_skin_tone(img, strength=0.0):
    if strength <= 0:
        return img

    blur = cv2.GaussianBlur(img, (0, 0), 10)
    return cv2.addWeighted(img, 1 - strength, blur, strength, 0)


# ------------------ MAIN ENHANCEMENT ------------------
def enhance_skin(
    frame,
    mask,
    brightness=0,
    exposure=1.0,
    saturation=0,
    smoothness=5,
    contrast=1.0,
    even_tone_strength=0.0
):
    """
    Full enhancement pipeline (skin-only)

    Parameters:
    - brightness: (-50 to +50)
    - exposure: (0.8 to 1.5)
    - saturation: (-30 to +50)
    - smoothness: (1 to 10)
    - contrast: (0.8 to 1.5)
    - even_tone_strength: (0.0 to 0.7)
    """

    if mask is None:
        return frame

    # Extract skin
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    # Apply enhancements (order matters!)
    skin = adjust_exposure(skin, exposure)
    skin = adjust_brightness(skin, brightness)
    skin = adjust_saturation(skin, saturation)
    skin = smooth_skin(skin, smoothness)
    skin = adjust_contrast(skin, contrast)
    skin = even_skin_tone(skin, even_tone_strength)

    # Merge back
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    final = cv2.add(background, skin)

    return final
    