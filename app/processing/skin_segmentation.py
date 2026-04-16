import cv2
import numpy as np
from app.config import LOWER_YCRCB, UPPER_YCRCB

def get_skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, LOWER_YCRCB, UPPER_YCRCB)
    return mask
