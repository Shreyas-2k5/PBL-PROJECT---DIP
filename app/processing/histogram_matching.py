import cv2
import numpy as np


def extract_skin_pixels(image, mask):
    """
    Extract pixels where mask > 0
    """
    return image[mask > 0]


def match_histograms_channel(source_channel, reference_channel):
    """
    Manual histogram matching for one channel
    """

    # Compute histograms
    src_hist, _ = np.histogram(
        source_channel.flatten(),
        256,
        [0,256]
    )

    ref_hist, _ = np.histogram(
        reference_channel.flatten(),
        256,
        [0,256]
    )

    # Compute CDFs
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Lookup table
    lookup = np.zeros(256, dtype=np.uint8)

    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup[src_val] = np.argmin(diff)

    return lookup


def match_to_reference(frame,
                       frame_mask,
                       reference_img,
                       reference_mask):
    """
    Match source skin tones to reference skin tones
    """

    result = frame.copy()

    # Convert both to LAB
    src_lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    ref_lab = cv2.cvtColor(
        reference_img,
        cv2.COLOR_BGR2LAB
    )

    # Split channels
    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    # Get skin pixels only
    src_l_skin = src_l[frame_mask > 0]
    src_a_skin = src_a[frame_mask > 0]
    src_b_skin = src_b[frame_mask > 0]

    ref_l_skin = ref_l[reference_mask > 0]
    ref_a_skin = ref_a[reference_mask > 0]
    ref_b_skin = ref_b[reference_mask > 0]

    # Safety check
    if len(src_l_skin)==0 or len(ref_l_skin)==0:
        return frame

    # Build mappings
    map_l = match_histograms_channel(
        src_l_skin,
        ref_l_skin
    )

    map_a = match_histograms_channel(
        src_a_skin,
        ref_a_skin
    )

    map_b = match_histograms_channel(
        src_b_skin,
        ref_b_skin
    )

    # Apply only on skin
    src_l[frame_mask > 0] = map_l[
        src_l[frame_mask > 0]
    ]

    src_a[frame_mask > 0] = map_a[
        src_a[frame_mask > 0]
    ]

    src_b[frame_mask > 0] = map_b[
        src_b[frame_mask > 0]
    ]

    # Merge back
    matched_lab = cv2.merge(
        [src_l, src_a, src_b]
    )

    result = cv2.cvtColor(
        matched_lab,
        cv2.COLOR_LAB2BGR
    )

    return result
