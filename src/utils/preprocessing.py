import numpy as np


def truncate_normalization(img: np.array, mask: np.array):
    """Normalize an image within a given ROI mask

    Args:
        img (np.array): original image to normalize
        mask (np.array): roi mask

    Returns:
        np.array: normalized image
    """
    Pmin = np.percentile(img[mask != 0], 2)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    return np.array(normalized * 255, dtype=np.uint8)
