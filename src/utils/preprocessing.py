import cv2
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


def clahe(img, clip=1.5):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(img)
    return cl
