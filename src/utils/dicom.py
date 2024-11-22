import cv2
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_voi_lut


def load_dicom_mask(roi_paths: list, x_shape: tuple):
    """Loads the ROI mask associated to a mammogram.
    As some of the studies have 2 ROI files (one mask and one image patch) and that no information
    is given regarding their nature. This function checks which mask image has the same shape as the original data.
    If no match is found, the smallest image is returned, being the abnormality image patch.

    Args:
        roi_paths (list): List of ROI files
        x_shape (tuple): Shape of the original mammogram image

    Returns:
        np.array: image mask or image patch
    """
    if len(roi_paths) > 1:
        base_mask = load_dicom_image(roi_paths[0])
        second_mask = load_dicom_image(roi_paths[1])

        if base_mask.shape == (x_shape[0], x_shape[1]):
            return base_mask
        elif second_mask.shape == (x_shape[0], x_shape[1]):
            return second_mask
        else:
            return None
    else:
        mask = load_dicom_image(roi_paths[0])
        return mask


def load_dicom_image(path):
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d
    img2d = cv2.normalize(
        img2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)
    return img2d
