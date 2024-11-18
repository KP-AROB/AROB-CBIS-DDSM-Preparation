import cv2
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_voi_lut


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
