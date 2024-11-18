import cv2
import random
import numpy as np


def crop_to_roi(img):
    """Crop a mammogram to breast ROI

    Args:
        img (np.array): original image

    Returns:
        tuple: (cropped image, cropping mask, bounding box)
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        breast_mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y: y + h, x: x + w], breast_mask[y: y + h, x: x + w], [x, y, w, h]


def crop_img(img, coordinates):
    x, y, w, h = coordinates
    return img[y: y + h, x: x + w]


def extract_patch(image, mask):
    padding = 1
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_max = min(image.shape[1], x_max + padding)

    return image[y_min:y_max, x_min:x_max]


def random_crop(image, size=(200, 200)):
    height, width = image.shape[:2]
    top = random.randint(0, height - size[0])
    left = random.randint(0, width - size[1])
    return image[top:top + size[0], left:left + size[1]]
