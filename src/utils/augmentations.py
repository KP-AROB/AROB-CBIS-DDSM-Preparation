import os
import cv2
import logging
import albumentations as A
from glob import glob
from tqdm import tqdm


def make_augmentation(data_dir, num_augmentations: int = 3):

    logging.info("Running data augmentation")

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), p=0.5),
    ])

    label_folders = glob(os.path.join(data_dir, '*'))

    def augment_image(image_path):
        image = cv2.imread(image_path)
        augmented_images = []
        for _ in range(num_augmentations):
            augmented = augmentation_pipeline(image=image)['image']
            augmented_images.append(augmented)
        return augmented_images

    for label in label_folders:
        number_of_images = glob(label + '/*.png')
        with tqdm(total=len(number_of_images), desc=f"Augmenting {label} images") as pbar:
            for i, img in enumerate(number_of_images):
                augmented_images = augment_image(img)
                for j, augmented_image in enumerate(augmented_images):
                    output_path = f"{label}/aug_{i}_{j}.png"
                    cv2.imwrite(output_path, augmented_image)
                pbar.update()

    logging.info("Augmentations finished.")
