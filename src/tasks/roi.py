import shutil
import os
import logging
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob
from src.utils.dicom import load_dicom_image, load_dicom_mask
from src.utils.crop import random_crop, extract_patch, crop_to_roi, crop_img
from src.utils.preprocessing import truncate_normalization
from concurrent.futures import ProcessPoolExecutor


def prepare_roi_severity_row(row, data_dir: str, out_folder: str, img_size: int):
    try:
        sev = 'BENIGN' if row['pathology'] == 'BENIGN_WITHOUT_CALLBACK' else row['pathology']
        image_path = os.path.join(data_dir, row['image_file_path'])
        mask_path = os.path.join(data_dir, row['roi_mask_file_path'])

        # 1. Load image, crop and normalize breast region
        image = load_dicom_image(glob(image_path + '/*.dcm')[0])
        cropped_image, cropped_roi, bounding_box = crop_to_roi(image)
        normalized_image = truncate_normalization(
            cropped_image, cropped_roi)

        # 2. Load mask, crop using the cropped image bounding box
        mask = load_dicom_mask(glob(mask_path + '/*.dcm'), image.shape)
        cropped_mask = crop_img(mask, bounding_box)

        # 3. Extract normalized patch from using cropped mask
        patch = extract_patch(normalized_image, cropped_mask)
        crops_size = min(patch.shape) - 10

        # 4. Resize patch and save image
        patches = [random_crop(patch, size=(crops_size, crops_size))
                   for i in range(3)]

        for idx, p in enumerate(patches):
            resized_patch = cv2.resize(
                p,
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )

            output_image_path = os.path.join(
                out_folder, '{}_{}'.format(row['abnormality type'], sev), "{}_{}.png".format(row.name, idx))
            cv2.imwrite(output_image_path, resized_patch)
    except Exception as e:
        print(f"Failed to process row {row['roi_mask_file_path']}: {e}")


def prepare_roi_severity_dataset(data_dir: str, out_dir: str, img_size: int):
    task = 'roi-severity'
    shutil.rmtree(os.path.join(out_dir, task), ignore_errors=True)
    for csv_data_file in glob(data_dir + '/*corrected.csv'):
        logging.info(f'Saving images defined in {csv_data_file}')
        data_type = 'train' if 'train' in csv_data_file else 'test'
        df = pd.read_csv(csv_data_file)
        cls = df['abnormality type'].iloc[0]
        pathologies = ['BENIGN', 'MALIGNANT']
        out_folder = os.path.join(out_dir, task,
                                  data_type)
        for i in pathologies:
            os.makedirs(os.path.join(out_folder, f'{cls}_{i}'), exist_ok=True)

        with tqdm(total=len(df)) as pbar:
            for _, row in df.iterrows():
                prepare_roi_severity_row(row, data_dir, out_folder, img_size)
                pbar.update()
