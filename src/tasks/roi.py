import shutil
import os
import logging
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob
from src.utils.dicom import load_dicom_image, load_dicom_mask
from src.utils.crop import random_crop, extract_patch
from concurrent.futures import ProcessPoolExecutor


def prepare_roi_severity_row(row, data_dir: str, out_folder: str, img_size: int):
    # some cropped_image_file_path are equal to roi_image_path
    # here we check if a cropped image folder is available to load the image mask
    mask_file_path = row['roi_mask_file_path'] if 'cropped' not in row['cropped_image_file_path'] else row['cropped_image_file_path']
    try:
        sev = 'BENIGN' if row['pathology'] == 'BENIGN_WITHOUT_CALLBACK' else row['pathology']
        image_path = os.path.join(data_dir, row['image_file_path'])
        image = load_dicom_image(glob(image_path + '/*.dcm')[0])
        mask_file_path = row['roi_mask_file_path'] if 'cropped' not in row[
            'cropped_image_file_path'] else row['cropped_image_file_path']
        mask_path = glob(os.path.join(data_dir, mask_file_path, '*.dcm'))
        mask = load_dicom_mask(mask_path, image.shape)
        # 3. Extract normalized patch from using cropped mask.
        # If mask.shape is not equal to image shape, that means a patch was returned as a mask
        if mask.shape != image.shape:
            patch = mask
        else:
            patch = extract_patch(image, mask)

        crops_size = min(patch.shape) - 20
        patches = [random_crop(patch, size=(crops_size, crops_size))
                   for _ in range(3)]

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


def prepare_roi_severity_dataset(data_dir: str, out_dir: str, img_size: int, task: str, roi_type: str = None):
    shutil.rmtree(os.path.join(out_dir, task), ignore_errors=True)
    csv_file_list = glob(data_dir + '/*corrected.csv') if not roi_type else glob(
        data_dir + f'/*{roi_type}*corrected.csv')
    for csv_data_file in csv_file_list:
        logging.info(f'Saving images defined in {csv_data_file}')
        data_type = 'train' if 'train' in csv_data_file else 'test'
        df = pd.read_csv(csv_data_file)
        cls = df['abnormality type'].iloc[0]
        pathologies = ['BENIGN', 'MALIGNANT']
        out_folder = os.path.join(out_dir, task,
                                  data_type)
        for i in pathologies:
            os.makedirs(os.path.join(out_folder, f'{cls}_{i}'), exist_ok=True)

        with ProcessPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(
                        prepare_roi_severity_row,
                        [row for _, row in df.iterrows()],
                        [data_dir] * len(df),
                        [out_folder] * len(df),
                        [img_size] * len(df),
                    ),
                    total=len(df),
                )
            )
