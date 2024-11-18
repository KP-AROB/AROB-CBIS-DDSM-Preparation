import shutil
import os
import logging
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from src.utils.dicom import load_dicom_image, load_dicom_mask
from src.utils.crop import random_crop, extract_patch
from tqdm import tqdm


def prepare_roi_severity_dataset(data_dir: str, out_dir: str, img_size: int, random_patches: int = 3):
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
                sev = 'BENIGN' if row['pathology'] == 'BENIGN_WITHOUT_CALLBACK' else row['pathology']

                image_path = os.path.join(data_dir, row['image_file_path'])
                mask_path = os.path.join(data_dir, row['image_file_path'])

                image = load_dicom_image(glob(image_path + '/*.dcm')[0])
                mask = load_dicom_mask(glob(mask_path + '/*.dcm'), image.shape)

                patch = extract_patch(image, mask)

                crops_size = min(patch.shape) - 10
                patches = [random_crop(patch, size=(
                    crops_size, crops_size)) for i in range(random_patches)]

                for idx, p in enumerate(patches):
                    resized_patch = cv2.resize(
                        p,
                        (img_size, img_size),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    output_image_path = os.path.join(
                        out_folder, '{}_{}'.format(row['abnormality type'], sev), "{}_{}.png".format(row.name, idx))
                    print(resized_patch.shape)
                    cv2.imwrite(output_image_path, resized_patch)
                pbar.update()
