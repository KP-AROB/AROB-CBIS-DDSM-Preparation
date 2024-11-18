from glob import glob
import logging
import os
import cv2
import shutil
from tqdm import tqdm
from src.utils.dicom import load_dicom_image
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def prepare_lesion_row(row, data_dir: str, out_folder: str, img_size: int):
    image_path = os.path.join(
        data_dir, row['image_file_path'])
    image = glob(image_path + '/*.dcm')[0]

    original_image = load_dicom_image(image)
    resized_image = cv2.resize(
        original_image,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )
    output_image_path = os.path.join(out_folder,
                                     "{}.png".format(row.name))  # Use row.name as idx
    cv2.imwrite(output_image_path, resized_image)


def prepare_lesion_dataset(data_dir: str, out_dir: str, img_size: int):
    """Prepare the CBIS dataset for lesion specific classification

    Args:
        data_dir (str): Path to original cbis dataset
        out_dir (str): Path to save the prepared cbis dataset
        img_size (int): New image size
    """
    task = 'lesion'
    shutil.rmtree(os.path.join(out_dir, task), ignore_errors=True)
    for csv_data_file in glob(data_dir + '/*corrected.csv'):
        logging.info(f'Saving images defined in {csv_data_file}')
        data_type = 'train' if 'train' in csv_data_file else 'test'
        df = pd.read_csv(csv_data_file)
        cls = df['abnormality type'].iloc[0]
        out_folder = os.path.join(out_dir, task,
                                  data_type, cls)
        os.makedirs(out_folder, exist_ok=True)

        with ProcessPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(
                        prepare_lesion_row,
                        [row for _, row in df.iterrows()],
                        [data_dir] * len(df),
                        [out_folder] * len(df),
                        [img_size] * len(df),
                    ),
                    total=len(df),
                )
            )
