import os
import pandas as pd
from tqdm import tqdm


def normalize_and_format_path(path: str) -> str:
    if path.startswith(".\\"):
        path = path[2:]
    path = path.replace("\\", "/")
    path_parts = path.split("/")
    if path_parts:
        last_part = path_parts[-1]
        number, *rest = last_part.split("-", 1)
        if number.isdigit():
            number = number.zfill(2)
        path_parts[-1] = f"{number}-{''.join(rest)}"
    return "/".join(path_parts)


def get_image_path_ids(row, key):
    path = row[key]
    path_segment = path.split(os.sep)
    study_id = path_segment[1]
    series_uid = path_segment[2]
    return study_id, series_uid


def correct_metadata_files(data_dir: str):

    metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

    lesion_description_files = {
        f"{desc}_case_description_{set_type}_set": os.path.join(data_dir, f"{desc}_case_description_{set_type}_set.csv")
        for desc in ["mass", "calc"]
        for set_type in ["train", "test"]
    }

    with tqdm(total=len(lesion_description_files.keys()), desc='Correcting csv files') as pbar:
        for key in lesion_description_files.keys():
            df = pd.read_csv(lesion_description_files[key])
            df = df.rename(columns={
                'left or right breast': 'left_or_right_breast',
                'image view': 'image_view',
                'abnormality id': 'abnormality_id',
                'mass shape': 'mass_shape',
                'mass margins': 'mass_margins',
                'image file path': 'image_file_path',
                'cropped image file path': 'cropped_image_file_path',
                'ROI mask file path': 'roi_mask_file_path'})

            for idx, row in df.iterrows():
                image_study_id, image_series_uid = get_image_path_ids(
                    row, 'image_file_path')
                roi_study_id, roi_series_uid = get_image_path_ids(
                    row, 'roi_mask_file_path')
                cropped_study_id, cropped_series_uid = get_image_path_ids(
                    row, 'cropped_image_file_path')

                meta_image = metadata_df[(metadata_df['Series UID'] == image_series_uid) & (
                    metadata_df['Study UID'] == image_study_id)]
                meta_roi = metadata_df[(metadata_df['Series UID'] == roi_series_uid) & (
                    metadata_df['Study UID'] == roi_study_id)]
                meta_cropped = metadata_df[(metadata_df['Series UID'] == cropped_series_uid) & (
                    metadata_df['Study UID'] == cropped_study_id)]

                correct_img_path = meta_image['File Location'].values[0]
                correct_roi_path = meta_roi['File Location'].values[0]
                correct_cropped_path = meta_cropped['File Location'].values[0]

                df.loc[idx, 'image_file_path'] = normalize_and_format_path(
                    correct_img_path)
                df.loc[idx, 'roi_mask_file_path'] = normalize_and_format_path(
                    correct_roi_path)
                df.loc[idx, 'cropped_image_file_path'] = normalize_and_format_path(
                    correct_cropped_path)

            df.to_csv(os.path.join(data_dir, key + '_corrected.csv'))
            pbar.update()
