import argparse
import logging
import os
import shutil
from src.utils.metadata import correct_metadata_files
from src.utils.dicom import load_dicom_image
from src.tasks.lesion import prepare_lesion_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBIS-DDSM data preparation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--task", type=str, default='lesion',
                        choices=['lesion', 'mass-severity', 'calc-severity'])
    args = parser.parse_args()

    logging_message = "[AROB-2025-KAPTIOS-CBISDDSM]"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    logging.info('Running CBIS-DDSM dataset preparation')
    logging.info(f'Creating dataset for {args.task} task')
    correct_metadata_files(args.data_dir)
    logging.info(f'Corrected csv files saved at {args.data_dir}')
    os.makedirs(os.path.join(args.out_dir, args.task, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, args.task, 'test'), exist_ok=True)

    if args.task == 'lesion':
        prepare_lesion_dataset(args.data_dir, args.out_dir, args.img_size)

    logging.info('Preparation done !')
    logging.info('Exiting. May the force be with you')
