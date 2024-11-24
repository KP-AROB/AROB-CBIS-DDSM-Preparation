import argparse
import logging
import os
from src.utils.metadata import correct_metadata_files
from src.tasks.lesion import prepare_lesion_dataset, prepare_lesion_severity_dataset
from src.tasks.roi import prepare_roi_severity_dataset
from src.utils.print import read_poem
from src.utils.augmentations import make_augmentation
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBIS-DDSM data preparation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--synthetize", action='store_true')
    parser.add_argument("--patch_padding", type=int, default=100)
    parser.add_argument("--aug_ratio", type=int, default=8)
    parser.add_argument("--task", type=str, default='roi-severity', choices=['scan', 'scan-severity',
                                                                             'scan-mass-severity', 'scan-calc-severity',
                                                                             'roi-severity', 'roi-mass-severity',
                                                                             'roi-calc-severity'])
    args = parser.parse_args()
    parser.set_defaults(synthetize=False)

    logging_message = "[AROB-2025-KAPTIOS-CBISDDSM]"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    logging.info('Running CBIS-DDSM dataset preparation')
    logging.info(f'Creating dataset for {args.task} task')
    if len(glob(args.data_dir + '/*corrected.csv')) != 4:
        logging.info('Corrected csv files not found. Creating ...')
        correct_metadata_files(args.data_dir)
        logging.info(f'Corrected csv files saved at {args.data_dir}')
    os.makedirs(os.path.join(args.out_dir, args.task, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, args.task, 'test'), exist_ok=True)

    if args.task == 'scan':
        prepare_lesion_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, synthetize=args.synthetize)
    elif args.task == 'scan-severity':
        prepare_lesion_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, synthetize=args.synthetize)
    elif args.task == 'scan-mass-severity':
        prepare_lesion_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, 'mass', synthetize=args.synthetize)
    elif args.task == 'scan-calc-severity':
        prepare_lesion_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, 'calc', synthetize=args.synthetize)
    elif args.task == 'roi-severity':
        prepare_roi_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, patch_padding=args.patch_padding, synthetize=args.synthetize)
    elif args.task == 'roi-mass-severity':
        prepare_roi_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, roi_type='mass', patch_padding=args.patch_padding, synthetize=args.synthetize)
    elif args.task == 'roi-calc-severity':
        prepare_roi_severity_dataset(
            args.data_dir, args.out_dir, args.img_size, args.task, roi_type='calc', patch_padding=args.patch_padding, synthetize=args.synthetize)

    if args.aug_ratio > 0:
        synthetize_str = "_synthetized" if args.synthetize else ""
        task = f"{args.task}{synthetize_str}"
        make_augmentation(os.path.join(
            args.out_dir, args.task, 'train'), args.aug_ratio)

    logging.info('You made it. Have a piece of a french poem :')
    read_poem()
