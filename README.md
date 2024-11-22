# CBIS-DDSM dataset preparation

This repo aims to prepare the CBIS-DDSM dataset for breast anomaly classification tasks.

## 1. Requirements

The pip dependencies for this project can also be downloaded again using :

```bash
pip install -r requirements.txt
```

## 2. Installation

First, you need to download the dataset. It is available on the cancer imaging archive website : [CBIS-DDSM dataset](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
Then just clone this repository.

## 3. Usage

You can run the preparation script with the following command and given flags : 

```bash
python run.py --data_dir ./cbis_ddsm --out_dir ./data
```

| Flag                  | Description                                                                                                       | Default Value   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|-----------------|
| --data_dir            | The folder where the CBIS-DDSM dataset is stored                                                                  | None            |
| --out_dir             | The folder where the prepared dataset will be stored                                                              | ./data          |
| --img_size            | The size to which the image should be resized                                                                     | 256             |
| --aug_ratio           | The number of new images to create with augmentations                                                             | 0               |
| --task                | The task for which the dataset will be prepared                                                                   | 'roi-severity'  |
| --patch_padding       | The size of the padding around the ROI patches                                                                    | 100             |


### 3.1. Dataset task

We implemented different ways to prepare the dataset depending on the targetted classification system development. The tasks with keyword "scan" will prepare a whole breast image dataset, while the "roi" keyword is used to only extract roi patches.

- ```scan```: It's the original class split, it separates the image dataset in two classes, namely "calc" and "mass"
- ```scan-severity```: This task separates both calc and mass datasets into "benign" and "malignant" classes leading to 4 different classes.
- ```scan-mass-severity```: This task separates the mass scan datasets into "benign" and "malignant" classes.
- ```scan-calc-severity```: This task separates the cacl scan datasets into "benign" and "malignant" classes.
- ```roi-severity```: This task separates both calc and mass roi datasets into "benign" and "malignant" classes leading to 4 different classes.
- ```roi-mass-severity```: This task separates mass roi datasets into "benign" and "malignant" classes.
- ```roi-calc-severity```: This task separates calc roi datasets into "benign" and "malignant" classes.

### 3.2. Preprocessing

For each task the images are loaded and normalized using the truncated normalization method.
This step is done by first cropping the image to the breast region through the Otsu threshold method.

### 3.3. File structure

For each task the script will create training and testing sets based on the original dataset split. For a given task the file structure will then look like :

- 📂 data/
    - 📂 task_name/
        - 📂 train/
            - 📂 calc/
                - 📄 01.png
                - 📄 02.png
            - 📂 mass/
                - 📄 01.png
                - 📄 02.png
        - 📂 test/
            - 📂 mass/
                - 📄 01.png
                - 📄 02.png
            - 📂 calc/
                - 📄 01.png
                - 📄 02.png


### 3.4. Data augmentation

The dataset can be augmented during the preparation process following a pre-defined pipeline, the augmentation can be called with the ```--aug_ratio``` flag.
This ratio controls the amount of new images (per scan) that will be created. By default this flag has a value of 0 meaning that the dataset will not be augmented.

This augmentation process uses the albumentations library and the augmentation pipeline follows the following code : 

```python

transform = A.Compose([
    A.HorizontalFlip(p=0.5),    
    A.VerticalFlip(p=0.5),    
    A.ElasticTransform(p=0.2),
    A.CLAHE(p=0.5),
])

```

## 4. Data Statistics

### 4.1. roi-severity task

- Mean : 0.5451
- Standard Deviation : 0.1577

### 4.2. Synthetized roi-mass-severity task

- Mean : 0.5344
- Standard Deviation : 0.1731