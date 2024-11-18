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

| Flag                  | Description                                                                                       | Default Value   |
|-----------------------|---------------------------------------------------------------------------------------------------|-----------------|
| --data_dir            | The folder where the CBIS-DDSM dataset is stored                                                  | None            |
| --out_dir             | The folder where the prepared dataset will be stored                                              | ./data          |
| --img_size            | The size to which the image should be resized                                                     | 128             |
| --task                | The task for which the dataset will be prepared ('lesion', 'mass-severity', 'calc-severity)       | 'lesion'        |


### 3.1. Dataset task

We implemented different ways to prepare the dataset depending on the targetted classification system development.

- ```lesion```: It's the original class split, it separates the dataset in two classes, namely "calc" and "mass"
- ```mass-severity```: This task separates the mass dataset into "benign" and "malignant" classes
- ```calc-severity```: This task separates the calc dataset into "benign" and "malignant" classes.

### 3.2. File structure

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