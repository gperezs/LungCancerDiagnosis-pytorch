# Automated Diagnosis of Lung Cancer with 3D Convolutional Neural Networks

Pytorch implementation for [Automated lung cancer diagnosis using three-dimensional convolutional neural networks](https://rdcu.be/b4Gc7). 
Our cancer predictor obtained a ROC AUC of 0.913 and was ranked 1st place at the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS).

## Table of contents
* [Getting started](#getting-started)
* [Datasets](#datasets)
* [Lung cancer diagnosis](#lung-cancer-diagnosis)

## Getting started

In this section we show how to setup the repository, install virtual environments (Virtualenv or Anaconda), and install requirements.

<details>
<summary>Click to expand</summary>

1. **Clone the repository:** To download this repository run:
```
$ git clone https://github.com/gperezs/LungCancerDiagnosis-pytorch.git
$ cd LungCancerDiagnosis-pytorch
```

In the following sections we show two ways to setup StarcNet. Use the one that suits you best:
* [Using virtualenv](#using-virtualenv)
* [Using Anaconda](#using-anaconda)

### Using virtualenv

2. **Install virtualenv:** To install virtualenv run after installing pip:

```
$ sudo pip3 install virtualenv
```

3. **Virtualenv  environment:** To set up and activate the virtual environment,
run:
```
$ virtualenv -p /usr/bin/python3 venv3
$ source venv3/bin/activate
```

To install requirements, run:
```
$ pip install -r requirements.txt
```

To install dicom library run:
```
$ pip install dicom
```

4. **PyTorch:** To install pytorch run:
```
$ pip install torch torchvision
```

-------
### Using Anaconda

2. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
$ conda create -n <env name> python=3.*
$ conda activate <env name>
```

To install requirements, run:
```
$ conda install --yes --file requirements.txt
```

To install dicom library run:
```
$ pip install dicom
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).
</details>

## Trained models

Download the trained models from this [link](https://www.dropbox.com/s/to7pmlajtr0tyos/models.zip?dl=0). Detector model was trained with the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) and the predictor with the [Kaggle DSB2017 dataset](https://www.kaggle.com/c/data-science-bowl-2017).

## Lung cancer diagnosis

To run the code save the folder of each patient with the dicom files in the folder `data/ISBI-deid-TRAIN/`. Then, sun:
```
python test.py
```

The program will print a single lung cancer probability per subject.

### Run with ISBI 2018 lung challenge subjects

To run the code save the folder of each patient with the dicom files (of the ISBI 2018 Lung challenge) in the folder `data/ISBI-deid-TRAIN/`. Then, run:
```
python test_ISBI.py
```
If the dataset from the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS) is used, the AUC will be printed using the challenge released labels (including the mask post-processing). 


In folder `data/sorted_slices_jpgs/` the program will save images of the axial, sagittal and coronal planes of the 30 detected nodules with highest score of each patient.


