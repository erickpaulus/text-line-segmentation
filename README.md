The code in this repository was utilized for the publication mentioned below, but [following publication is still under-reviewed](). 

```
@article{paulus2023,
title = "Text Line Extraction Strategy for Palm Leaf Manuscripts",
author = "",
journal = "",
volume = "",
pages = "",
year = "2023",
issn = "",
doi = ""
}
```

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Acknowledgment](#acknowledgment)


## Installation
- create a virtual environtment 
- pip install requirements.txt

## Usage
Below are guidelines for extracting text lines and reproducing the experiments.

The `run_LS_singleImage.py`  script performs the text line segmentation of an input image using a trained model. Some parameters of this script are the following:


| Parameter    | Default | Description                      |
| ------------ | ------- | -------------------------------- |
| `-fname`     |         | Filename                         |
| `-path`   |         | Path folder containe image      |
| `-smooth` |  40       | Smoothing for projection profile matching     |
| `-s`         |  6     | Number of image slices for projection profile matching |
| `-sigma` |  3       | Standard deviation for gaussian smoothing     |
| `-modelpath` |  (*)       | Path to the model to load     |
| `-w`         |  320    | Input window size                |
| `-step`      |  128     | Step size  |
| `-f`         |  16     | Number of filters                |
| `-k`         |  5      | Kernel size                      |
| `-drop`      |  0.2      | Dropout percentage               |


For example, to segment the image `CB-3-22-90-23.tif` which located in default folder path 'datasets/Exp1-mixedPLM-GR' you can run the following command:

```
$ python run_LS_singleImage.py -fname CB-3-22-90-23.tif 
```

if  you have different folder path, please set the path parameter as well.

```
$ python run_LS_singleImage.py - path <your folder path> -fname 033_pp_ttp_01_001.jpg 
```

If you want to segment all images within folder, you have to run the following command:

```
$ python run_LS.py 
```

or if you have set a certain folder

```
$ python run_LS.py -path datasets/Exp1-mixedPLM-GR 
```

The `MODELS` folder consists of  several trained models for the palm leaf manuscripts datasets. This following model is set as default :

* `MODELS/model_weights_DC5_palm_6_nb15_320x320_s128_drop0.2_f16_k5_s2_se1_e200_b10_esp.h5`

please download the models in this [url](https://drive.google.com/drive/folders/1n8qx38BMhxSrgfkTJd1tfJsTJBT3UhdR?usp=sharing)



The folders of  dataset must have a specific name. Each collection there must be two folders, one with the suffix `-GR` with the input images in grayscale or RGB, and others with the suffix `-GT` for the ground truth.



## Datasets

Below is a summarized table of the datasets utilized in this project: 

 
| No | Collections | Script        | Pages | Lines | Source|
| -- | ----------- | ------------- | ----- |------ | ------|
| 1  | 12 SPLM     | Old Sundanese | 12  | 46  | Trining,Testing |
| 2  | 61 SPLM     | Old Sundanese | 61  | 242 | Trining,Testing |
| 3  | 30 SPLM     | Old Sundanese | 30 | 118 | Testing    |
| 4  | 49 BPLM     | Old Balinese  | 49 | 182 | Testing    |
| 5  | 200 KPLM    | Old Khmer     | 200 | 970 | Testing   |
| 6  | 30 PLM      | Mix          | 30 | 115 | Testing-Mix    |

Training : http://amadi.univ-lr.fr/ICFHR2018_Contest/images/images/Train-ChallengeB-TrackMixed.zip

Testing : 
- http://amadi.univ-lr.fr/ICFHR2018_Contest/images/images/Test-ChallengeB-TrackMixed.zip

- http://amadi.univ-lr.fr/ICFHR2018_Contest/images/images/GT-ChallengeB-TrackMixed.zip

If you need the complete mixed collection, please contact us to get the data.

It consists of 
- Exp1-mixedPLM-GR -> put this folder in datasets folder for extracting text lines
- Exp1-mixedPLM-GT -> put this folder in datasets folder for extracting text lines in Dat format. It also uses in evaluation metrics and put all images in <path>\images in evaluator tool path
- Exp1-mixedPLM-GT-Dat -> -> put all images in <path>\gt\lines in evaluator tool path

## Evaluation-Metrics
For evaluation metrics, we adapted the criteria used in the
ICDAR2013 Handwriting Segmentation Competition and occupied their evaluator tool, which can be downloaded in https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/resources.html

After installing the evaluator tool, it saves defaultly in C:\Program Files (x86)\ICDAR2013HandSegmCont
- gt/lines : consist of ground truth image in Dat file
- images : consist of ground truth image in bmp file
- results/lines : consist of resulted image in Dat file

## Acknowledgment
Thanks to Gallego for sharing the code for binarization process and some codes are adopted from https://github.com/ajgallego/document-image-binarization

