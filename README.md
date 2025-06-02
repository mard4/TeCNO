

--------- this works
conda create -n tecno_py38 python=3.8
conda activate tecno_py38
pip install pytorch-lightning==0.8.5
pip install pandas==1.1.4 scikit-learn==0.23.2 opencv-python==4.2.0.34 ConfigArgParse==1.5.1 torchsummary==1.5.1 pycm==3.0 albumentations==0.5.0 scikit-video torchvision wandb torchmetrics

PRIOR TO FEATURE EXTRACTOR:
- python3 utils/tecno/split_vid.py
- python3 utils/tecno/create_dataframe.py

FEATURE EXTRACTOR
- python train.py -c modules/cnn/config/config_feature_extract.yml

TRAIN
- python train.py -c modules/mstcn/config/config_tcn.yml

RUN TESTING SEPARATELY
- python train.py -c modules/cnn/config/config_feature_extract.yml --resume_from_checkpoint=path/to/checkpoint.ckpt


cose da fare:
- Make Loss Optional
     - In add_module_specific_args() (TeCNO-specific CLI):
        
       Add optional flags:
- Comprare
    - Validation accuracy
    - Per-phase F1
    - Inference smoothness (e.g. fewer erratic transitions)




logs:
feature extractor: https://wandb.ai/martidan182-university-of-twente/tecno/runs/ckb555tr/logs
NOTA: recall to reference the correct model










-------------------------------







# Table of Contents
<img src="assets/tecno_logo.png"
     alt="logo tecno"
     width=150px
     style="margin-right:50px"
     align="right" />
     
1. [**About**](#about)
2. [**Getting started**](#getting-started)
     1. [**Prerequisites**](#prerequisites)
     2. [**Usage**](#usage)
3. [**Reference**](#reference)
4. [**Contact**](#contact)


# About
TeCNO performs hierarchical prediction refinement with causal, dilated convolutions for surgical phase recognition and outperforms various state-of-the-art LSTM approaches!
     
Link to paper: [**TeCNO Paper**](https://arxiv.org/abs/2003.10751)


<p align="center">
     <img src="assets/abstract_tecno.png"
          alt="logo tecno"
          width=1000px />
</p>

# Getting started
Follow these steps to get the code running on your local machine!

## Prerequisites

```
pip install -r requirements.txt
```

## Usage

We are using the publicly available [Cholec80 dataset](http://camma.u-strasbg.fr/datasets). For training we [split](utils/tecno/split_vid.py) the videos into individual frames.

### Stage 1 - Train Feature Extractor

Run:
```
python train.py -c modules/cnn/config/config_feature_extract.yml

```
This will train your feature extractor and in the *Test Step* it will extract for each Video the features of all images and save it as *.pkl*

### Stage 2 - Train Temporal Convolutional Network

```
python train.py -c modules/mstcn/config/config_tcn.yml
```


# Reference 

```
@inproceedings{czempiel2020,
 author    = {Tobias Czempiel and
               Magdalini Paschali and
               Matthias Keicher and
               Walter Simson and
               Hubertus Feussner and
               Seong Tae Kim and
               Nassir Navab},
 title     = {TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional
               Networks},
  booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI}
               2020 - 23nd International Conference, Shenzhen, China, October 4-8,
               2020, Proceedings, Part {III}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12263},
  pages     = {343--352},
  publisher = {Springer},
  year      = {2020},
}
```



# Contact

For any problems and question please open an [Issue](https://github.com/tobiascz/TeCNO/issues/new/choose)


[1.2]: http://i.imgur.com/wWzX9uB.png (twitter icon without padding)
[1]: http://www.twitter.com/tobiasczempiel
Follow me on [![alt text][1.2]][1]
