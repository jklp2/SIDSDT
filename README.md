##### Table of Content

1. [Introduction](#introduction)
1. [Datasets & checkpoints](#datasets-and-checkpoints)
1. [Getting Started](#getting-started)
	- [Requirements](#requirements)
	- [Usage Example](#usage)
1. [Generated images](#genereated-images-on-several-database)
# Introduction
This is the PyTorch implementation for our SPL paper:
***Single Image Dehazing via Semi-Supervised Domain Translation and Architecture Search***
We provide a semi-supervised dehazing method in this repo.
# Datasets and Checkpoints
### Datasets
We adopted the same training set as [DA_dehazing](https://github.com/HUSTSYJ/DA_dahazing). 
 - [DA_DATASETS](https://drive.google.com/drive/folders/10cP6Z-n2G0006_ppW1WxkQpNKg3mSfnj?usp=sharing).

We use four test sets to evaluate our method:
 - [SOTS-OD](https://drive.google.com/drive/folders/10cP6Z-n2G0006_ppW1WxkQpNKg3mSfnj?usp=sharing)
 - [O-HAZE](http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip)
 - [DENSE=HAZE](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/files/Dense_Haze_NTIRE19.zip)
 - [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip)
### Pretrained Checkpoint
 - Google drive: [Checkpoint](https://drive.google.com/file/d/1vHydxmBH8o5HuxVyG4ojRZWrr788zOp5/view?usp=sharing)

# Getting Started
### Requirements
1. Python 3.7
2. PyTorch 1.8.0
3. CUDA 9.1
4. Ubuntu 20.04

### Usage
Clone the repo
```
git clone https://github.com/jklp2/SID_Semi-Supervised_Domain_Translation.git
cd SID_Semi-Supervised_Domain_Translation
```
Download the pretrained [checkpoints](pretrained-checkpoint)
Put your test images in the input directory, and run:
```
python test.py --model cra_unrolled_final --resume --ckpt_path $CKPT_PATH
```
The results are in the output directory.
## Genereated images on several database
#### SOTS-OD
[Download link](https://drive.google.com/file/d/10EbzBsxML4DqvxapvhuXXOmLcA11pNRD/view?usp=sharing)
![image](imgs/sots.jpg)
#### O-HAZE
[Download link](https://drive.google.com/file/d/1n2VRc5iiYbPefthuM471X_psuhig4MD2/view?usp=sharing)
![image](imgs/o-haze.jpg)
#### DENSE-HAZE
[Download link](https://drive.google.com/file/d/1ISdv7ugn_b74zDqvO5mjTZSqytjwz3IV/view?usp=sharing)
![image](imgs/d-haze.jpg)
#### NH-HAZE
[Download link](https://drive.google.com/file/d/1xH-99_KfctaDDhV9ajZxiNl611BJMy6i/view?usp=sharing)
![image](imgs/nh-haze.jpg)
