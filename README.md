# Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation (CVPR 2023)
by Yunhao Bai, Duowen Chen, Qingli Li, Wei Shen, and Yan Wang.
## Introduction
Official code for "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation". (CVPR 2023)
## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.
## Usage
We provide `code`, `data_split` and `models` for three datasets.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data), [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [Pancreas-CT](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT). 

To train a model,
```
python ./code/LA&ACDC/LA_BCP_train.py  #for LA training
python ./code/LA&ACDC/ACDC_BCP_train.py  #for ACDC training
python ./code/NIH_Pancreas/train_BCP_pancreas.py  #for Pancreas training
``` 

To test a model,
```
python ./code/LA&ACDC/test_LA.py  #for LA testing
python ./code/LA&ACDC/test_ACDC.py  #for ACDC testing
```
while for Pancreas testing, there is a individual function `test_model` in `train_BCP_pancreas.py`.

## Citation

## Acknowledgements
Our code is based on [SS-Net](https://github.com/ycwu1997/SS-Net) and [CoraNet](https://github.com/koncle/CoraNet). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions
If you have any questions, welcome contact me at '51215904056@stu.ecnu.edu.cn'



