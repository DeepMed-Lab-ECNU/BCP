# Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation (CVPR 2023)
by Yunhao Bai, Duowen Chen, Qingli Li, Wei Shen, and Yan Wang.
## Introduction
Official code for "[Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2305.00673)". (CVPR 2023)
## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.
## Usage
We provide `code`, `data_split` and `models` for LA and ACDC dataset.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

To train a model,
```
python ./code/LA_BCP_train.py  #for LA training
python ./code/ACDC_BCP_train.py  #for ACDC training
``` 

To test a model,
```
python ./code/test_LA.py  #for LA testing
python ./code/test_ACDC.py  #for ACDC testing
```

## Citation

## Acknowledgements
Our code is largely based on [SS-Net](https://github.com/ycwu1997/SS-Net). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at 'yhbai@stu.ecnu.edu.cn'



