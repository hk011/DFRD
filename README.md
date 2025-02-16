# DFRD: Dual Flow Reverse Distillation for Unsupervised Anomaly Detection

Official PyTorch implementation of DFRD

## Datasets
We use the MVTec AD dataset for experiments.

The data directory structure should be:

'''
data
└── mvtec
├── bottle
│ ├── ground_truth
│ ├── test
│ └── train
├── cable
│ ├── ground_truth
│ ├── test
│ └── train
...
└── zipper
├── ground_truth
├── test
└── train
'''


## Installation

### Requirements
- PyTorch 2.0.1
- CUDA 11.8+
- Other dependencies:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


## Testing

1. Download the pretrained weights

2. Run test_visaul.py:

Pretrained Checkpoints
Download pretrained checkpoints here and put the checkpoints under <project_dir>/checkpoints/.

Baidu Netdisk: 

Acknowledgement
We borrow some codes from RD4AD
