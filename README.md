# Learning Heavily Degraed Piror

This repo is the official implementation of [Learning Heavily-Degraded Prior for Underwater Object Detection](). It is base on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Introduction
We propose a residual feature transference module (RFTM) to learn a mapping between deep representations of the heavily degraded patches of DFUI and underwater images, and make the mapping as a heavily degraded prior (HDP) for underwater detection. Since the statistical properties are independent to image content, HDP can be learned without the supervision of semantic labels and plugged into popular CNN-based feature extraction networks to improve their performance on underwater object detection. Without bells and whistles, evaluations on URPC2020 and UODD show that our methods outperform CNN-based detectors by a large margin. Our method with higher speeds and less parameters still performs better than transformer-based detectors.

## Results and Models

### URPC2020
| Methods | Backbone | Pretrain | AP | AP_50 | AP_75 | config | model |
| ------- | -------- | -------- | -- | ----- | ----- | ------ | ----- |
| RFTM-50 | ResNet50 | [cascade_rcnn_r50_dfui]() | 48.2 | 80.7 | 50.0 | [config](configs/rftm/rftm_50.py) | [baidu]()/[github]() |
| RFTM-x101 | ResNetXT101 | [cascade_rcnn_x101_dfui]() | 50.9 | 84.7 | 55.2 | [config](configs/rftm/rftm_x101.py) | [baidu]()/[github]()|

### UODD
| Methods | Backbone | Pretrain | AP | AP_50 | AP_75 | config | model |
| ------- | -------- | -------- | -- | ----- | ----- | ------ | ----- |
| RFTM-50 | ResNet50 | [cascade_rcnn_r50_dfui]() | 50.8 | 89.0 | 53.6 | [config](configs/rftm/rftm_50.py) | [baidu]()/[github]() |
| RFTM-x101 | ResNetXT101 | [cascade_rcnn_x101_dfui]() | 52.7 | 90.8 | 50.0 | [config](configs/rftm/rftm_x101.py) | [baidu]()/[github]() 

## Usage
### Installation
To install pytorch, run:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
To install mmdetection, run:
```
# install mmcv-full
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html

# install mmdet
pip install -r requirements/build.txt
python setup.py develop
```
To install guided-filter-pytorch, run:
```
pip install guided-filter-pytorch
```
### Inference
```
python tools/test.py <config_file> <checkpoint_file> --eval bbox
```
### Training
To train RFTM-50, run:
```
python tools/train.py configs/rftm/rftm_50.py --work-dir <work_dir>
```
To train RFTM-X101, run:
```
python tools/train.py configs/rftm/rftm_x101.py --work-dir <work_dir>
```