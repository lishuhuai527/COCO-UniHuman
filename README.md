# COCO-UniHuman
This is the official repo for ECCV2024 paper ["You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception"](https://arxiv.org/abs/2312.05525). 

The repo contains COCO-UniHuman annotations and HumanQueryNet proposed in this paper. 

## News
*2024/08/09: code and model weight of HumanQueryNet released!*

*2024/07/09: COCO_UniHuman dataset released!*


## COCO-UniHuman Dataset
Please refer to the introduction of dataset [COCO_UniHuman](COCO_UniHuman.md).

## HumanQueryNet

### Environment Setup

```shell
conda create -n HQN python==3.9

conda activate HQN

pip install -r requirements.txt
```



### Training

1. Download COCO'17 images and COCO-UniHuman v1 annotations, add data_prefix and anno_prefix to the data config file configs/coco_unihuman_v1.py

2. Download the converted SMPL models from [download link](https://drive.google.com/drive/folders/1SWPPPlgOo3mNmLgMkXL4ukpGPmUGr9Nk) and put all files in HumanQueryNet/models/smpl/models:
```shell
HumanQueryNet/models/smpl/models/
├── gmm_08.pkl
├── SMPL_FEMALE.pth
├── SMPL_MALE.pth
└── SMPL_NEUTRAL.pth
```

3. Then modify train.sh to train the model (Please refer to mmdet-2.5.3 training scripts).


### Testing

Our r50 model can be downloaded [here](https://drive.google.com/drive/folders/1SWPPPlgOo3mNmLgMkXL4ukpGPmUGr9Nk).

Please refer to test.sh to test the model on all HCP tasks.


## License
Codes and data are freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please contact Mr. Sheng Jin (jinsheng13[at]foxmail[dot]com). We will send the detail agreement to you.

## Citation
if you find our paper and code useful in your research, please consider giving a star and citation:

```bibtex
@inproceedings{jin2023you,
  title={You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception},
  author={Jin, Sheng and Li, Shuhuai and Li, Tong and Liu, Wentao and Qian, Chen and Luo, Ping},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  month={September}
}
```