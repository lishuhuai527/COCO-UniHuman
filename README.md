# COCO-UniHuman

This is the official repo for ECCV2024 paper ["You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception"](https://arxiv.org/abs/2312.05525). 

The repo contains COCO-UniHuman annotations proposed in this paper. 

## What is COCO-UniHuman? 

We introduce two versions of COCO-UniHuman datasets.

COCO-UniHuman v1 is the first large-scale dataset which provides annotations for human-centric perception tasks in multi-person scenarios. 
The annotations include bounding boxes, keypoints, segmetation masks, smpl parameters, human attributes (age and gender).
It is an extension of [COCO 2017 dataset](https://cocodataset.org/#keypoints-2017) with the same train/val split as COCO'17.

COCO-UniHuman v2 is an extension of COCO-UniHuman v1, which incorportates COCO-UniHuman v1, COCO-WholeBody[16] and COCO-DensePose[17], encouraing further research on multi-task human-centric perception.

## How to Use?

### Download
Images can be downloaded from [COCO 2017 website](https://cocodataset.org/#keypoints-2017).

COCO-UniHuman annotations for train/val can be downloaded from [download link](https://drive.google.com/drive/folders/1PIPYHRuV_TnERQkOdz4aijBZL_ir_zKT) (GoogleDrive).

Alternatively, we also provide the BaiduPan download link for the annotation files.

BaiduPan Link: https://pan.baidu.com/s/11PP70mlE03G6xoon6L7U4A    

Password: a8wq


### Annotation Format
The data format is defined in [DATA_FORMAT](data_format.md).


### Terms of Use

1. COCO-UniHuman dataset is **ONLY** available for research and non-commercial use. The annotations of COCO-UniHuman dataset belong to [SenseTime Research](https://www.sensetime.com), and are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

2. For commercial usage of our COCO-UniHuman annotations, please contact Mr. Sheng Jin (jinsheng13[at]foxmail[dot]com). We will send the detail agreement to you.

3. We do not own the copyright of the images. Use of the images must abide by the [Flickr Terms of Use](https://www.flickr.com/creativecommons/). The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset.

## Compare with other popular datasets.

Overview of representative HCP datasets. “Images”, “Instances”, and “IDs” mean the number of total images, instances and identities respectively. 
“Crop” indicates whether the images are cropped for “face” or “body”. * means head box annotation. “group:n” means age classification with n groups, “real” means real age estimation, 
and “appa” means apparent age estimation.

| DataSet             | Images    | Instances | IDs  | Crop | BodyBox | FaceBox | BodyKpt | BodyMask | Gender | Age         | Mesh |
|---------------------|-----------|-----------|------|------|---------|---------|---------|----------|--------|-------------|------|
| Caltech [1]         | 250K      | 350K      | 2.3K | ✗    | ✔️      | ✗       | ✗       | ✗        | ✗      | ✗           | ✗    | 
| CityPersons [2]     | 5K        | 32K       | 32K  | ✗    | ✔️      | ✗       | ✗       | ✗        | ✗      | ✗           | ✗    |                  
| CrowdHuman [3]      | 24K       | 552K      | 552K | ✗    | ✔️      | *       | ✗       | ✗        | ✗      | ✗           | ✗    |              
| MPII [4]            | 25K       | 40K       | -    | ✗    | ✔️      | *       | ✔️      | ✗        | ✗      | ✗           | ✗    |                
| PoseTrack [5]       | 23K       | 153K      | -    | ✗    | ✔️      | ️ *     | ✔️️     | ✗        | ✗      | ✗           | ✗    |               
| CIHP [6]            | 38K       | 129K      | 129K | ✗    | ✔️      | ️ ✗     | ✗       | ✔️       | ️ ✗    | ✗           | ✗    |             
| MHP [7]             | 5K        | 15K       | 15K  | ✗    | ✔️      | ✗       | ✗       | ✔️       | ✗      | ✗           | ✗    |              
| CelebA [8]          | 200K      | 200K      | 10K  | face | ✗       | ✗       | ✗       | ✗        | ✔️     | group:4     | ✗    |
| APPA-REAL [9]       | 7.5K      | 7.5K      | 7.5K | face | ✗       | ✗       | ✗       | ✗        | ✔️     | appa & real | ✗    | 
| MegaAge [10]        | 40K       | 40K       | 40K  | face | ✗       | ✗       | ✗       | ✗        | ✔️     | real        | ✗    |   
| WIDER-Attr [11]     | 13K       | 57K       | 57K  | ✗    | ✔️      | ✗       | ✗       | ✗        | ✔️     | group:6     | ✗    |    
| PETA [12]           | 19K       | 19K       | 8.7K | body | ✗       | ✗       | ✗       | ✗        | ✔️     | group:4     | ✗    |     
| PA-100K [13]        | 100K      | 100K      | -    | body | ✗       | ✗       | ✗       | ✗        | ✔️     | group:3     | ✗    |       
| OCHuman [14]        | 5K        | 13K       | 13K  | ✗    | ✔️      | ✗       | ✔️      | ✔️       | ✗      | ✗           | ✗    |             
| COCO [15]           | 200K      | 273K      | 273K | ✗    | ✔️      | ✗       | ✔️      | ✔️       | ✗      | ✗           | ✗    |            
| COCO-WholeBody [16] | 200K      | 273K      | 273K | ✗    | ✔️      | ✔️      | ✔️      | ✗        | ✗      | ✗           | ✗    |            
| COCO-UniHuman v1    | 200K(64k) | 273K      | 273K | ✗    | ✔️      | ✔️      | ✔️      | ✔️       | ✔️     | appa        | ✔️   |      

## Citation

If you use this dataset in your project, please cite these papers.

```
@article{jin2023you,
  title={You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception},
  author={Jin, Sheng and Li, Shuhuai and Li, Tong and Liu, Wentao and Qian, Chen and Luo, Ping},
  journal={arXiv preprint arXiv:2312.05525},
  year={2023}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@inproceedings{joo2021exemplar,
  title={Exemplar fine-tuning for 3d human model fitting towards in-the-wild 3d human pose estimation},
  author={Joo, Hanbyul and Neverova, Natalia and Vedaldi, Andrea},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={42--52},
  year={2021},
  organization={IEEE}
}
```
If you use the v2 version dataset in your project, please cite these additional papers.
```
@inproceedings{jin2020whole,
  title={Whole-body human pose estimation in the wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part IX 16},
  pages={196--214},
  year={2020},
  organization={Springer}
}

@inproceedings{guler2018densepose,
  title={Densepose: Dense human pose estimation in the wild},
  author={G{\"u}ler, R{\i}za Alp and Neverova, Natalia and Kokkinos, Iasonas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7297--7306},
  year={2018}
}

```

## Reference

```
[1] Dollár, P., Wojek, C., Schiele, B., Perona, P.: Pedestrian detection: A benchmark. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 304–311 (2009)
[2] Zhang, S., Benenson, R., Schiele, B.: Citypersons: A diverse dataset for pedestrian detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 3213–3221 (2017)
[3] Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., Sun, J.: Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123 (2018)
[4] Andriluka, M., Pishchulin, L., Gehler, P., Schiele, B.: 2d human pose estimation: New benchmark and state of the art analysis. In: IEEE Conf. Comput. Vis. Pattern Recog. (2014)
[5] Andriluka, M., Iqbal, U., Insafutdinov, E., Pishchulin, L., Milan, A., Gall, J., Schiele, B.: Posetrack: A benchmark for human pose estimation and tracking. In: IEEE Conf. Comput. Vis. Pattern Recog. (2018)
[6] Gong, K., Liang, X., Li, Y., Chen, Y., Yang, M., Lin, L.: Instance-level human parsing via part grouping network. In: Eur. Conf. Comput. Vis. pp. 770–785 (2018)
[7] Li, J., Zhao, J., Wei, Y., Lang, C., Li, Y., Sim, T., Yan, S., Feng, J.: Multiple-human parsing in the wild. arXiv preprint arXiv:1705.07206 (2017)
[8] Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: Int. Conf. Comput. Vis. (2015)
[9] gustsson, E., Timofte, R., Escalera, S., Baro, X., Guyon, I., Rothe, R.: Apparent and real age estimation in still images with deep residual regressors on appa-real database. In: IEEE Int. Conf. Auto. Face & Gesture Recog. pp. 87–94 (2017)
[10] Zhang, Y., Liu, L., Li, C., Loy, C.C.: Quantifying facial age by posterior of age comparisons. In: Brit. Mach. Vis. Conf. (2017)
[11] Li, Y., Huang, C., Loy, C.C., Tang, X.: Human attribute recognition by deep hierarchical contexts. In: Eur. Conf. Comput. Vis. (2016)
[12] Deng, Y., Luo, P., Loy, C.C., Tang, X.: Pedestrian attribute recognition at far distance. In: ACM Int. Conf. Multimedia. pp. 789–792 (2014)
[13] Liu, X., Zhao, H., Tian, M., Sheng, L., Shao, J., Yan, J., Wang, X.: Hydraplus-net: Attentive deep features for pedestrian analysis. In: Int. Conf. Comput. Vis. pp. 1–9 (2017)
[14] Zhang, S.H., Li, R., Dong, X., Rosin, P., Cai, Z., Han, X., Yang, D., Huang, H., Hu, S.M.: Pose2seg: Detection free human instance segmentation. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 889–898 (2019)
[15] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Eur. Conf. Comput. Vis. (2014)
[16] Jin, S., Xu, L., Xu, J., Wang, C., Liu, W., Qian, C., Ouyang, W., Luo, P.: Whole-body human pose estimation in the wild. In: Eur. Conf. Comput. Vis. (2020)
[17] Alp Güler, R., Neverova, N., Kokkinos, I.: Densepose: Dense human pose estimation in the wild. In: IEEE Conf. Comput. Vis. Pattern Recog. (2018)
```
