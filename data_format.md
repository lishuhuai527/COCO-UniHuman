### COCO-UniHuman Annotation File Format

COCO-UniHuman only keeps the images which have 'person' category annotations.

COCO-UniHuman annotation contains all the data of [COCO keypoint annotation](https://cocodataset.org/#format-data)
(including keypoints, num_keypoints, etc.) and additional fields.

Note that, we do not change the existing fields in the COCO keypoint dataset, such as 
"keypoints" and "num_keypoints". 

"keypoints" is a length 3*17 array (x, y, v) for `body` keypoints. 
Each keypoint has a 0-indexed location x,y and a visibility flag v defined as 
v=0: not labeled (in which case x=y=0), 
v=1: labeled but not visible, and 
v=2: labeled and visible. 
A keypoint is considered visible if it falls inside the object segment. 

"num_keypoints" indicates the number of labeled `body` keypoints (v>0), (e.g. crowds and small objects, will have num_keypoints=0). 

Additional fields include:

| key      | content                               | sample                                                                                                        |
|----------|---------------------------------------|---------------------------------------------------------------------------------------------------------------|
| valid    | a valid dict of the annotations       | {"labels": 1, "bboxes": 1, "keypoints_2d": 1, "segmentation": 1, "smpl": 0, "gender": 0, "age": 0, "face": 1} |
| gender   | manual annotations of gender          | 0                                                                                                             |
| age      | manual annotations of age             | 25                                                                                                            |
| smpl     | pose and betas parameters from EFT[1] | {"pose": [x * 72], "betas": [y * 10]}                                                                         |
| face_box | face_box from COCO-WholeBody[2]       | [x,y,w,h]                                                                                                     |

We provide manual annotations of gender (0 or 1) and age ([0, 85]).

Please note that invalid gender, age and smpl has no key-value in the annotation. We recommend using the valid field.

```
annotation{
"valid": {"labels": 1, "bboxes": 1, "keypoints_2d": 1, "segmentation": 1, "smpl": 1, "gender": 1, "age": 1} ,
"gender": bool,
"age": int,
"smpl": {"pose": list(float*72), "betas": list(float*10)},

"[cloned]": ...,
}

categories[{
"[cloned]": ...,
}]
```

### COCO-UniHuman V2 Annotation File Format
In short, 
COCO-UniHuman V2 = COCO-UniHuman V1 + COCO-WholeBody[2] + COCO-DensePose[3]

In order to be compatible with whole-body, we merge all data from whole-body and collect all valid flags into the valid dict. 
Then we add dense-pose annotations (matching the annotations of coco14 and coco17) and add dense_pose to the valid dict.

Finally, we got COCO-UniHuman V2 train annotations:

| key            | nums   | 
|----------------|--------|
| bbox_ignore    | 262465 |
| segmentation   | 262465 |
| num_keypoints  | 262465 |
| iscrowd        | 262465 |
| bbox           | 262465 |
| area           | 262465 |
| id             | 262465 |
| keypoints      | 262465 |
| category_id    | 262465 |
| image_id       | 262465 |
| valid          | 262465 |
| face_box       | 262465 |
| lefthand_box   | 262465 |
| righthand_box  | 262465 |
| lefthand_kpts  | 262465 |
| righthand_kpts | 262465 |
| face_kpts      | 262465 |
| foot_kpts      | 262465 |
| smpl           | 47775  |
| dense_pose     | 46507  |
| gender         | 52812  |
| age            | 53542  |

A sample of v2 valid dict is shown as follows:

```
"valid": {"labels": 1, "bboxes": 1, "keypoints_2d": 1, "segmentation": 0, "smpl": 0, "gender": 0, "age": 0, "face": 0, "lefthand": 1, "righthand": 1, "foot": 1, "dense_pose": 1}
```
Please note that invalid "gender", "age", "smpl" and "dense_pose" has no key-value in the annotation.

### Reference

```
[1] Joo, H., Neverova, N., Vedaldi, A.: Exemplar fine-tuning for 3d human model fitting towards in-the-wild 3d human pose estimation. In: Int. Conf. 3D Vis. pp. 42–52. IEEE (2021)
[2] Jin, S., Xu, L., Xu, J., Wang, C., Liu, W., Qian, C., Ouyang, W., Luo, P.: Whole-body human pose estimation in the wild. In: Eur. Conf. Comput. Vis. (2020)
[3] Alp Güler, R., Neverova, N., Kokkinos, I.: Densepose: Dense human pose estimation in the wild. In: IEEE Conf. Comput. Vis. Pattern Recog. (2018)

```