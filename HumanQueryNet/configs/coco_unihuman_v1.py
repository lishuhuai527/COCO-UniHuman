# dataset settings
dataset_type = 'CocoUnihumanDataset'
data_prefix = ''
anno_prefix = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadMultitaskInstanceAnnotations', with_bbox=True, with_mask=True,
         extra_anno_list=['attributes', 'keypoints', 'areas', 'smpl_pose', 'smpl_betas', 'valids']),
    dict(type='KPRandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='KPResize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='KPResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='UniRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    crop_list=['gt_keypoints', 'gt_attributes', 'gt_areas', 'gt_labels', 'gt_masks'],
                    allow_negative_crop=True),
                dict(
                    type='KPResize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='MultiTaskFormatBundle',
         keys=['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_attributes', 'gt_keypoints', 'gt_areas',
               'gt_smpl_pose', 'gt_smpl_betas', 'gt_valids']),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_attributes', 'gt_keypoints', 'gt_areas', 'gt_masks', 'gt_smpl_pose',
               'gt_smpl_betas', 'gt_valids', ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=anno_prefix + 'coco_unihuman_train_v1.json',
        img_prefix=data_prefix + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=anno_prefix + 'coco_unihuman_val_v1.json',
        img_prefix=data_prefix + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=anno_prefix + 'coco_unihuman_val_v1.json',  # '2022_1024_coco_bbox_kp_gender_val.json',
        img_prefix=data_prefix + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
