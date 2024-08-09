_base_ = [
    'coco_unihuman_v1.py', 'default_runtime.py'
]

model = dict(
    type='HumanQueryNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    bbox_head=dict(
        type='HumanQueryHead',
        num_query=300,
        num_classes=1,
        with_kp=True,
        with_det=True,
        with_seg=True,
        with_attr=True,
        with_smpl=True,
        attr_cfg=[
            dict(name='gender', num_classes=1),
            dict(name='age', num_classes=85)
        ],
        transformer=dict(
            type='UniTransformer',
            strides=[8, 16, 32, 64],
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            num_keypoints=17,
            with_kpt_refine=True,
            encoder=dict(
                type='UnihumanTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=4,
                        dropout=0.0),  # 0.1 for DeformDETR
                    ffn_cfgs=dict(
                        type='FFN',
                        feedforward_channels=2048,  # 1024 for DeformDETR
                        num_fcs=2,
                        ffn_drop=0.0,  # 0.1 for DeformDETR
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='UniTransformerDecoder',
                num_layers=6,
                num_feat=128,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    feedforward_channels=2048,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        feedforward_channels=2048,  # 1024 for DeformDETR
                        num_fcs=2,
                        ffn_drop=0.0,  # 0.1 for DeformDETR
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            ),
            kpt_refine_decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            im2col_step=128)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            ),
            hm_encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))
            ),
        ),
        kp_cfg=dict(
            with_kpt_refine=True,
            num_kpt_fcs=2,
            num_keypoints=17,
            with_hm=True,
            as_two_stage=True,
            loss_kpt_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_kpt=dict(type='L1Loss', loss_weight=50.0),
            loss_kpt_rpn=dict(type='L1Loss', loss_weight=50.0),
            loss_oks=dict(type='OKSLoss', loss_weight=1.5),
            loss_hm=dict(type='CenterFocalLoss', loss_weight=4.0),
            loss_kpt_refine=dict(type='L1Loss', loss_weight=60.0),
            loss_oks_refine=dict(type='OKSLoss', loss_weight=2.0),
        ),
        seg_cfg=dict(
            feat_channels=256,
            out_channels=256,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=8.0),
            loss_dice=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                mask_downsample_ratio=4,
                sampler=dict(type='MaskPseudoSampler')),
        ),
        smpl_cfg=dict(
            loss_smpl_pose=dict(type='L1Loss', loss_weight=5.0),
            loss_smpl_betas=dict(type='L1Loss', loss_weight=10.0),
            smpl_beta_weights=[1, 0.64, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16],
            smpl_model_path=dict(
                n='models/smpl/models/SMPL_NEUTRAL.pth',
                m='models/smpl/models/SMPL_MALE.pth',
                f='models/smpl/models/SMPL_FEMALE.pth'
            ),
            # loss_pj2d=dict(type='L1Loss', loss_weight=10.0),
            # loss_gtpj2d=dict(type='L1Loss', loss_weight=50.0),
            loss_kp3d=dict(type='L1Loss', loss_weight=10.0),
            # pckh_filter=False,
            # pckh_thres=0.143,
            # pj2d_gt_trans=True,
            # pj_lsp=True,
            # cam_feature_fusion="pose",
            # min_2d_kp_num=6,
            # min_2d_kp_visible_num_3d=3,
            pose_prior_loss=dict(
                type='MaxMixturePrior',
                prior_folder='models/smpl/models/',
                num_gaussians=8,
                loss_weight=0.01,
                reduction='mean'),
        ),
        num_feature_levels=4,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_attrs=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                    dict(type='MeanVarianceSoftmaxLoss', weight_mean=0.02, weight_var=0.1, loss_weight=0.1)],
    ),
    # training and testing settings
    train_cfg=dict(
        uni_assigner=dict(
            type='UnihumanHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            kpt_cost=dict(type='KptL1Cost', weight=50.0),
            oks_cost=dict(type='OksV4Cost', weight=1.5),
            mask_cost=dict(type='CrossEntropyLossCost', weight=8.0, use_sigmoid=True),
            dice_cost=dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
            kpt_cls_cost=dict(type='FocalLossCostV2', weight=1.0),
        ),
    ),
    test_cfg=dict(max_per_img=100))  # 256 for dino, 100 for DeformDETR
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    # 'sampling_offsets': dict(lr_mult=0.1),
                                    # 'kp_reference_points': dict(lr_mult=0.1)
                                    }
                       )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=2)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
find_unused_parameters = True

# load_from = "./workdir/human_query_net_r50/epoch_100.pth"
# load_from = "./workdir/human_query_net_humanbenchL/epoch_95.pth"
# resume_from = "./workdir/human_query_net_r50/epoch_30.pth"


