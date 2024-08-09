# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, build_activation_layer, bias_init_with_prob, constant_init, normal_init)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops import point_sample
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.core.mask import mask2bbox
# from ..builder import HEADS, build_loss
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer, get_uncertain_point_coords_with_randomness

from core.keypoint import gaussian_radius, draw_umich_gaussian
from models.smpl import SMPLWrapper
from models.smpl.smpl_wrapper import rot6D_to_angular
from models.smpl.utils import rot6d_to_rotmat, batch_rodrigues
from models.utils import build_dn_generator
from models.utils.transformer import inverse_sigmoid


@HEADS.register_module()
class HumanQueryHead(AnchorFreeHead):
    """
    original version of unihuman head, align with v9
    Args:
        num_classes (int): Number of categories excluding the background.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 num_query=900,
                 num_reg_fcs=2,
                 # with_box_refine=False,
                 as_two_stage=False,
                 sync_cls_avg_factor=False,
                 with_kp=True,
                 with_det=True,
                 with_seg=True,
                 with_attr=True,
                 with_smpl=True,
                 attr_cfg=[
                     dict(name='gender', num_classes=1),
                     dict(name='age', num_classes=85),
                 ],
                 kp_cfg=None,
                 seg_cfg=None,
                 smpl_cfg=None,
                 dn_cfg=None,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_attrs=[],
                 train_cfg=dict(
                     uni_assigner=dict(
                         type='UnihumanHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=1.0),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                         kpt_cost=dict(type='KptL1Cost', weight=70.0),
                         oks_cost=dict(type='OksV2Cost', weight=2.0),
                         mask_cost=dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                         dice_cost=dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
                         kpt_cls_cost=dict(type='FocalLossCost', weight=2.0),
                         seg_cls_cost=dict(type='ClassificationCost', weight=2.0),
                     ),
                 ),
                 test_cfg=dict(max_per_img=256),
                 init_cfg=None,
                 **kwargs):

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query

        self.attr_cfg = attr_cfg
        self.reg_out_channels = 4

        # print("===== [Task info] =====")
        assert with_det or with_seg or with_kp
        self.with_kp = with_kp
        self.with_det = with_det
        self.with_seg = with_seg
        self.with_attr = with_attr
        self.with_smpl = with_smpl
        self.kp_cfg = kp_cfg
        self.seg_cfg = seg_cfg
        self.smpl_cfg = smpl_cfg
        print("===== [Task info] =====")
        print("with_kp:", with_kp)
        print("with_det:", with_det)
        print("with_seg:", with_seg)
        print("with_attr:", with_attr)
        print("with_smpl:", with_smpl)
        # self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0

        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.num_classes = num_classes
        if self.with_kp == False or self.kp_cfg['with_kpt_refine'] == False:
            transformer['kpt_refine_decoder'] = None
        if self.with_kp == False:
            transformer['hm_encoder'] = None
        transformer['num_keypoints'] = self.kp_cfg['num_keypoints']
        self.transformer = build_transformer(transformer)
        self.decoder_nums = [transformer['decoder']['num_layers']]
        # , transformer['kpt_decoder']['num_layers']]
        self.embed_dims = self.transformer.embed_dims
        # self.query_pos = nn.Embedding(self.num_query, self.embed_dims)
        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)
        if loss_cls['use_sigmoid']:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_cls = build_loss(loss_cls)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.num_pred = self.transformer.decoder.num_layers + 1
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.cls_branches = _get_clones(fc_cls, self.num_pred)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        if train_cfg:
            assert 'uni_assigner' in train_cfg, 'assigner should be provided ' \
                                                'when train_cfg is set.'
            uni_assigner = train_cfg['uni_assigner']
            assert loss_cls['loss_weight'] == uni_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            if self.with_det and 'reg_cost' in uni_assigner:
                assert loss_bbox['loss_weight'] == uni_assigner['reg_cost'][
                    'weight'], 'The regression L1 weight for loss and matcher ' \
                               'should be exactly the same.'
                assert loss_iou['loss_weight'] == uni_assigner['iou_cost']['weight'], \
                    'The regression iou weight for loss and matcher should be' \
                    'exactly the same.'
            if self.with_kp and 'kpt_cost' in uni_assigner:
                assert self.kp_cfg['loss_kpt']['loss_weight'] == uni_assigner['kpt_cost'][
                    'weight'], 'The kpt weight for loss and matcher ' \
                               'should be exactly the same.'
                assert self.kp_cfg['loss_oks']['loss_weight'] == uni_assigner['oks_cost']['weight'], \
                    'The oks weight for loss and matcher should be' \
                    'exactly the same.'
            if self.with_seg and 'mask_cost' in uni_assigner:
                assert self.seg_cfg['loss_mask']['loss_weight'] == uni_assigner['mask_cost'][
                    'weight'], 'The mask weight for loss and matcher ' \
                               'should be exactly the same.'
                assert self.seg_cfg['loss_dice']['loss_weight'] == uni_assigner['dice_cost']['weight'], \
                    'The dice weight for loss and matcher should be' \
                    'exactly the same.'
            self.uni_assigner = build_assigner(uni_assigner)
        if self.with_det:
            # DETR sampling=False, so use PseudoSampler
            self.num_reg_fcs = num_reg_fcs
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg
            self.fp16_enabled = False
            self.loss_bbox = build_loss(loss_bbox)
            self.loss_iou = build_loss(loss_iou)

            self.act_cfg = transformer.get('act_cfg',
                                           dict(type='ReLU', inplace=True))
            self.activate = build_activation_layer(self.act_cfg)

            assert 'num_feats' in positional_encoding
            num_feats = positional_encoding['num_feats']

            assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                     f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                     f' and {num_feats}.'
            self._init_det_layers()

            self.init_denoising(dn_cfg)
            assert self.as_two_stage, \
                'as_two_stage must be True for DINO'
            # assert self.with_box_refine, \
            #     'with_box_refine must be True for DINO'
        if self.with_kp and 'kp_cfg' is not None:
            self._init_kp_layers()
        if self.with_seg and 'seg_cfg' is not None:
            self._init_seg_layers()
        if self.with_smpl:
            self._init_smpl_layers()
        if self.with_attr:
            self._init_attr_layers(loss_attrs)

    def _init_det_layers(self):
        """Initialize classification branch and regression branch of head."""

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.reg_out_channels))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.

        self.reg_branches = _get_clones(reg_branch, self.num_pred)

        self.label_embedding = nn.Embedding(self.num_classes,
                                            self.embed_dims)

    def _init_attr_layers(self, loss_attrs):
        fc_attrs = []
        for i in self.attr_cfg:
            fc_attrs.append(Linear(self.embed_dims, i['num_classes']))

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.attr_branches = nn.ModuleList()
        for i in range(len(self.attr_cfg)):
            self.attr_branches.append(_get_clones(fc_attrs[i], self.num_pred))
        assert len(loss_attrs) == len(self.attr_cfg)
        self.loss_attr = []
        for i in loss_attrs:
            self.loss_attr.append(build_loss(i))

    def _init_kp_layers(self):
        """Initialize classification branch and keypoint branch of head."""
        self.with_hm = self.kp_cfg['with_hm']
        self.num_keypoints = self.kp_cfg['num_keypoints']
        self.with_kpt_refine = self.kp_cfg['with_kpt_refine']
        self.num_kpt_fcs = self.kp_cfg['num_kpt_fcs']
        self.loss_kpt_cls = build_loss(self.kp_cfg['loss_kpt_cls'])
        self.loss_kpt = build_loss(self.kp_cfg['loss_kpt'])
        self.loss_kpt_refine = build_loss(self.kp_cfg['loss_kpt_refine'])
        self.loss_oks = build_loss(self.kp_cfg['loss_oks'])
        self.loss_oks_refine = build_loss(self.kp_cfg['loss_oks_refine'])
        self.loss_hm = build_loss(self.kp_cfg['loss_hm'])
        kpt_branch = []
        kpt_branch.append(Linear(self.embed_dims, 512))
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            kpt_branch.append(Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(512, 2 * self.num_keypoints))
        kpt_branch = nn.Sequential(*kpt_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.kpt_branches = _get_clones(kpt_branch, self.num_pred)
        refine_kpt_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_kpt_branch.append(nn.ReLU())
        refine_kpt_branch.append(Linear(self.embed_dims, 2))
        refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
        # kp need another cls branch because kp label is diff from bbox label due to invalid kps
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.kpt_cls_branches = _get_clones(fc_cls, self.num_pred)
        if self.with_kpt_refine:
            num_pred = self.transformer.kpt_refine_decoder.num_layers
            self.refine_kpt_branches = _get_clones(refine_kpt_branch, num_pred)
        if self.with_hm:
            self.fc_hm = Linear(self.embed_dims, self.num_keypoints)

    def _init_smpl_layers(self):
        """Initialize classification branch and keypoint branch of head."""
        # self.with_hm = self.kp_cfg['with_hm']
        # self.num_keypoints = self.kp_cfg['num_keypoints']
        # self.with_kpt_refine = self.kp_cfg['with_kpt_refine']
        # self.num_kpt_fcs = self.kp_cfg['num_kpt_fcs']
        self.loss_smpl_pose = build_loss(self.smpl_cfg['loss_smpl_pose'])
        self.loss_smpl_betas = build_loss(self.smpl_cfg['loss_smpl_betas'])
        # self.loss_pj2d = build_loss(self.smpl_cfg['loss_pj2d'])
        # self.loss_gtpj2d = build_loss(self.smpl_cfg['loss_gtpj2d'])
        self.loss_kp3d = build_loss(self.smpl_cfg['loss_kp3d'])
        self.pose_prior_loss = build_loss(self.smpl_cfg['pose_prior_loss'])

        if 'smpl_beta_weights' in self.smpl_cfg:
            self.smpl_beta_weights = self.smpl_cfg['smpl_beta_weights']
        else:
            self.smpl_beta_weights = [1, 0.64, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16]
        self.smpl_beta_weights = torch.Tensor(self.smpl_beta_weights).cuda()
        self.smpl_model_path = self.smpl_cfg['smpl_model_path']
        self.smpl_model = {
            'male': SMPLWrapper(self.smpl_model_path['m'], rot_type='6D'),
            'female': SMPLWrapper(self.smpl_model_path['f'], rot_type='6D'),
            'neutral': SMPLWrapper(self.smpl_model_path['n'], rot_type='6D')
        }

        # self.loss_kpt_refine = build_loss(self.kp_cfg['loss_kpt_refine'])
        # self.loss_oks = build_loss(self.kp_cfg['loss_oks'])
        # self.loss_oks_refine = build_loss(self.kp_cfg['loss_oks_refine'])
        # self.loss_hm = build_loss(self.kp_cfg['loss_hm'])
        smpl_pose_branch = []
        smpl_pose_branch.append(Linear(self.embed_dims, 512))
        smpl_pose_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs):
            smpl_pose_branch.append(Linear(512, 512))
            smpl_pose_branch.append(nn.ReLU())
        smpl_pose_branch.append(Linear(512, 144))
        smpl_pose_branch = nn.Sequential(*smpl_pose_branch)

        smpl_betas_branch = []
        smpl_betas_branch.append(Linear(self.embed_dims, 512))
        smpl_betas_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs):
            smpl_betas_branch.append(Linear(512, 512))
            smpl_betas_branch.append(nn.ReLU())
        smpl_betas_branch.append(Linear(512, 10))
        smpl_betas_branch = nn.Sequential(*smpl_betas_branch)

        smpl_cam_branch = []
        smpl_cam_branch.append(Linear(self.embed_dims, 512))
        smpl_cam_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs):
            smpl_cam_branch.append(Linear(512, 512))
            smpl_cam_branch.append(nn.ReLU())
        smpl_cam_branch.append(Linear(512, 3))
        smpl_cam_branch = nn.Sequential(*smpl_cam_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.smpl_pose_branches = _get_clones(smpl_pose_branch, self.num_pred)
        self.smpl_betas_branches = _get_clones(smpl_betas_branch, self.num_pred)
        # self.smpl_cam_branches = _get_clones(smpl_cam_branch, self.num_pred)
        self.SMPL54toLSP = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]

    def _init_seg_layers(self):
        self.feat_channels = self.seg_cfg['feat_channels']
        self.out_channels = self.seg_cfg['out_channels']
        seg_mask_branch = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels), nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels), nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.out_channels))

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.seg_mask_branches = _get_clones(seg_mask_branch,
                                             self.num_pred)  # [seg_mask_branch for _ in range(seg_pred_num)]
        self.seg_train_cfg = self.seg_cfg['train_cfg']
        if self.seg_train_cfg:
            self.seg_num_points = self.seg_train_cfg.get('num_points', 12544)
            self.seg_oversample_ratio = self.seg_train_cfg.get('oversample_ratio', 3.0)
            self.seg_importance_sample_ratio = self.seg_train_cfg.get('importance_sample_ratio', 0.75)

        self.loss_mask = build_loss(self.seg_cfg['loss_mask'])
        self.loss_dice = build_loss(self.seg_cfg['loss_dice'])

    def init_weights(self):
        nn.init.normal_(self.query_embed.weight.data)
        # nn.init.normal_(self.query_pos.weight.data)
        self.init_all_weights()

    def init_all_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        if self.with_attr:
            for i in self.attr_branches:
                bias_init = bias_init_with_prob(0.01)
                for m in i:
                    nn.init.constant_(m.bias, bias_init)
        if self.with_det:
            for m in self.reg_branches:
                constant_init(m[-1], 0, bias=0)
            nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
            if self.as_two_stage:
                for m in self.reg_branches:
                    nn.init.constant_(m[-1].bias.data[2:], 0.0)
        if self.with_kp:
            for m in self.kpt_branches:
                constant_init(m[-1], 0, bias=0)
            # initialization of keypoint refinement branch
            if self.with_kpt_refine:
                for m in self.refine_kpt_branches:
                    constant_init(m[-1], 0, bias=0)
            if self.with_hm:
                bias_init = bias_init_with_prob(0.1)
                normal_init(self.fc_hm, std=0.01, bias=bias_init)
        if self.with_smpl:
            self.init_smpl_weights(self.smpl_pose_branches)
            self.init_smpl_weights(self.smpl_betas_branches)
            # self.init_smpl_weights(self.smpl_cam_branches)

    def init_smpl_weights(self, mudule, bias=0):
        for m in mudule:
            for mm in m:
                if hasattr(mm, 'weight') and mm.weight is not None:
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
                if hasattr(mm, 'bias') and mm.weight is not None:
                    nn.init.constant_(mm.bias, bias)

    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_queries'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        self.dn_generator = build_dn_generator(dn_cfg)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        result = {}
        outs = self.forward_uni(feats, img_metas)
        (outputs_classes, outputs_coords, topk_score, topk_anchor, topk_attrs, outputs_attrs, \
         outputs_kpt_classes, outputs_kpts, topk_kpt_score, topk_kpt, hm_proto, memory, mlvl_masks, \
         topk_masks, outputs_masks, topk_smpl_pose, topk_smpl_betas, outputs_smpl_pose,
         outputs_smpl_betas) = outs
        inputs = (outputs_classes, outputs_coords, outputs_attrs, outputs_kpt_classes, outputs_kpts, outputs_masks, \
                  outputs_smpl_pose, outputs_smpl_betas, \
                  hm_proto, memory, mlvl_masks)
        results_list, kpt_results_list, mask_results_list = self.get_bboxes(*inputs, img_metas, rescale=rescale)
        bbox_result = results_list[0][0]
        label_result = results_list[0][1]
        result['det'] = bbox_result
        result['label'] = label_result
        if self.with_kp:
            kp_result = kpt_results_list[0]
            kp = kp_result[..., :2]
            kp = kp.reshape(max_per_img, -1)
            re = torch.cat((result['det'], kp), dim=-1)
            result['det'] = re
        if self.with_seg:
            result['seg'] = mask_results_list
        return result

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   attr_preds,
                   all_kpt_scores,
                   all_kpt_preds,
                   all_mask_preds,
                   all_smpl_pose_preds,
                   all_smpl_betas_preds,
                   # all_smpl_cams_preds,
                   hm_proto,
                   memory,
                   mlvl_masks,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        if self.with_det:
            bbox_preds = all_bbox_preds[-1]
        if self.with_attr:
            gender_preds = attr_preds[-1][..., 0]
            age_preds = attr_preds[-1][..., 1:]
        if self.with_kp:
            kpt_preds = all_kpt_preds[-1]
            kpt_cls_scores = all_kpt_scores[-1]
        if self.with_seg:
            mask_preds = all_mask_preds[-1]
        if self.with_smpl:
            smpl_pose_preds = all_smpl_pose_preds[-1]
            smpl_betas_preds = all_smpl_betas_preds[-1]
            # smpl_cams_preds = all_smpl_cams_preds[-1]
        result_list = []
        kpt_result_list = []
        mask_result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            if self.with_det:
                bbox_pred = bbox_preds[img_id]
            else:
                bbox_pred = None
            if self.with_attr:
                gender_pred = gender_preds[img_id]
                age_pred = age_preds[img_id]
            else:
                gender_pred = None
                age_pred = None
            if self.with_kp:
                kpt_cls_score = kpt_cls_scores[img_id]
                kpt_pred = kpt_preds[img_id]
            else:
                kpt_pred = None
                kpt_cls_score = None
            if self.with_smpl:
                smpl_pose_pred = smpl_pose_preds[img_id]
                smpl_betas_pred = smpl_betas_preds[img_id]
                # smpl_cams_pred = smpl_cams_preds[img_id]
            else:
                smpl_pose_pred = None
                smpl_betas_pred = None
                # smpl_cams_pred = None
            img_shape = img_metas[img_id]['img_shape']
            img_height, img_width = img_shape[:2]
            scale_factor = img_metas[img_id]['scale_factor']

            if self.with_seg:
                mask_pred = mask_preds[img_id]
                mask_pred = mask_pred[:, :img_height, :img_width]
            else:
                mask_pred = None

            bboxes_results, label_results, det_kpts = self._get_bboxes_single(cls_score, bbox_pred, gender_pred,
                                                                              age_pred,
                                                                              kpt_cls_score, kpt_pred, smpl_pose_pred,
                                                                              smpl_betas_pred,
                                                                              img_shape, scale_factor, memory,
                                                                              mlvl_masks,
                                                                              rescale)

            if self.with_seg:
                ori_height, ori_width = img_metas[img_id]['ori_shape'][:2]
                mask_pred = F.interpolate(
                    mask_pred[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                mask_result = dict()
                ins_results = self.instance_postprocess(
                    cls_score, mask_pred)
                mask_result['ins_results'] = ins_results
                mask_result_list.append(mask_result)
            kpt_result_list.append(det_kpts)
            results = [bboxes_results, label_results]
            result_list.append(results)
        return result_list, kpt_result_list, mask_result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           gender_pred,
                           age_pred,
                           kpt_cls_score,
                           kpt_pred,
                           smpl_pose_pred,
                           smpl_betas_pred,
                           # smpl_cams_pred,
                           img_shape,
                           scale_factor,
                           memory,
                           mlvl_masks,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background

        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        if self.with_attr:
            gender_pred = gender_pred[bbox_index]
            age_pred = age_pred[bbox_index]
        if self.with_kp:
            # TODO how to use kpt_cls_score
            if self.loss_cls.use_sigmoid:
                kpt_cls_score = kpt_cls_score.sigmoid()
                # kpt_scores, kpt_indexes = kpt_cls_score.view(-1).topk(max_per_img)
                # kpt_labels = kpt_indexes % self.num_classes
                # kpt_index = kpt_indexes // self.num_classes
            else:
                kpt_cls_score, kpt_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
                # kpt_scores, kpt_index = kpt_cls_score.topk(max_per_img)
                # # bbox_pred = bbox_pred[kpt_index]
                # kpt_labels = kpt_labels[kpt_index]
            # kpt_pred = kpt_pred[kpt_index]
            # kpt_cls_score = kpt_cls_score[kpt_index]
            kpt_pred = kpt_pred[bbox_index]
            kpt_cls_score = kpt_cls_score[bbox_index]
            # print("kpt_cls_score:",kpt_cls_score)
            # print("scores:", scores)
        if self.with_smpl:
            smpl_pose_pred = smpl_pose_pred[bbox_index]
            smpl_betas_pred = smpl_betas_pred[bbox_index]
            # smpl_cams_pred = smpl_cams_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred[..., :4])
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if self.with_attr:
            gender = (gender_pred.sigmoid() > 0.5).float()
            age_pred = age_pred.softmax(
                dim=-1)  # if use softmax loss, use softmax(); if use sigmoid loss, use sigmoid()
            age = 0
            for i in range(1, 85):
                age += i * age_pred[..., i]

        if self.with_kp:
            if self.with_kpt_refine:
                refine_targets = (kpt_pred, None, None, torch.ones_like(kpt_pred))
                refine_outputs = self.forward_kp_refine(memory, mlvl_masks,
                                                        refine_targets, None, None)
                kpts = refine_outputs[-1]
            else:
                num_q = kpt_pred.shape[0]
                kpts = kpt_pred.reshape(num_q, -1, 2)
            kpts[..., 0] = kpts[..., 0] * img_shape[1]
            kpts[..., 1] = kpts[..., 1] * img_shape[0]
            kpts[..., 0].clamp_(min=0, max=img_shape[1])
            kpts[..., 1].clamp_(min=0, max=img_shape[0])
        if rescale:
            if self.with_det:
                det_bboxes /= det_bboxes.new_tensor(scale_factor)
            if self.with_kp:
                kpts /= kpts.new_tensor(scale_factor[:2]).unsqueeze(0).unsqueeze(0)
        kp_cls_cat = scores.unsqueeze(1)
        if self.with_kp:
            kp_cls_cat = kpt_cls_score
        # print("det_bboxes:",det_bboxes.shape)
        # print("smpl_pose_pred:",smpl_pose_pred.shape)
        if not self.with_smpl:
            smpl_pose_pred = scores.unsqueeze(1).repeat(1, 144)
            smpl_betas_pred = scores.unsqueeze(1).repeat(1, 10)
            # smpl_cams_pred = scores.unsqueeze(1).repeat(1, 3)
        # else:
        #     smpl_pose_pred_angular=rot6D_to_angular(smpl_pose_pred)
        #     smpl_outs = self.smpl_model(poses=smpl_pose_pred, betas=smpl_betas_pred, cams=smpl_cams_pred)
        if not self.with_attr:
            gender = scores
            age = scores

        # print("smpl_pose_pred_placeholder:", smpl_pose_pred_placeholder.shape)
        # if self.with_attr:
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1), gender.unsqueeze(1), age.unsqueeze(1), smpl_pose_pred,
                                smpl_betas_pred, kp_cls_cat), -1)
        # else:
        #     det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1), scores.unsqueeze(1), scores.unsqueeze(1), smpl_pose_pred, smpl_betas_pred, smpl_cams_pred, kp_cls_cat), -1)
        # if self.with_attr:
        #     det_bboxes = torch.cat((det_bboxes, kpt_cls_score, gender.unsqueeze(1), age.unsqueeze(1)), -1)
        # else:
        #     det_bboxes = torch.cat((det_bboxes, kpt_cls_score, scores.unsqueeze(1), scores.unsqueeze(1)), -1)

        if self.with_kp:
            det_kpts = torch.cat((kpts, kpts.new_ones(kpts[..., :1].shape)), dim=2)
        else:
            det_kpts = None
        return det_bboxes, det_labels, det_kpts  # , mask_pred

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        # print("mask_cls:",mask_cls.shape)
        # scores, indexes = mask_cls.view(-1).topk(max_per_img)
        # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        scores = mask_cls.sigmoid()
        # print("scores:", scores.shape)
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device). \
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        # scores_per_image, top_indices = scores.flatten(0, 1).topk(
        #     max_per_img, sorted=False)
        scores_per_image, top_indices = scores.view(-1).topk(max_per_img)
        # scores_per_image, top_indices = mask_cls.view(-1).topk(max_per_img)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                        mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary

    def pad_single(self, gt_labels, gt_masks, img_metas):
        # print("gt_masks_before:",gt_masks.masks.shape)
        gt_masks = gt_masks.pad(img_metas['pad_shape'][:2], pad_val=0) \
            .to_tensor(dtype=torch.bool, device=gt_labels.device).long()
        # print("gt_masks_after:", gt_masks.shape)
        return gt_masks.unsqueeze(0)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_attributes,
                      gt_kps,
                      gt_areas,
                      gt_masks,
                      gt_valid,
                      gt_smpl_pose=None,
                      gt_smpl_betas=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        # seg_input=x[:-1]
        # det_kp_input=x[1:]
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        if self.with_det:
            assert self.dn_generator is not None, '"dn_cfg" must be set'
            dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
                self.dn_generator(gt_bboxes, gt_labels,
                                  self.label_embedding, img_metas)
        else:
            dn_label_query = None
            dn_bbox_query = None
            attn_mask = None
            dn_meta = None
        outputs_classes, outputs_coords, topk_score, topk_anchor, topk_attrs, outputs_attrs, \
            outputs_kpt_classes, outputs_kpts, topk_kpt_score, topk_kpt, kp_hm_proto, kp_memory, kp_mlvl_masks, \
            topk_masks, outputs_masks, \
            topk_smpl_pose, topk_smpl_betas, outputs_smpl_pose, outputs_smpl_betas = \
            self.forward_uni(x, img_metas, dn_label_query, dn_bbox_query, attn_mask)
        if self.with_seg:
            gt_masks = multi_apply(self.pad_single, gt_labels, gt_masks, img_metas)[0]

        gt_kpt_labels = copy.deepcopy(gt_labels)
        if self.with_kp:
            for i in range(len(gt_kps)):
                # print(i,gt_kps[i].shape)
                # print(gt_labels)
                num_instance, num_kps = gt_kps[i].shape
                for n_ins in range(num_instance):
                    is_valid = 0
                    for j in range(self.num_keypoints):
                        is_valid += gt_kps[i][n_ins, j * 3 + 2]
                    if not is_valid:
                        gt_kpt_labels[i][n_ins] = -1
        losses, kp_refine_targets = self.loss_all(outputs_classes, outputs_coords, outputs_attrs,
                                                  outputs_kpt_classes, outputs_kpts, outputs_masks,
                                                  outputs_smpl_pose, outputs_smpl_betas,
                                                  topk_score, topk_anchor, topk_attrs,  # kp_enc_outputs_class,
                                                  topk_kpt_score, topk_kpt, kp_hm_proto, topk_masks,
                                                  topk_smpl_pose, topk_smpl_betas,
                                                  gt_labels, gt_bboxes, gt_attributes,
                                                  gt_kpt_labels, gt_kps, gt_areas, gt_masks, gt_valid,
                                                  gt_smpl_pose, gt_smpl_betas,
                                                  img_metas, dn_meta)
        if self.with_kp and self.with_kpt_refine:
            losses = self.forward_kp_refine(kp_memory, kp_mlvl_masks, kp_refine_targets,
                                            losses, img_metas)

        return losses

    def loss_all(self,
                 all_cls_scores,
                 all_bbox_preds,
                 all_attrs_preds,
                 all_kpt_cls_scores,
                 all_kpt_preds,
                 all_mask_preds,
                 all_smpl_pose_preds,
                 all_smpl_betas_preds,
                 enc_topk_scores,
                 enc_topk_anchors,
                 enc_topk_attrs,
                 enc_topk_kpt_scores,
                 enc_topk_kpt,
                 enc_hm_proto,
                 enc_mask_pred,
                 enc_smpl_pose_pred,
                 enc_smpl_betas_pred,
                 gt_labels_list,
                 gt_bboxes_list,
                 gt_attrs_list,
                 gt_kpt_labels_list,
                 gt_keypoints_list,
                 gt_areas_list,
                 gt_masks_list,
                 gt_valids_list,
                 gt_smpl_pose,
                 gt_smpl_betas,
                 img_metas,
                 dn_meta=None,
                 gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        loss_dict = dict()
        # extract denoising and matching part of outputs
        all_cls_scores, all_bbox_preds, dn_cls_scores, dn_bbox_preds, dn_kpt_scores, dn_kpt_preds, dn_mask_preds, \
            all_attrs_preds, all_kpt_cls_scores, all_kpt_preds, all_mask_preds, all_smpl_pose_preds, all_smpl_betas_preds = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds, all_attrs_preds,
                                    all_kpt_cls_scores, all_kpt_preds, all_mask_preds,
                                    all_smpl_pose_preds, all_smpl_betas_preds, dn_meta)

        # Step1. Det losses
        # Step1.1 Det interm losses
        # TODO I dont know wheater it is right to use common assigner result to superwise enc_topk_output
        if enc_topk_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_attr, enc_losses_kpt_cls, enc_losses_kpt, enc_losses_oks, enc_losses_mask, enc_losses_dice, \
                a_, b_, c_, d_, e_, f_, g_, h_ = \
                self.loss_single(enc_topk_scores, enc_topk_anchors, enc_topk_kpt_scores, enc_topk_kpt, enc_mask_pred,
                                 gt_bboxes_list, gt_labels_list,
                                 enc_topk_attrs, enc_smpl_pose_pred, enc_smpl_betas_pred,
                                 gt_attrs_list,
                                 gt_kpt_labels_list, gt_keypoints_list, gt_areas_list,
                                 gt_masks_list, gt_valids_list, gt_smpl_pose, gt_smpl_betas,
                                 img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore)

            # collate loss from encode feature maps
            loss_dict['interm_loss_cls'] = enc_loss_cls
            if self.with_det:
                loss_dict['interm_loss_bbox'] = enc_losses_bbox
                loss_dict['interm_loss_iou'] = enc_losses_iou
            if self.with_kp:
                loss_dict['interm_loss_kpt_cls'] = enc_losses_kpt_cls
                loss_dict['interm_loss_kpt'] = enc_losses_kpt
                # loss_dict['interm_loss_oks'] = enc_losses_oks
            if self.with_seg:
                loss_dict['interm_loss_mask'] = enc_losses_mask
                loss_dict['interm_loss_dice'] = enc_losses_dice
            if self.with_attr:
                for i in range(len(self.attr_cfg)):
                    loss_dict['interm_loss_attr_' + self.attr_cfg[i]['name']] = enc_losses_attr[i]
        # Step1.2 calculate loss from all decoder layers
        num_det_dec_layers = self.decoder_nums[0]

        all_labels_list = [gt_labels_list for _ in range(num_det_dec_layers)]
        all_bbox_targets_list = [gt_bboxes_list for _ in range(num_det_dec_layers)]
        all_attrs_list = [gt_attrs_list for _ in range(num_det_dec_layers)]
        all_kpt_cls_list = [gt_kpt_labels_list for _ in range(num_det_dec_layers)]
        all_kpt_list = [gt_keypoints_list for _ in range(num_det_dec_layers)]
        all_areas_list = [gt_areas_list for _ in range(num_det_dec_layers)]
        all_mask_list = [gt_masks_list for _ in range(num_det_dec_layers)]
        all_valids_list = [gt_valids_list for _ in range(num_det_dec_layers)]
        all_smpl_pose_list = [gt_smpl_pose for _ in range(num_det_dec_layers)]
        all_smpl_shape_list = [gt_smpl_betas for _ in range(num_det_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_det_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_det_dec_layers)]
        if all_bbox_preds is None:
            all_bbox_preds = [None for _ in range(num_det_dec_layers)]
        if all_kpt_preds is None:
            all_kpt_cls_scores = [None for _ in range(num_det_dec_layers)]
            all_kpt_preds = [None for _ in range(num_det_dec_layers)]
        if all_mask_preds is None:
            all_mask_preds = [None for _ in range(num_det_dec_layers)]
        if all_attrs_preds is None:
            all_attrs_preds = [None for _ in range(num_det_dec_layers)]
        if all_smpl_pose_preds is None:
            all_smpl_pose_preds = [None for _ in range(num_det_dec_layers)]
        if all_smpl_betas_preds is None:
            all_smpl_betas_preds = [None for _ in range(num_det_dec_layers)]
        losses_cls, losses_bbox, losses_iou, losses_attrs, losses_kpt_cls, losses_kpt, losses_oks, losses_mask, losses_dice, \
            kpt_preds_list, kpt_targets_list, area_targets_list, kpt_weights_list, \
            losses_smpl_pose, losses_smpl_betas, losses_smpl_j3d, losses_smpl_pose_prior = \
            multi_apply(self.loss_single,
                        all_cls_scores, all_bbox_preds,
                        all_kpt_cls_scores, all_kpt_preds, all_mask_preds,
                        all_bbox_targets_list,
                        all_labels_list,
                        all_attrs_preds,
                        all_smpl_pose_preds,
                        all_smpl_betas_preds,
                        all_attrs_list,
                        all_kpt_cls_list,
                        all_kpt_list,
                        all_areas_list,
                        all_mask_list,
                        all_valids_list,
                        all_smpl_pose_list,
                        all_smpl_shape_list,
                        img_metas_list, all_gt_bboxes_ignore_list
                        )

        # collate loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        if self.with_det:
            loss_dict['loss_bbox'] = losses_bbox[-1]
            loss_dict['loss_iou'] = losses_iou[-1]
        if self.with_kp:
            loss_dict['loss_kpt_cls'] = losses_kpt_cls[-1]
            loss_dict['loss_kpt'] = losses_kpt[-1]
            loss_dict['loss_oks'] = losses_oks[-1]
        if self.with_seg:
            loss_dict['loss_mask'] = losses_mask[-1]
            loss_dict['loss_dice'] = losses_dice[-1]
        if self.with_attr:
            for i in range(len(self.attr_cfg)):
                loss_dict['loss_attr_' + self.attr_cfg[i]['name']] = losses_attrs[-1][i]
        if self.with_smpl:
            loss_dict['loss_smpl_pose'] = losses_smpl_pose[-1]
            loss_dict['loss_smpl_betas'] = losses_smpl_betas[-1]
            loss_dict['loss_smpl_j3d'] = losses_smpl_j3d[-1]
            loss_dict['loss_smpl_pose_prior'] = losses_smpl_pose_prior[-1]
        # collate loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_attr_i, loss_kpt_cls_i, loss_kpt_i, \
                loss_oks_i, loss_mask_i, loss_dice_i, loss_smpl_pose_i, loss_smpl_betas_i, loss_smpl_j3d_i, loss_smpl_pose_prior_i in \
                zip(losses_cls[:-1],
                    losses_bbox[:-1],
                    losses_iou[:-1],
                    losses_attrs[:-1],
                    losses_kpt_cls[:-1],
                    losses_kpt[:-1],
                    losses_oks[:-1],
                    losses_mask[:-1],
                    losses_dice[:-1],
                    losses_smpl_pose[:-1],
                    losses_smpl_betas[:-1],
                    losses_smpl_j3d[:-1],
                    losses_smpl_pose_prior[:-1]
                    ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            if self.with_det:
                loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            if self.with_kp:
                loss_dict[f'd{num_dec_layer}.loss_kpt_cls'] = loss_kpt_cls_i
                loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
                loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            if self.with_seg:
                loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
                loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            if self.with_attr:
                for i in range(len(self.attr_cfg)):
                    loss_dict[f'd{num_dec_layer}.loss_attr_' + self.attr_cfg[i]['name']] = loss_attr_i[i]
            if self.with_smpl:
                loss_dict[f'd{num_dec_layer}.loss_smpl_pose'] = loss_smpl_pose_i
                loss_dict[f'd{num_dec_layer}.loss_smpl_betas'] = loss_smpl_betas_i
                loss_dict[f'd{num_dec_layer}.loss_smpl_j3d'] = loss_smpl_j3d_i
                loss_dict[f'd{num_dec_layer}.loss_smpl_pose_prior'] = loss_smpl_pose_prior_i
            num_dec_layer += 1
        # Step1.3 Det dn losses
        if dn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_meta = [dn_meta for _ in img_metas]
            # dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_kpt, dn_losses_oks, dn_losses_mask, dn_losses_dice = self.loss_dn(
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, dn_kpt_preds, dn_mask_preds, gt_bboxes_list, gt_labels_list,
                gt_keypoints_list, gt_areas_list, gt_masks_list,
                img_metas, dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            if self.with_det:
                loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
                loss_dict['dn_loss_iou'] = dn_losses_iou[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1]
            ):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                if self.with_det:
                    loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                    loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1

        #     # Step 2.3 losses of heatmap generated from feature map
        if self.with_kp:
            hm_pred, hm_mask = enc_hm_proto
            loss_hm = self.loss_heatmap(hm_pred, hm_mask, gt_keypoints_list,
                                        gt_labels_list, gt_bboxes_list)
            loss_dict['loss_hm'] = loss_hm

        if self.with_kp:
            return loss_dict, (kpt_preds_list[-1], kpt_targets_list[-1],
                               area_targets_list[-1], kpt_weights_list[-1])
        else:
            return loss_dict, None

    def forward_uni(self,
                    mlvl_feats,
                    img_metas,
                    dn_label_query=None,
                    dn_bbox_query=None,
                    attn_mask=None):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats[1:]:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = self.query_embed
        hs, inter_references, topk_score, topk_kpt_score, topk_anchor, topk_attrs, topk_kpt, topk_mask_queries, \
            topk_smpl_pose, topk_smpl_betas, _, \
            hm_proto, memory, \
            mask_features, mlvl_features = \
            self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                # query_pos=self.query_pos.weight,
                cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                kpt_cls_branches=self.kpt_cls_branches if self.with_kp else None,
                reg_branches=self.reg_branches if self.with_det else None,  # noqa:E501
                attr_branches=self.attr_branches if self.with_attr else None,
                kpt_branches=self.kpt_branches if self.with_kp else None,
                mask_branches=self.seg_mask_branches if self.with_seg else None,
                smpl_pose_branches=self.smpl_pose_branches if self.with_smpl else None,
                smpl_betas_branches=self.smpl_betas_branches if self.with_smpl else None
            )
        # slice original detection cls and pred bbox
        topk_score = topk_score[..., :self.num_classes]
        topk_anchor = topk_anchor[..., :4]
        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        if self.with_det:
            outputs_coords = []
        if self.with_attr:
            outputs_attrs = []
        if self.with_kp:
            outputs_kpt_classes = []
            outputs_kpts = []
        if self.with_seg:
            outputs_masks = []
            topk_masks = torch.einsum('bqc,bchw->bqhw', topk_mask_queries, mask_features)
        else:
            topk_masks = None
        if self.with_smpl:
            outputs_smpl_pose = []
            outputs_smpl_betas = []
            # outputs_smpl_cams = []
        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_cls_branch = self.cls_branches[lvl](hs[lvl])
            outputs_classes.append(outputs_cls_branch)

            if self.with_attr:
                outputs_attr_branch = []
                for i in range(len(self.attr_cfg)):
                    outputs_attr_branch.append(self.attr_branches[i][lvl](hs[lvl]))
                outputs_attr_branch = torch.cat(outputs_attr_branch, -1)
                outputs_attrs.append(outputs_attr_branch)
            if self.with_kp:
                tmp_kpt = self.kpt_branches[lvl](hs[lvl])
                tmp_kpt[..., 0::2] += reference[..., 0:1]
                tmp_kpt[..., 1::2] += reference[..., 1:2]
                outputs_kpts.append(tmp_kpt.sigmoid())
                outputs_kpt_cls_branch = self.kpt_cls_branches[lvl](hs[lvl])
                outputs_kpt_classes.append(outputs_kpt_cls_branch)
            # pred bbox shape [n, 4]
            # outputs_class = outputs_cls_branch[...,:self.num_classes]
            if self.with_det:
                tmp = self.reg_branches[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    bbox_pred = tmp[..., :4]
                    bbox_pred += reference
                else:
                    assert reference.shape[-1] == 2
                    bbox_pred = tmp[..., :4]
                    bbox_pred[..., :2] += reference

                outputs_coord = bbox_pred.sigmoid()
                outputs_coords.append(outputs_coord)
            if self.with_seg:
                mask_embed = self.seg_mask_branches[lvl](hs[lvl])
                mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
                outputs_masks.append(mask_pred)
            if self.with_smpl:
                outputs_smpl_pose.append(self.smpl_pose_branches[lvl](hs[lvl]))
                outputs_smpl_betas.append(self.smpl_betas_branches[lvl](hs[lvl]))
                # print("hs[lvl]:",hs[lvl].shape)
                # print("mask_features:",mask_features.shape)
                # cam_feature = torch.einsum('bqc,bchw->bqc', hs[lvl], mask_features)
                # print("outputs_coord:",outputs_coord)
                # if self.cam_feature_fusion == "det":
                #     cam_feature = torch.einsum('bqc,bqd->bqc', hs[lvl], outputs_coord.detach())
                # elif self.cam_feature_fusion == "pose":
                #     cam_feature = torch.einsum('bqc,bqd->bqc', hs[lvl], tmp_kpt.sigmoid().detach())
                # else:
                #     cam_feature = hs[lvl]
                # outputs_smpl_cams.append(self.smpl_cam_branches[lvl](cam_feature))

        outputs_classes = torch.stack(outputs_classes)
        if self.with_det:
            outputs_coords = torch.stack(outputs_coords)
        else:
            outputs_coords = None
        if self.with_attr:
            outputs_attrs = torch.stack(outputs_attrs)
        else:
            outputs_attrs = None
        if self.with_kp:
            outputs_kpts = torch.stack(outputs_kpts)
            outputs_kpt_classes = torch.stack(outputs_kpt_classes)
        else:
            outputs_kpts = None
            outputs_kpt_classes = None
        if self.with_seg:
            outputs_masks = torch.stack(outputs_masks)
        else:
            outputs_masks = None
        if self.with_smpl:
            outputs_smpl_pose = torch.stack(outputs_smpl_pose)
            outputs_smpl_betas = torch.stack(outputs_smpl_betas)
        else:
            outputs_smpl_pose = None
            outputs_smpl_betas = None
        if self.with_kp and hm_proto is not None:
            # get heatmap prediction (training phase)
            hm_memory, hm_mask = hm_proto
            hm_pred = self.fc_hm(hm_memory)
            hm_proto = (hm_pred.permute(0, 3, 1, 2), hm_mask)

        return outputs_classes, outputs_coords, topk_score, topk_anchor, topk_attrs, outputs_attrs, \
            outputs_kpt_classes, outputs_kpts, topk_kpt_score, topk_kpt, hm_proto, memory, mlvl_masks, \
            topk_masks, outputs_masks, topk_smpl_pose, topk_smpl_betas, outputs_smpl_pose, outputs_smpl_betas

    def forward_kp_refine(self, memory, mlvl_masks, refine_targets, losses,
                          img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        kpt_preds, kpt_targets, area_targets, kpt_weights = refine_targets

        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_preds = torch.zeros_like(kpt_preds[:1])
            pos_img_inds = kpt_preds.new_zeros([1], dtype=torch.int64)
        else:
            pos_kpt_preds = kpt_preds[pos_inds]
            pos_img_inds = (pos_inds.nonzero() / self.num_query).squeeze(1).to(
                torch.int64)
        hs, init_reference, inter_references = self.transformer.forward_kpt_refine(
            mlvl_masks,
            memory,
            pos_kpt_preds.detach(),
            pos_img_inds,
            kpt_branches=self.refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_kpts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 2
            tmp_kpt += reference
            outputs_kpt = tmp_kpt.sigmoid()
            outputs_kpts.append(outputs_kpt)
        outputs_kpts = torch.stack(outputs_kpts)

        if not self.training:
            return outputs_kpts

        batch_size = mlvl_masks[0].size(0)
        factors = []
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            factor = mlvl_masks[0].new_tensor(
                [img_w, img_h, img_w, img_h],
                dtype=torch.float32).unsqueeze(0).repeat(self.num_query, 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)

        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        num_total_pos = kpt_weights.new_tensor([outputs_kpts.size(1)])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        pos_kpt_weights = kpt_weights[pos_inds]
        pos_kpt_targets = kpt_targets[pos_inds]
        pos_kpt_targets_scaled = pos_kpt_targets * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        for i, kpt_refine_preds in enumerate(outputs_kpts):
            if pos_inds.sum() == 0:
                loss_kpt = loss_oks = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                losses[f'd{i}.loss_oks_refine'] = loss_oks
                continue
            # kpt L1 Loss
            pos_refine_preds = kpt_refine_preds.reshape(
                kpt_refine_preds.size(0), -1)
            loss_kpt = self.loss_kpt_refine(
                pos_refine_preds,
                pos_kpt_targets,
                pos_kpt_weights,
                avg_factor=num_valid_kpt)
            losses[f'd{i}.loss_kpt_refine'] = loss_kpt
            # kpt oks loss
            pos_refine_preds_scaled = pos_refine_preds * factors
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks_refine(
                pos_refine_preds_scaled,
                pos_kpt_targets_scaled,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)
            losses[f'd{i}.loss_oks_refine'] = loss_oks
        return losses

    def get_all_targets(self,
                        cls_scores_list,
                        bbox_preds_list,
                        kpt_cls_scores_list,
                        kpt_preds_list,
                        mask_preds_list,
                        gt_labels_list,
                        gt_bboxes_list,
                        gt_attrs_list,
                        gt_kpt_labels_list,
                        gt_keypoints_list,
                        gt_areas_list,
                        gt_masks_list,
                        gt_valids_list,
                        gt_smpl_pose_list,
                        gt_smpl_betas_list,
                        img_metas,
                        gt_bboxes_ignore_list=None
                        ):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list[0])
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        for i in range(len(gt_attrs_list)):
            if gt_attrs_list[i].shape[0] == 0:
                gt_attrs_list[i] = torch.Tensor([[-1, -1]]).long().cuda()

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, attrs_list, attrs_weight_list,
         kpt_labels, kpt_label_weights, kpt_targets_list, kpt_weights_list,
         area_targets_list,
         mask_targets_list, mask_weights_list,
         valids_list,
         smpl_pose_targets, smpl_betas_targets, smpl_pose_weights, smpl_betas_weights,
         pos_inds_list, neg_inds_list,
         kpt_pos_inds_list, kpt_neg_inds_list) = \
            multi_apply(self.get_targets_single,
                        cls_scores_list, bbox_preds_list,
                        kpt_cls_scores_list, kpt_preds_list, mask_preds_list,
                        gt_labels_list, gt_bboxes_list, gt_attrs_list,
                        gt_kpt_labels_list, gt_keypoints_list, gt_areas_list,
                        gt_masks_list, gt_valids_list,
                        gt_smpl_pose_list, gt_smpl_betas_list,
                        img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        if self.with_kp:
            num_total_kpt_pos = sum((inds.numel() for inds in kpt_pos_inds_list))
            num_total_kpt_neg = sum((inds.numel() for inds in kpt_neg_inds_list))
        else:
            num_total_kpt_pos = 0
            num_total_kpt_neg = 0
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, attrs_list, attrs_weight_list,
                kpt_labels, kpt_label_weights, kpt_targets_list, kpt_weights_list,
                area_targets_list,
                mask_targets_list, mask_weights_list,
                valids_list,
                smpl_pose_targets, smpl_betas_targets, smpl_pose_weights, smpl_betas_weights,
                num_total_pos, num_total_neg,
                num_total_kpt_pos, num_total_kpt_neg)

    def get_targets_single(self,
                           cls_score,
                           bbox_pred,
                           kpt_cls_score,
                           kpt_pred,
                           mask_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attrs,
                           gt_kpt_labels,
                           gt_keypoints,
                           gt_areas,
                           gt_masks,
                           gt_valids,
                           gt_smpl_pose,
                           gt_smpl_betas,
                           img_meta,
                           gt_bboxes_ignore=None
                           ):
        num_bboxes = bbox_pred.size(0)

        num_seg_query = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        if self.with_seg:
            point_coords = torch.rand((1, self.seg_num_points, 2),
                                      device=cls_score.device)
            mask_points_pred = point_sample(
                mask_pred.unsqueeze(1), point_coords.repeat(num_seg_query, 1,
                                                            1)).squeeze(1)
            # shape (num_gts, num_points)
            gt_points_masks = point_sample(
                gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                                   1)).squeeze(1)
        else:
            mask_points_pred = None
            gt_points_masks = None

        assign_result = self.uni_assigner.assign(cls_score, bbox_pred, kpt_pred, mask_points_pred,
                                                 gt_labels, gt_bboxes, gt_keypoints, gt_areas, gt_points_masks,
                                                 img_meta, gt_kpt_labels=gt_kpt_labels, kpt_cls_pred=kpt_cls_score)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        valids = gt_bboxes.new_full((num_bboxes, 8),
                                    0,
                                    dtype=torch.long)
        # print(valids.shape)
        valid_indx = gt_kpt_labels[sampling_result.pos_assigned_gt_inds] == 0
        kpt_fix_pos_inds = pos_inds[valid_indx]
        kpt_fix_pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds[valid_indx]

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # print("pos_inds:",pos_inds)
        # print("gt_valids:",gt_valids.shape)
        # print("valids:",valids.shape)
        if len(pos_inds > 0):
            valids[pos_inds] = gt_valids[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # assigner attrs

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        attrs = []
        attrs_weight = []
        if gt_attrs is not None and len(gt_attrs) > 0:
            for i in range(len(self.attr_cfg)):
                attrs.append(gt_bboxes.new_full((num_bboxes,),
                                                -1,
                                                dtype=torch.long))
                attrs_weight.append(torch.zeros_like(label_weights))
                attrs[-1][pos_inds] = gt_attrs[..., i][sampling_result.pos_assigned_gt_inds]
                attrs_weight[-1][pos_inds] = 1.0
                attrs_weight[-1][attrs[-1] == -1] = 0.0
        if self.with_kp:
            kpt_labels = gt_bboxes.new_full((num_bboxes,),
                                            self.num_classes,
                                            dtype=torch.long)
            kpt_labels[kpt_fix_pos_inds] = gt_kpt_labels[kpt_fix_pos_assigned_gt_inds]
            kpt_label_weights = gt_bboxes.new_ones(num_bboxes)
            # for i in pos_inds:
            #     if kpt_labels[i] != 0:
            #         kpt_label_weights[i] = 0
            # print("kpt_labels:",kpt_labels)
            kpt_pos_inds = torch.nonzero(
                kpt_labels == 0, as_tuple=False).squeeze(-1).unique()
            kpt_neg_inds = torch.nonzero(
                kpt_labels != 0, as_tuple=False).squeeze(-1).unique()
            # print("pos_inds:", pos_inds)
            # print("kpt_fix_pos_inds:",kpt_fix_pos_inds)
            # print("kpt_pos_inds:", kpt_pos_inds)
            # keypoint targets
            kpt_targets = torch.zeros_like(kpt_pred)
            kpt_weights = torch.zeros_like(kpt_pred)
            pos_gt_kpts = gt_keypoints[kpt_fix_pos_assigned_gt_inds]
            pos_gt_kpts = pos_gt_kpts.reshape(pos_gt_kpts.shape[0],
                                              pos_gt_kpts.shape[-1] // 3, 3)
            valid_idx = pos_gt_kpts[:, :, 2] > 0
            area_targets = kpt_pred.new_zeros(
                kpt_pred.shape[0])  # get areas for calculating oks
            if kpt_fix_pos_inds.shape[0] != 0:
                pos_kpt_weights = kpt_weights[kpt_pos_inds].reshape(
                    pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2)
                pos_kpt_weights[valid_idx] = 1.0
                kpt_weights[kpt_pos_inds] = pos_kpt_weights.reshape(
                    pos_kpt_weights.shape[0], kpt_pred.shape[-1])

                factor = kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0)
                pos_gt_kpts_normalized = pos_gt_kpts[..., :2]
                pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
                                                 factor[:, 0:1]
                pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
                                                 factor[:, 1:2]
                kpt_targets[kpt_pos_inds] = pos_gt_kpts_normalized.reshape(
                    pos_gt_kpts.shape[0], kpt_pred.shape[-1])
                pos_gt_areas = gt_areas[kpt_fix_pos_assigned_gt_inds]
                area_targets[kpt_pos_inds] = pos_gt_areas

        else:
            kpt_targets = None
            kpt_weights = None
            area_targets = None
            kpt_labels = None
            kpt_label_weights = None
            kpt_pos_inds = None
            kpt_neg_inds = None
        if self.with_seg:
            # mask target
            mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
            mask_weights = mask_pred.new_zeros((self.num_query,))
            mask_weights[pos_inds] = 1.0
        else:
            mask_targets = None
            mask_weights = None
        if self.with_smpl:
            *a, b = bbox_pred.shape
            smpl_pose_targets = torch.zeros((*a, 72)).cuda()
            smpl_betas_targets = torch.zeros((*a, 10)).cuda()
            # print("gt_smpl_pose:",gt_smpl_pose.shape)
            pos_smpl_poses = gt_smpl_pose[kpt_fix_pos_assigned_gt_inds]
            pos_smpl_betas = gt_smpl_betas[kpt_fix_pos_assigned_gt_inds]
            smpl_pose_targets[kpt_pos_inds] = pos_smpl_poses
            smpl_betas_targets[kpt_pos_inds] = pos_smpl_betas

            # the valid flag index of smpl is 5
            invalid_smpl = valids[:, 5] < 1

            smpl_pose_weights = torch.zeros_like(smpl_pose_targets)
            # print("smpl_pose_weights:", smpl_pose_weights.shape)

            # step1. is matching target
            # step2. target gt has smpl label
            # step3. for kp and kp3d, the visible of each point
            smpl_pose_weights[kpt_pos_inds] = 1.0
            smpl_pose_weights[invalid_smpl] = 0

            smpl_betas_weights = torch.zeros_like(smpl_betas_targets)
            smpl_betas_weights[kpt_pos_inds] = 1.0
            smpl_betas_weights[invalid_smpl] = 0

            # kpt3d_targets = torch.zeros((*a, 54 * 3)).cuda()
            # kpt3d_targets[kpt_pos_inds] = gt_keypoints3d[kpt_fix_pos_assigned_gt_inds]
        else:
            smpl_pose_targets = None
            smpl_betas_targets = None
            smpl_pose_weights = None
            smpl_betas_weights = None
            # kpt3d_targets = None
        return labels, label_weights, bbox_targets, bbox_weights, \
            attrs, attrs_weight, kpt_labels, kpt_label_weights, kpt_targets, kpt_weights, area_targets, \
            mask_targets, mask_weights, valids, smpl_pose_targets, smpl_betas_targets, smpl_pose_weights, smpl_betas_weights, \
            pos_inds, neg_inds, kpt_pos_inds, kpt_neg_inds

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    kpt_cls_scores,
                    kpt_preds,
                    mask_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    attrs_preds,
                    smpl_pose_preds,
                    smpl_betas_preds,
                    # smpl_cams_preds,
                    gt_attrs_list,
                    gt_kpt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    gt_masks_list,
                    gt_valids_list,
                    gt_smpl_pose_list,
                    gt_smpl_betas_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i, ..., :self.num_classes] for i in range(num_imgs)]
        if self.with_det:
            bbox_preds_list = [bbox_preds[i, ..., :4] for i in range(num_imgs)]
        else:
            bbox_preds_list = [None for i in range(num_imgs)]
        if self.with_kp:
            kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
            kpt_cls_scores_list = [kpt_cls_scores[i, ..., :self.num_classes] for i in range(num_imgs)]
        else:
            kpt_preds_list = [None for i in range(num_imgs)]
            kpt_cls_scores_list = [None for i in range(num_imgs)]
        if self.with_seg:
            mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        else:
            mask_preds_list = [None for i in range(num_imgs)]
        if self.with_smpl:
            smpl_pose_preds_list = [smpl_pose_preds[i] for i in range(num_imgs)]
            smpl_betas_preds_list = [smpl_betas_preds[i] for i in range(num_imgs)]
            # smpl_cams_preds_list = [smpl_cams_preds[i] for i in range(num_imgs)]
        else:
            smpl_pose_preds_list = [None for i in range(num_imgs)]
            smpl_betas_preds_list = [None for i in range(num_imgs)]
            # smpl_cams_preds_list = [None for i in range(num_imgs)]
        cls_reg_targets = self.get_all_targets(cls_scores_list, bbox_preds_list, kpt_cls_scores_list, kpt_preds_list,
                                               mask_preds_list,
                                               gt_labels_list, gt_bboxes_list, gt_attrs_list, gt_kpt_labels_list,
                                               gt_keypoints_list, gt_areas_list, gt_masks_list,
                                               gt_valids_list, gt_smpl_pose_list, gt_smpl_betas_list,
                                               img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, attrs_list, attrs_weight_list,
         kpt_labels_list, kpt_label_weight_list, kpt_targets_list, kpt_weights_list, area_targets_list,
         mask_list, mask_weight_list, valids_list, smpl_pose_targets, smpl_betas_targets,
         smpl_pose_weights, smpl_betas_weights, num_total_pos, num_total_neg, num_total_kpt_pos,
         num_total_kpt_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        # print("valids_list:", valids_list)
        valids = torch.cat(valids_list, 0)
        # print("valids:",valids)
        if self.with_det:
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)
        if self.with_kp:
            kpt_targets = torch.cat(kpt_targets_list, 0)
            kpt_weights = torch.cat(kpt_weights_list, 0)
            area_targets = torch.cat(area_targets_list, 0)
            kpt_labels = torch.cat(kpt_labels_list, 0)
            kpt_label_weights = torch.cat(kpt_label_weight_list, 0)
        if self.with_smpl:
            smpl_pose_targets = torch.cat(smpl_pose_targets, 0)
            smpl_betas_targets = torch.cat(smpl_betas_targets, 0)
            smpl_pose_weights = torch.cat(smpl_pose_weights, 0)
            smpl_betas_weights = torch.cat(smpl_betas_weights, 0)
            smpl_pose_preds = torch.cat(smpl_pose_preds_list, 0)
            smpl_betas_preds = torch.cat(smpl_betas_preds_list, 0)
            # smpl_cams_preds = torch.cat(smpl_cams_preds_list, 0)
            # kpt3d_targets = torch.cat(kpt_3d_targets_list, 0)
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        loss_attrs = []
        if self.with_attr:
            attrs = []
            attrs_weight = []
            for i in range(len(self.attr_cfg)):
                tmp_attr = []
                tmp_attr_weight = []
                for j in range(len(attrs_list)):
                    tmp_attr.append(attrs_list[j][i])
                    tmp_attr_weight.append(attrs_weight_list[j][i])
                attrs.append(torch.cat(tmp_attr, 0))
                attrs_weight.append(torch.cat(tmp_attr_weight, 0))

            start = 0
            for i in range(len(self.attr_cfg)):
                reshape_attrs_preds = attrs_preds[..., start:start + self.attr_cfg[i]['num_classes']].reshape(-1,
                                                                                                              self.attr_cfg[
                                                                                                                  i][
                                                                                                                  'num_classes'])
                start += self.attr_cfg[i]['num_classes']
                attr_num_total = (attrs[i] > -0.5).sum()
                # no pos sample, to avoid devide by 0 bug, change attr_num_total to 1, the loss will be 0 because all-0 loss_weight
                if attr_num_total == 0:
                    attr_num_total += 1
                loss_attrs.append(
                    self.loss_attr[i](reshape_attrs_preds, attrs[i], attrs_weight[i], avg_factor=attr_num_total))
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        if self.with_det:
            # DETR regress the relative position of boxes (cxcywh) in the image,
            # thus the learning target is normalized by the image size. So here
            # we need to re-scale them for calculating IoU loss

            bbox_preds = bbox_preds.reshape(-1, 4)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
            # regression L1 loss
            loss_bbox = self.loss_bbox(
                bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        else:
            loss_iou = 0
            loss_bbox = 0

        if self.with_kp:
            kpt_cls_scores = kpt_cls_scores.reshape(-1, self.num_classes)
            # construct weighted avg_factor to match with the official DETR repo
            kpt_cls_avg_factor = num_total_kpt_pos * 1.0 + \
                                 num_total_kpt_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                kpt_cls_avg_factor = reduce_mean(
                    kpt_cls_scores.new_tensor([kpt_cls_avg_factor]))
            kpt_cls_avg_factor = max(kpt_cls_avg_factor, 1)
            loss_kpt_cls = self.loss_kpt_cls(
                kpt_cls_scores, kpt_labels, kpt_label_weights, avg_factor=kpt_cls_avg_factor)

            # keypoint regression loss
            kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
            num_valid_kpt = torch.clamp(
                reduce_mean(kpt_weights.sum()), min=1).item()
            # assert num_valid_kpt == (kpt_targets>0).sum().item()
            loss_kpt = self.loss_kpt(
                kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)
            # keypoint oks loss
            pos_inds = kpt_weights.sum(-1) > 0
            factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)
            pos_kpt_preds = kpt_preds[pos_inds] * factors
            pos_kpt_targets = kpt_targets[pos_inds] * factors
            pos_areas = area_targets[pos_inds]
            pos_valid = kpt_weights[pos_inds, 0::2]
            # print("len(pos_areas):",len(pos_areas))
            num_total_kpt_pos = loss_kpt_cls.new_tensor([num_total_kpt_pos])
            num_total_kpt_pos = torch.clamp(reduce_mean(num_total_kpt_pos), min=1).item()
            if len(pos_areas) == 0:
                loss_oks = pos_kpt_preds.sum() * 0
            else:
                assert (pos_areas > 0).all()
                loss_oks = self.loss_oks(
                    pos_kpt_preds,
                    pos_kpt_targets,
                    pos_valid,
                    pos_areas,
                    avg_factor=num_total_kpt_pos)

        else:
            loss_kpt = 0
            loss_oks = 0
            loss_kpt_cls = 0

        if self.with_seg:
            max_h = -1
            max_w = -1
            for i in range(len(mask_list)):
                _, h, w = mask_list[i].shape
                max_h = max(max_h, h)
                max_w = max(max_w, w)
            for i in range(len(mask_list)):
                n, h, w = mask_list[i].shape
                new_mask = torch.zeros((n, max_h, max_w))
                new_mask[:, :h, :w] = mask_list[i].clone()
                mask_list[i] = new_mask.cuda()
            mask_targets = torch.cat(mask_list, dim=0)
            # shape (batch_size, num_seg_query)
            mask_weights = torch.stack(mask_weight_list, dim=0)

            num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
            num_total_masks = max(num_total_masks, 1)

            # extract positive ones
            # shape (batch_size, num_seg_query, h, w) -> (num_total_gts, h, w)
            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum() * 0.0
                loss_mask = mask_preds.sum() * 0.0
                # return loss_cls, loss_bbox, loss_iou, loss_attrs, loss_kpt, loss_oks, loss_mask, loss_dice, \
                #     kpt_preds, kpt_targets, area_targets, kpt_weights
            else:
                with torch.no_grad():
                    points_coords = get_uncertain_point_coords_with_randomness(
                        mask_preds.unsqueeze(1), None, self.seg_num_points,
                        self.seg_oversample_ratio, self.seg_importance_sample_ratio)
                    # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                    mask_point_targets = point_sample(
                        mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
                # shape (num_seg_query, h, w) -> (num_seg_query, num_points)
                mask_point_preds = point_sample(
                    mask_preds.unsqueeze(1), points_coords).squeeze(1)

                # dice loss
                loss_dice = self.loss_dice(
                    mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

                # mask loss
                # shape (num_seg_query, num_points) -> (num_seg_query * num_points, )
                mask_point_preds = mask_point_preds.reshape(-1)
                # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
                mask_point_targets = mask_point_targets.reshape(-1)
                loss_mask = self.loss_mask(
                    mask_point_preds,
                    mask_point_targets,
                    avg_factor=num_total_masks * self.seg_num_points)
        else:
            loss_mask = 0
            loss_dice = 0

        loss_j3d = loss_pose_prior = loss_smpl_pose = loss_smpl_betas = labels.sum() * 0.0
        if self.with_smpl:
            # step 1 pos_inds (from assigner)
            # step 2 has 3d gt
            # 有3dgt的index
            # print("{'labels': 0, 'bboxes': 1, 'keypoints_2d': 2, 'keypoints_3d': 3, 'mask': 4, 'smpl': 5,'gender': 6, 'age': 7}")
            valid_smpl_gt_index = valids[:, 5] > 0
            pos_has_3d_inds = pos_inds & valid_smpl_gt_index

            smpl_pose_preds = smpl_pose_preds.reshape(-1, 144)
            smpl_betas_preds = smpl_betas_preds.reshape(-1, 10)
            smpl_pose_weights_ = smpl_pose_weights.reshape(-1, 24, 3, 1)

            select_smpl_poses = smpl_pose_preds[pos_has_3d_inds]
            select_smpl_betas = smpl_betas_preds[pos_has_3d_inds]
            select_smpl_poses_gt = smpl_pose_targets[pos_has_3d_inds]
            select_smpl_betas_gt = smpl_betas_targets[pos_has_3d_inds]
            smpl_pose_weights_ = smpl_pose_weights_[pos_has_3d_inds]
            smpl_betas_weights = smpl_betas_weights[pos_has_3d_inds]
            gender = attrs[0][pos_has_3d_inds]
            # male  female 1
            male_index = (gender == 0)
            female_index = (gender == 1)
            unknown_index = (gender < 0)
            num_pos_has_3dgt = pos_has_3d_inds.sum()
            num_valid_smpl_pose = torch.clamp(reduce_mean(smpl_pose_weights_.sum() * 3), min=1).item()
            num_valid_smpl_betas = torch.clamp(reduce_mean(smpl_betas_weights.sum()), min=1).item()
            num_valid_smpl_3d = torch.clamp(reduce_mean(smpl_pose_weights_.sum()) * 9 / 4, min=1).item()
            unknown_num = unknown_index.sum()
            male_num = male_index.sum()
            female_num = female_index.sum()
            num_valid_smpl_3d_unknown = torch.clamp(
                reduce_mean(smpl_pose_weights_.sum()) * 9 / 4 * (unknown_num / num_pos_has_3dgt), min=1).item()
            num_valid_smpl_3d_male = torch.clamp(
                reduce_mean(smpl_pose_weights_.sum()) * 9 / 4 * (male_num / num_pos_has_3dgt), min=1).item()
            num_valid_smpl_3d_female = torch.clamp(
                reduce_mean(smpl_pose_weights_.sum()) * 9 / 4 * (female_num / num_pos_has_3dgt), min=1).item()
            if num_pos_has_3dgt != 0:
                smpl_pose_preds_angular = rot6D_to_angular(select_smpl_poses)
                loss_pose_prior = self.pose_prior_loss(smpl_pose_preds_angular)

                smpl_pose_preds_ = rot6d_to_rotmat(select_smpl_poses).view(num_pos_has_3dgt, 24, 3, 3)
                smpl_pose_targets_ = batch_rodrigues(select_smpl_poses_gt.view(-1, 3)).view(-1, 24, 3, 3)

                # if smpl_pose_preds_.shape[0] > 0:
                loss_smpl_pose = self.loss_smpl_pose(
                    smpl_pose_preds_, smpl_pose_targets_, smpl_pose_weights_, avg_factor=num_valid_smpl_pose)
                loss_smpl_betas = self.loss_smpl_betas(
                    select_smpl_betas, select_smpl_betas_gt, smpl_betas_weights * self.smpl_beta_weights,
                    avg_factor=num_valid_smpl_betas)

                loss_j3d_n = loss_j3d_m = loss_j3d_f = loss_j3d
                if unknown_num > 0:
                    smpl_outs_gt = self.smpl_model['neutral'](poses=select_smpl_poses_gt[unknown_index],
                                                              betas=select_smpl_betas_gt[unknown_index],
                                                              cams=None,
                                                              rot_wrapper="3D")
                    smpl_outs_preds = self.smpl_model['neutral'](poses=select_smpl_poses[unknown_index],
                                                                 betas=select_smpl_betas[unknown_index],
                                                                 cams=None)
                    loss_j3d_n = self.loss_kp3d(smpl_outs_preds['j3d'], smpl_outs_gt['j3d'],
                                                avg_factor=num_valid_smpl_3d_unknown)

                if male_num > 0:
                    smpl_outs_gt_m = self.smpl_model['male'](poses=select_smpl_poses_gt[male_index],
                                                             betas=select_smpl_betas_gt[male_index],
                                                             cams=None,
                                                             rot_wrapper="3D")
                    smpl_outs_preds_m = self.smpl_model['male'](poses=select_smpl_poses[male_index],
                                                                betas=select_smpl_betas[male_index],
                                                                cams=None)
                    loss_j3d_m = self.loss_kp3d(smpl_outs_preds_m['j3d'], smpl_outs_gt_m['j3d'],
                                                avg_factor=num_valid_smpl_3d_male)
                if female_num > 0:
                    smpl_outs_gt_f = self.smpl_model['female'](poses=select_smpl_poses_gt[female_index],
                                                               betas=select_smpl_betas_gt[female_index],
                                                               cams=None,
                                                               rot_wrapper="3D")
                    smpl_outs_preds_f = self.smpl_model['female'](poses=select_smpl_poses[female_index],
                                                                  betas=select_smpl_betas[female_index],
                                                                  cams=None)
                    loss_j3d_f = self.loss_kp3d(smpl_outs_preds_f['j3d'], smpl_outs_gt_f['j3d'],
                                                avg_factor=num_valid_smpl_3d_female)
                loss_j3d = loss_j3d_n + loss_j3d_m + loss_j3d_f

        if self.with_kp:
            return loss_cls, loss_bbox, loss_iou, loss_attrs, loss_kpt_cls, loss_kpt, loss_oks, loss_mask, loss_dice, \
                kpt_preds, kpt_targets, area_targets, kpt_weights, loss_smpl_pose, loss_smpl_betas, loss_j3d, loss_pose_prior
        else:
            return loss_cls, loss_bbox, loss_iou, loss_attrs, loss_kpt_cls, loss_kpt, loss_oks, loss_mask, loss_dice, \
                None, None, None, None, loss_smpl_pose, loss_smpl_betas, loss_j3d, loss_pose_prior

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, dn_kpt_preds, dn_mask_preds, gt_bboxes_list,
                gt_labels_list, gt_kpts_list, gt_areas_list, gt_masks_list, img_metas, dn_meta):
        num_dec_layers = len(dn_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_kpts_list = [gt_kpts_list for _ in range(num_dec_layers)]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        all_gt_mask_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        if dn_bbox_preds is None:
            dn_bbox_preds = [None for _ in range(num_dec_layers)]
        if dn_kpt_preds is None:
            dn_kpt_preds = [None for _ in range(num_dec_layers)]
        if dn_mask_preds is None:
            dn_mask_preds = [None for _ in range(num_dec_layers)]
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds, dn_kpt_preds, dn_mask_preds,
                           all_gt_bboxes_list, all_gt_labels_list, all_gt_kpts_list, all_gt_areas_list,
                           all_gt_mask_list,
                           img_metas_list, dn_meta_list)

    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, dn_kpt_preds, dn_mask_preds, gt_bboxes_list,
                       gt_labels_list, gt_kpts_list, gt_areas_list, gt_masks_list, img_metas, dn_meta):
        num_imgs = dn_cls_scores.size(0)
        if dn_bbox_preds is not None:
            bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        else:
            bbox_preds_list = None
        if dn_kpt_preds is not None:
            kpt_preds_list = [dn_kpt_preds[i] for i in range(num_imgs)]
        else:
            kpt_preds_list = None
        if dn_mask_preds is not None:
            mask_preds_list = [dn_mask_preds[i] for i in range(num_imgs)]
        else:
            mask_preds_list = None
        cls_reg_targets = self.get_dn_target(bbox_preds_list, kpt_preds_list, mask_preds_list, gt_bboxes_list,
                                             gt_labels_list, gt_kpts_list, gt_areas_list, gt_masks_list, img_metas,
                                             dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         kpt_targets_list, kpt_weights_list, areas_list, mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        if self.with_det:
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)
        # if self.with_kp:
        #     kpt_targets = torch.cat(kpt_targets_list, 0)
        #     kpt_weights = torch.cat(kpt_weights_list, 0)
        #     area_targets = torch.cat(areas_list, 0)
        # if self.with_seg:
        #     mask_targets = torch.cat(mask_targets_list, 0)
        #     mask_weights = torch.stack(mask_weights_list, dim=0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(  # TODO: How to better return zero loss
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        if self.with_det:
            # DETR regress the relative position of boxes (cxcywh) in the image,
            # thus the learning target is normalized by the image size. So here
            # we need to re-scale them for calculating IoU loss
            bbox_preds = dn_bbox_preds.reshape(-1, 4)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

            # regression L1 loss
            loss_bbox = self.loss_bbox(
                bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        else:
            loss_iou = None
            loss_bbox = None

        return loss_cls, loss_bbox, loss_iou  # , loss_kpt, loss_oks, loss_mask, loss_dice

    def get_dn_target(self, dn_bbox_preds_list, dn_kpt_preds_list, dn_mask_preds_list, gt_bboxes_list, gt_labels_list,
                      gt_kpt_list, gt_area_list, gt_mask_list,
                      img_metas, dn_meta):
        if not self.with_det:
            dn_bbox_preds_list = [None for _ in range(len(gt_labels_list))]
        if not self.with_kp:
            dn_kpt_preds_list = [None for _ in range(len(gt_labels_list))]
        if not self.with_seg:
            dn_mask_preds_list = [None for _ in range(len(gt_labels_list))]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         kpt_list, kpt_weights_list, area_list, mask_list, mask_weight_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, dn_kpt_preds_list, dn_mask_preds_list,
                                      gt_bboxes_list,
                                      gt_labels_list, gt_kpt_list, gt_area_list, gt_mask_list,
                                      img_metas, dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list,
                kpt_list, kpt_weights_list, area_list, mask_list, mask_weight_list,
                num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, dn_kpt_pred, dn_mask_pred, gt_bboxes, gt_labels, gt_keypoints,
                              gt_areas, gt_masks,
                              img_meta, dn_meta):
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        assert pad_size % num_groups == 0
        single_pad = pad_size // num_groups
        num_bboxes = dn_bbox_pred.size(0)

        if len(gt_labels) > 0:
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()

        neg_inds = pos_inds + single_pad // 2
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # print("pos_inds:",pos_inds)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        if self.with_det:
            # bbox targets
            bbox_targets = torch.zeros_like(dn_bbox_pred)
            bbox_weights = torch.zeros_like(dn_bbox_pred)
            bbox_weights[pos_inds] = 1.0
            img_h, img_w, _ = img_meta['img_shape']

            # DETR regress the relative position of boxes (cxcywh) in the image.
            # Thus the learning target should be normalized by the image size, also
            # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
            factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                              img_h]).unsqueeze(0)
            gt_bboxes_normalized = gt_bboxes / factor
            gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
            bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])
        else:
            bbox_targets = None
            bbox_weights = None

        if self.with_kp:
            # keypoint targets
            kpt_targets = torch.zeros_like(dn_kpt_pred)
            kpt_weights = torch.zeros_like(dn_kpt_pred)
            pos_gt_kpts = gt_keypoints[pos_assigned_gt_inds]
            pos_gt_kpts = pos_gt_kpts.reshape(pos_gt_kpts.shape[0],
                                              pos_gt_kpts.shape[-1] // 3, 3)
            valid_idx = pos_gt_kpts[:, :, 2] > 0
            pos_kpt_weights = kpt_weights[pos_inds].reshape(
                pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2)
            pos_kpt_weights[valid_idx] = 1.0
            kpt_weights[pos_inds] = pos_kpt_weights.reshape(
                pos_kpt_weights.shape[0], dn_kpt_pred.shape[-1])

            kp_factor = dn_kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0)
            pos_gt_kpts_normalized = pos_gt_kpts[..., :2]
            pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
                                             kp_factor[:, 0:1]
            pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
                                             kp_factor[:, 1:2]
            kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
                pos_gt_kpts.shape[0], dn_kpt_pred.shape[-1])

            area_targets = dn_kpt_pred.new_zeros(
                dn_kpt_pred.shape[0])  # get areas for calculating oks
            pos_gt_areas = gt_areas[pos_assigned_gt_inds]
            area_targets[pos_inds] = pos_gt_areas
        else:
            kpt_targets = None
            kpt_weights = None
            area_targets = None
        if self.with_seg:
            mask_targets = gt_masks[pos_assigned_gt_inds]
            mask_weights = dn_mask_pred.new_zeros((num_bboxes,))
            # print("dn_mask_targets:", mask_targets.shape)
            # print("dn_mask_weights:",mask_weights.shape)
            mask_weights[pos_inds] = 1.0
        else:
            mask_targets = None
            mask_weights = None

        return (labels, label_weights, bbox_targets, bbox_weights,
                kpt_targets, kpt_weights, area_targets, mask_targets, mask_weights,
                pos_inds, neg_inds)

    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, all_attrs_preds, all_kpt_cls_scores, all_kps_preds,
                           all_mask_preds,
                           all_smpl_pose_preds, all_smpl_betas_preds,
                           dn_meta):  # , all_kps_score_preds, dn_meta):
        if dn_meta is not None:
            denoising_cls_scores = all_cls_scores[:, :, :dn_meta['pad_size'], :]
            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]

            if all_bbox_preds is not None:
                denoising_bbox_preds = all_bbox_preds[:, :, :dn_meta['pad_size'], :]
                matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
            else:
                denoising_bbox_preds = None
                matching_bbox_preds = None
            if all_kps_preds is not None:
                denoising_kpt_preds = all_kps_preds[:, :, :dn_meta['pad_size'], :]
                matching_kps_preds = all_kps_preds[:, :, dn_meta['pad_size']:, :]
                denoising_kpt_cls_scores = all_kpt_cls_scores[:, :, :dn_meta['pad_size'], :]
                matching_kpt_cls_scores = all_kpt_cls_scores[:, :, dn_meta['pad_size']:, :]
            else:
                denoising_kpt_preds = None
                matching_kps_preds = None
                denoising_kpt_cls_scores = None
                matching_kpt_cls_scores = None
            if all_mask_preds is not None:
                denoising_mask_preds = all_mask_preds[:, :, :dn_meta['pad_size'], :]
                matching_masks_preds = all_mask_preds[:, :, dn_meta['pad_size']:, :]
            else:
                denoising_mask_preds = None
                matching_masks_preds = None
            if all_attrs_preds is not None:
                matching_attrs_preds = all_attrs_preds[:, :, dn_meta['pad_size']:, :]
            else:
                matching_attrs_preds = None
            if all_smpl_pose_preds is not None:
                matching_smpl_pose_preds = all_smpl_pose_preds[:, :, dn_meta['pad_size']:, :]
            else:
                matching_smpl_pose_preds = None
            if all_smpl_betas_preds is not None:
                matching_smpl_betas_preds = all_smpl_betas_preds[:, :, dn_meta['pad_size']:, :]
            else:
                matching_smpl_betas_preds = None
        else:
            denoising_cls_scores = None
            denoising_bbox_preds = None
            denoising_kpt_preds = None
            denoising_mask_preds = None
            denoising_kpt_cls_scores = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
            matching_attrs_preds = all_attrs_preds
            matching_masks_preds = all_mask_preds
            matching_kps_preds = all_kps_preds
            matching_kpt_cls_scores = all_kpt_cls_scores
            matching_smpl_pose_preds = all_smpl_pose_preds
            matching_smpl_betas_preds = all_smpl_betas_preds
            # matching_smpl_cams_preds = all_smpl_cams_preds
            # matching_kps_scores_preds = all_kps_score_preds
        return (matching_cls_scores, matching_bbox_preds, denoising_cls_scores,
                denoising_bbox_preds, denoising_kpt_cls_scores, denoising_kpt_preds, denoising_mask_preds,
                matching_attrs_preds, matching_kpt_cls_scores, matching_kps_preds, matching_masks_preds,
                matching_smpl_pose_preds,
                matching_smpl_betas_preds)  # , matching_kps_preds) #, matching_kps_scores_preds)

    def loss_heatmap(self, hm_pred, hm_mask, gt_keypoints, gt_labels,
                     gt_bboxes):
        assert hm_pred.shape[-2:] == hm_mask.shape[-2:]
        num_img, _, h, w = hm_pred.size()
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = hm_pred.new_zeros(hm_pred.shape)
        for i, (gt_label, gt_bbox, gt_keypoint) in enumerate(
                zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.size(0) == 0:
                continue
            gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1,
                                              3).clone()
            gt_keypoint[..., :2] /= 8
            assert gt_keypoint[..., 0].max() <= w  # new coordinate system
            assert gt_keypoint[..., 1].max() <= h  # new coordinate system
            gt_bbox /= 8
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            for j in range(gt_label.size(0)):
                # get heatmap radius
                kp_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]), min_overlap=0.9)),
                    min=0, max=3)
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = torch.floor(gt_kp)
                        draw_umich_gaussian(hm_target[i, k], gt_kp_int,
                                            kp_radius)
        # compute heatmap loss
        hm_pred = torch.clamp(
            hm_pred.sigmoid_(), min=1e-4, max=1 - 1e-4)  # refer to CenterNet
        loss_hm = self.loss_hm(hm_pred, hm_target, mask=~hm_mask.unsqueeze(1))
        return loss_hm

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2):  # and self.__class__ is DETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        pass

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        pass
