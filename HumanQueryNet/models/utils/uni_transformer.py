# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence, TransformerLayerSequence)
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_

# from .petr_transformer import MultiScaleDeformablePoseAttention
from .ms_pose_attn import MultiScaleDeformablePoseAttention
from .transformer import DeformableDetrTransformerDecoder, build_MLP, inverse_sigmoid

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn import (build_norm_layer, xavier_init)


@TRANSFORMER.register_module()
class UniTransformer(BaseModule):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 out_channels=256,
                 num_outs=3,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 encoder=None,
                 decoder=None,
                 hm_encoder=None,
                 kpt_refine_decoder=None,
                 num_keypoints=17,
                 with_kpt_refine=False,
                 init_cfg=None,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300):
        super(UniTransformer, self).__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.with_kpt_refine = False
        if kpt_refine_decoder is not None:
            self.kpt_refine_decoder = build_transformer_layer_sequence(kpt_refine_decoder)
            self.with_kpt_refine = True
        if hm_encoder is not None:
            self.hm_encoder = build_transformer_layer_sequence(hm_encoder)
        self.embed_dims = self.encoder.embed_dims
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        # kp transformer
        self.num_keypoints = num_keypoints

        self.in_channels = in_channels
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder.transformerlayers.attn_cfgs.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        self.strides = strides
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        if self.with_kpt_refine:
            self.refine_query_embedding = nn.Embedding(self.num_keypoints,
                                                       self.embed_dims * 2)
        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = self.norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(1):
            lateral_conv = ConvModule(
                self.in_channels[i],
                self.feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        self.mask_feature = nn.Conv2d(
            self.feat_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # It is actually content query, which is named tgt in other
        # DETR-like models

    def init_weights(self):
        # super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
            if isinstance(m, MultiScaleDeformablePoseAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        if self.with_kpt_refine:
            normal_(self.refine_query_embedding.weight)
        # nn.init.normal_(self.query_embed.weight.data)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_det_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        det_proposals = []
        kp_proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            det_proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            # print("uni proposal:", det_proposal.shape)
            kp_proposal = grid.view(N, -1, 2)
            det_proposals.append(det_proposal)
            kp_proposals.append(kp_proposal)
            _cur += (H * W)
        output_det_proposals = torch.cat(det_proposals, 1)
        output_det_proposals_valid = ((output_det_proposals > 0.01) &
                                      (output_det_proposals < 0.99)).all(
            -1, keepdim=True)
        output_det_proposals = torch.log(output_det_proposals / (1 - output_det_proposals))
        output_det_proposals = output_det_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_det_proposals = output_det_proposals.masked_fill(
            ~output_det_proposals_valid, float('inf'))

        output_kpt_proposals = torch.cat(kp_proposals, 1)
        output_kpt_proposals_valid = ((output_kpt_proposals > 0.01) &
                                      (output_kpt_proposals < 0.99)).all(
            -1, keepdim=True)
        output_kpt_proposals = torch.log(output_kpt_proposals / (1 - output_kpt_proposals))
        output_kpt_proposals = output_kpt_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_kpt_proposals = output_kpt_proposals.masked_fill(
            ~output_kpt_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_det_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_det_proposals, output_kpt_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                query_pos=None,
                reg_branches=None,
                cls_branches=None,
                kpt_cls_branches=None,
                attr_branches=None,
                kpt_branches=None,
                mask_branches=None,
                smpl_pose_branches=None,
                smpl_betas_branches=None,
                smpl_cams_branches=None,
                det_query_transformer=None,
                kpt_query_transformer=None,
                task_specific_query_mode="None",
                **kwargs):
        # assert self.as_two_stage and query_embed is None, \
        #     'as_two_stage must be True for DINO'
        assert self.as_two_stage, \
            'as_two_stage must be True for DINO'
        batch_size = mlvl_feats[0].shape[0]
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        mlvl_feats_ = mlvl_feats[1:]
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats_, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            # print("det_feat:",feat.shape)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        # get mask pixel map
        # from [query_num, 1, query_dim] to [1, query_dim, query_num]
        memory_pixel_map = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory_pixel_map, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        x = mlvl_feats[0]
        cur_feat = self.lateral_convs[0](x)
        y = cur_feat + F.interpolate(
            outs[0],
            size=cur_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        y = self.output_convs[0](y)

        # mask_feature = self.mask_feature(outs[-1])
        mask_feature = self.mask_feature(y)
        # from [query_num, 1, query_dim] to [1, query_num, query_dim]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        output_memory, output_det_proposals, output_kpt_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
        enc_output_detcls = enc_outputs_class[..., :1]

        enc_outputs_attrs = []
        if attr_branches is not None:
            for attr_branch in attr_branches:
                enc_outputs_attrs.append(attr_branch[self.decoder.num_layers](output_memory))
        if reg_branches is not None:
            enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](output_memory)
            enc_outputs_bbox_coord_unact = enc_outputs_coord_unact[..., :4] + output_det_proposals
        if kpt_branches is not None:
            enc_outputs_kpt_unact = kpt_branches[self.decoder.num_layers](output_memory)
            enc_outputs_kpt_unact[..., 0::2] += output_kpt_proposals[..., 0:1]
            enc_outputs_kpt_unact[..., 1::2] += output_kpt_proposals[..., 1:2]
            # add kpt cls
            if kpt_cls_branches is not None:
                enc_outputs_kpt_class = kpt_cls_branches[self.decoder.num_layers](output_memory)
                enc_output_kptcls = enc_outputs_kpt_class[..., :1]
        if mask_branches is not None:
            enc_outputs_mask_query_unact = mask_branches[self.decoder.num_layers](output_memory)
        if smpl_pose_branches is not None:
            enc_outputs_smpl_pose_unact = smpl_pose_branches[self.decoder.num_layers](output_memory)
        if smpl_betas_branches is not None:
            enc_outputs_smpl_shape_unact = smpl_betas_branches[self.decoder.num_layers](output_memory)
        if smpl_cams_branches is not None:
            enc_outputs_smpl_cam_unact = smpl_cams_branches[self.decoder.num_layers](output_memory)
        topk = self.two_stage_num_proposals
        topk_indices = torch.topk(enc_output_detcls.max(-1)[0], topk, dim=1)[1]
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO

        # topk_proposals = torch.topk(
        #     enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_score = torch.gather(enc_output_detcls, 1, topk_indices.unsqueeze(-1))  # .repeat(1, 1, cls_out_features))

        topk_attrs = []
        if attr_branches is not None:
            for i in range(len(attr_branches)):
                topk_attrs.append(
                    torch.gather(
                        enc_outputs_attrs[i], 1,
                        topk_indices.unsqueeze(-1).repeat(1, 1, attr_branches[i][self.decoder.num_layers].out_features))
                )
            topk_attrs = torch.cat(topk_attrs, -1)
        if reg_branches is not None:
            topk_coords_unact = torch.gather(
                enc_outputs_bbox_coord_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_anchor = topk_coords_unact.sigmoid()
            # print("topk_anchor:",topk_anchor.shape)
            topk_coords_unact = topk_coords_unact.detach()
        else:
            topk_anchor = None
            topk_coords_unact = None

        if mask_branches is not None:
            # print("enc_outputs_mask_query_unact:",enc_outputs_mask_query_unact.shape)
            mask_topk_indices = topk_indices.unsqueeze(-1).repeat(1, 1, enc_outputs_mask_query_unact.size(-1))
            # print("mask_topk_indices:",mask_topk_indices.shape)
            topk_mask_queries_unact = torch.gather(enc_outputs_mask_query_unact, 1, mask_topk_indices).detach()
            # print("topk_masks_unact:",topk_mask_queries_unact.shape)
            # quit()
        else:
            topk_mask_queries_unact = None
        if kpt_branches is not None:
            topk_kpts_unact = torch.gather(
                enc_outputs_kpt_unact, 1,
                topk_indices.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_kpt_unact.size(-1)))
            topk_kpt = topk_kpts_unact.sigmoid()
            if kpt_cls_branches is not None:
                topk_kpt_score = torch.gather(enc_output_kptcls, 1, topk_indices.unsqueeze(-1))
            else:
                topk_kpt_score = None
            # topk_kpts_unact = topk_kpts_unact.detach()
            #
            # kpt_reference_points = topk_kpts_unact.sigmoid()
            # kpt_init_reference_out = kpt_reference_points
        else:
            topk_kpt = None
            topk_kpt_score = None

        if smpl_pose_branches is not None:
            topk_smpl_pose_unact = torch.gather(
                enc_outputs_smpl_pose_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 144))
        else:
            topk_smpl_pose_unact = None
        if smpl_betas_branches is not None:
            topk_smpl_shape_unact = torch.gather(
                enc_outputs_smpl_shape_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 10))
        else:
            topk_smpl_shape_unact = None
        if smpl_cams_branches is not None:
            topk_smpl_cam_unact = torch.gather(
                enc_outputs_smpl_cam_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 3))
        else:
            topk_smpl_cam_unact = None
        if hasattr(query_embed, 'weight'):
            query = query_embed.weight
        else:
            query = query_embed
        # if task_specific_query_mode=="FC":
        #     query = det_query_transformer(query)
        # elif task_specific_query_mode == "conv":
        #     query_tmp = query.unsqueeze(-1).unsqueeze(-1)
        #     query = det_query_transformer(query_tmp).squeeze()
        query = query[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            det_query = torch.cat([dn_label_query, query], dim=1)
        else:
            det_query = query
        if dn_bbox_query is not None:
            det_decoder_reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                                     dim=1)
        else:
            det_decoder_reference_points = topk_coords_unact
        det_decoder_reference_points = det_decoder_reference_points.sigmoid()
        # det decoder
        det_query = det_query.permute(1, 0, 2)
        # from [1, query_num, query_dim] to [query_num, 1, query_dim]
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=det_query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=det_decoder_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)
        # print("det_inter_states:",inter_states.shape)
        # for i in inter_references:
        #     print("det_inter_reference:",i.shape)
        # kpt decoder
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        hm_proto = None
        if self.training and hasattr(self, "hm_encoder"):
            hm_memory = memory[
                        :, level_start_index[0]:level_start_index[1], :]
            hm_pos_embed = lvl_pos_embed_flatten[
                           level_start_index[0]:level_start_index[1], :, :]
            hm_mask = mask_flatten[
                      :, level_start_index[0]:level_start_index[1]]
            hm_reference_points = reference_points[
                                  :, level_start_index[0]:level_start_index[1], [0], :]
            hm_memory = hm_memory.permute(1, 0, 2)
            # print("det_hm_memory:",hm_memory.shape)
            hm_memory = self.hm_encoder(
                query=hm_memory,
                key=None,
                value=None,
                query_pos=hm_pos_embed,  # TODO fix pose to pos
                query_key_padding_mask=hm_mask,
                spatial_shapes=spatial_shapes[[0]],
                reference_points=hm_reference_points,
                level_start_index=level_start_index[0],
                valid_ratios=valid_ratios[:, [0], :],
                **kwargs)
            hm_memory = hm_memory.permute(1, 0, 2).reshape(bs,
                                                           spatial_shapes[0, 0], spatial_shapes[0, 1], -1)
            hm_proto = (hm_memory, mlvl_masks[0])

        # query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)

        memory = memory.permute(1, 0, 2)

        inter_references_out = inter_references
        multi_scale_features = None
        return inter_states, inter_references_out, topk_score, topk_kpt_score, topk_anchor, topk_attrs, topk_kpt, topk_mask_queries_unact, \
            topk_smpl_pose_unact, topk_smpl_shape_unact, topk_smpl_cam_unact, \
            hm_proto, memory, mask_feature, multi_scale_features
        # kpt_inter_states, kpt_init_reference_out, kpt_inter_references, \

    def forward_kpt_refine(self,
                           mlvl_masks,
                           memory,
                           reference_points_pose,
                           img_inds,
                           kpt_branches=None,
                           **kwargs):
        mask_flatten = []
        spatial_shapes = []
        for lvl, mask in enumerate(mlvl_masks):
            bs, h, w = mask.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mask_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # pose refinement (17 queries corresponding to 17 keypoints)
        # learnable query and query_pos
        refine_query_embedding = self.refine_query_embedding.weight
        query_pos, query = torch.split(
            refine_query_embedding, refine_query_embedding.size(1) // 2, dim=1)
        pos_num = reference_points_pose.size(0)
        query_pos = query_pos.unsqueeze(0).expand(pos_num, -1, -1)
        query = query.unsqueeze(0).expand(pos_num, -1, -1)
        reference_points = reference_points_pose.reshape(
            pos_num,
            reference_points_pose.size(1) // 2, 2)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        pos_memory = memory[:, img_inds, :]
        mask_flatten = mask_flatten[img_inds, :]
        valid_ratios = valid_ratios[img_inds, ...]
        inter_states, inter_references = self.kpt_refine_decoder(
            query=query,
            key=None,
            value=pos_memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=kpt_branches,
            **kwargs)
        # [num_decoder, num_query, bs, embed_dim]
        init_reference_out = reference_points
        return inter_states, init_reference_out, inter_references


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UnihumanTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(UnihumanTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # print(self.layers)
        x = super(UnihumanTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, num_feat, **kwargs):
        super(UniTransformerDecoder, self).__init__(*args, **kwargs)
        self.num_feat = num_feat
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, num_feat):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / num_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.num_feat)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            # print("reference_points_input:",reference_points_input.shape)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)[..., :4]
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
