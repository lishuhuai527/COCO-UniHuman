# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST


# @MATCH_COST.register_module()
# class BBoxL1Cost:
#     """BBoxL1Cost.
#
#      Args:
#          weight (int | float, optional): loss_weight
#          box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
#
#      Examples:
#          >>> from core.bbox.match_costs.match_cost import BBoxL1Cost
#          >>> import torch
#          >>> self = BBoxL1Cost()
#          >>> bbox_pred = torch.rand(1, 4)
#          >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
#          >>> factor = torch.tensor([10, 8, 10, 8])
#          >>> self(bbox_pred, gt_bboxes, factor)
#          tensor([[1.6172, 1.6422]])
#     """
#
#     def __init__(self, weight=1., box_format='xyxy'):
#         self.weight = weight
#         assert box_format in ['xyxy', 'xywh']
#         self.box_format = box_format
#
#     def __call__(self, bbox_pred, gt_bboxes):
#         """
#         Args:
#             bbox_pred (Tensor): Predicted boxes with normalized coordinates
#                 (cx, cy, w, h), which are all in range [0, 1]. Shape
#                 (num_query, 4).
#             gt_bboxes (Tensor): Ground truth boxes with normalized
#                 coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
#
#         Returns:
#             torch.Tensor: bbox_cost value with weight
#         """
#         if self.box_format == 'xywh':
#             gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
#         elif self.box_format == 'xyxy':
#             bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
#         # gt_bboxes_has_nan = np.any(np.isnan(np.asarray(gt_bboxes.detach().cpu())))
#         # bbox_pred_has_nan = np.any(np.isnan(np.asarray(bbox_pred.detach().cpu())))
#         bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
#         iou_has_nan = np.any(np.isnan(np.asarray(bbox_cost.detach().cpu())))
#         # print("BBoxL1Cost:",iou_has_nan, gt_bboxes_has_nan, bbox_pred_has_nan)
#         return bbox_cost * self.weight


# @MATCH_COST.register_module()
# class FocalLossCost:
#     """FocalLossCost.
#
#      Args:
#          weight (int | float, optional): loss_weight
#          alpha (int | float, optional): focal_loss alpha
#          gamma (int | float, optional): focal_loss gamma
#          eps (float, optional): default 1e-12
#          binary_input (bool, optional): Whether the input is binary,
#             default False.
#
#      Examples:
#          >>> from core.bbox.match_costs.match_cost import FocalLossCost
#          >>> import torch
#          >>> self = FocalLossCost()
#          >>> cls_pred = torch.rand(4, 3)
#          >>> gt_labels = torch.tensor([0, 1, 2])
#          >>> factor = torch.tensor([10, 8, 10, 8])
#          >>> self(cls_pred, gt_labels)
#          tensor([[-0.3236, -0.3364, -0.2699],
#                 [-0.3439, -0.3209, -0.4807],
#                 [-0.4099, -0.3795, -0.2929],
#                 [-0.1950, -0.1207, -0.2626]])
#     """
#
#     def __init__(self,
#                  weight=1.,
#                  alpha=0.25,
#                  gamma=2,
#                  eps=1e-12,
#                  binary_input=False):
#         self.weight = weight
#         self.alpha = alpha
#         self.gamma = gamma
#         self.eps = eps
#         self.binary_input = binary_input
#
#     def _focal_loss_cost(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): Predicted classification logits, shape
#                 (num_query, num_class).
#             gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
#
#         Returns:
#             torch.Tensor: cls_cost value with weight
#         """
#         cls_pred = cls_pred.sigmoid()
#         neg_cost = -(1 - cls_pred + self.eps).log() * (
#             1 - self.alpha) * cls_pred.pow(self.gamma)
#         pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
#             1 - cls_pred).pow(self.gamma)
#
#         cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
#         # cls_pred_has_nan = np.any(np.isnan(np.asarray(cls_pred.detach().cpu())))
#         # pos_cost_has_nan = np.any(np.isnan(np.asarray(pos_cost.detach().cpu())))
#         # neg_cost_has_nan = np.any(np.isnan(np.asarray(neg_cost.detach().cpu())))
#         # iou_has_nan = np.any(np.isnan(np.asarray(cls_cost.detach().cpu())))
#         # print("FocalLossCost:",iou_has_nan, pos_cost_has_nan, neg_cost_has_nan, cls_pred_has_nan)
#         return cls_cost * self.weight
#
#     def _mask_focal_loss_cost(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): Predicted classfication logits
#                 in shape (num_query, d1, ..., dn), dtype=torch.float32.
#             gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
#                 dtype=torch.long. Labels should be binary.
#
#         Returns:
#             Tensor: Focal cost matrix with weight in shape\
#                 (num_query, num_gt).
#         """
#         cls_pred = cls_pred.flatten(1)
#         gt_labels = gt_labels.flatten(1).float()
#         n = cls_pred.shape[1]
#         cls_pred = cls_pred.sigmoid()
#         neg_cost = -(1 - cls_pred + self.eps).log() * (
#             1 - self.alpha) * cls_pred.pow(self.gamma)
#         pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
#             1 - cls_pred).pow(self.gamma)
#
#         cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
#             torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
#         return cls_cost / n * self.weight
#
#     def __call__(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): Predicted classfication logits.
#             gt_labels (Tensor)): Labels.
#
#         Returns:
#             Tensor: Focal cost matrix with weight in shape\
#                 (num_query, num_gt).
#         """
#         if self.binary_input:
#             return self._mask_focal_loss_cost(cls_pred, gt_labels)
#         else:
#             return self._focal_loss_cost(cls_pred, gt_labels)


@MATCH_COST.register_module()
class FocalLossCostV2:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12
         binary_input (bool, optional): Whether the input is binary,
            default False.

     Examples:
         >>> from core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        for i in range(len(gt_labels)):
            if gt_labels[i] != 0:
                cls_cost[:, i] = 0
        # cls_pred_has_nan = np.any(np.isnan(np.asarray(cls_pred.detach().cpu())))
        # pos_cost_has_nan = np.any(np.isnan(np.asarray(pos_cost.detach().cpu())))
        # neg_cost_has_nan = np.any(np.isnan(np.asarray(neg_cost.detach().cpu())))
        # iou_has_nan = np.any(np.isnan(np.asarray(cls_cost.detach().cpu())))
        # print("FocalLossCost:",iou_has_nan, pos_cost_has_nan, neg_cost_has_nan, cls_pred_has_nan)
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
                   torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)


# @MATCH_COST.register_module()
# class ClassificationCost:
#     """ClsSoftmaxCost.
#
#      Args:
#          weight (int | float, optional): loss_weight
#
#      Examples:
#          >>> from core.bbox.match_costs.match_cost import \
#          ... ClassificationCost
#          >>> import torch
#          >>> self = ClassificationCost()
#          >>> cls_pred = torch.rand(4, 3)
#          >>> gt_labels = torch.tensor([0, 1, 2])
#          >>> factor = torch.tensor([10, 8, 10, 8])
#          >>> self(cls_pred, gt_labels)
#          tensor([[-0.3430, -0.3525, -0.3045],
#                 [-0.3077, -0.2931, -0.3992],
#                 [-0.3664, -0.3455, -0.2881],
#                 [-0.3343, -0.2701, -0.3956]])
#     """
#
#     def __init__(self, weight=1.):
#         self.weight = weight
#
#     def __call__(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): Predicted classification logits, shape
#                 (num_query, num_class).
#             gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
#
#         Returns:
#             torch.Tensor: cls_cost value with weight
#         """
#         # Following the official DETR repo, contrary to the loss that
#         # NLL is used, we approximate it in 1 - cls_score[gt_label].
#         # The 1 is a constant that doesn't change the matching,
#         # so it can be omitted.
#         cls_score = cls_pred.softmax(-1)
#         cls_cost = -cls_score[:, gt_labels]
#         # iou_has_nan = np.any(np.isnan(np.asarray(cls_cost.detach().cpu())))
#         # print("ClassificationCost:",iou_has_nan)
#         return cls_cost * self.weight


# @MATCH_COST.register_module()
# class IoUCost:
#     """IoUCost.
#
#      Args:
#          iou_mode (str, optional): iou mode such as 'iou' | 'giou'
#          weight (int | float, optional): loss weight
#
#      Examples:
#          >>> from core.bbox.match_costs.match_cost import IoUCost
#          >>> import torch
#          >>> self = IoUCost()
#          >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
#          >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
#          >>> self(bboxes, gt_bboxes)
#          tensor([[-0.1250,  0.1667],
#                 [ 0.1667, -0.5000]])
#     """
#
#     def __init__(self, iou_mode='giou', weight=1.):
#         self.weight = weight
#         self.iou_mode = iou_mode
#
#     def __call__(self, bboxes, gt_bboxes):
#         """
#         Args:
#             bboxes (Tensor): Predicted boxes with unnormalized coordinates
#                 (x1, y1, x2, y2). Shape (num_query, 4).
#             gt_bboxes (Tensor): Ground truth boxes with unnormalized
#                 coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
#
#         Returns:
#             torch.Tensor: iou_cost value with weight
#         """
#         # overlaps: [num_bboxes, num_gt]
#         overlaps = bbox_overlaps(
#             bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
#         # The 1 is a constant that doesn't change the matching, so omitted.
#         iou_cost = -overlaps
#         # iou_has_nan = np.any(np.isnan(np.asarray(iou_cost.detach().cpu())))
#         # print("IoUCost:",iou_has_nan)
#         return iou_cost * self.weight
#

# @MATCH_COST.register_module()
# class DiceCost:
#     """Cost of mask assignments based on dice losses.
#
#     Args:
#         weight (int | float, optional): loss_weight. Defaults to 1.
#         pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
#             Defaults to False.
#         eps (float, optional): default 1e-12.
#         naive_dice (bool, optional): If True, use the naive dice loss
#             in which the power of the number in the denominator is
#             the first power. If Flase, use the second power that
#             is adopted by K-Net and SOLO.
#             Defaults to True.
#     """
#
#     def __init__(self, weight=1., pred_act=False, eps=1e-3, naive_dice=True):
#         self.weight = weight
#         self.pred_act = pred_act
#         self.eps = eps
#         self.naive_dice = naive_dice
#
#     def binary_mask_dice_loss(self, mask_preds, gt_masks):
#         """
#         Args:
#             mask_preds (Tensor): Mask prediction in shape (num_query, *).
#             gt_masks (Tensor): Ground truth in shape (num_gt, *)
#                 store 0 or 1, 0 for negative class and 1 for
#                 positive class.
#
#         Returns:
#             Tensor: Dice cost matrix in shape (num_query, num_gt).
#         """
#         mask_preds = mask_preds.flatten(1)
#         gt_masks = gt_masks.flatten(1).float()
#         numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
#         if self.naive_dice:
#             denominator = mask_preds.sum(-1)[:, None] + \
#                 gt_masks.sum(-1)[None, :]
#         else:
#             denominator = mask_preds.pow(2).sum(1)[:, None] + \
#                 gt_masks.pow(2).sum(1)[None, :]
#         loss = 1 - (numerator + self.eps) / (denominator + self.eps)
#         return loss
#
#     def __call__(self, mask_preds, gt_masks):
#         """
#         Args:
#             mask_preds (Tensor): Mask prediction logits in shape (num_query, *)
#             gt_masks (Tensor): Ground truth in shape (num_gt, *)
#
#         Returns:
#             Tensor: Dice cost matrix with weight in shape (num_query, num_gt).
#         """
#         if self.pred_act:
#             mask_preds = mask_preds.sigmoid()
#         dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
#         return dice_cost * self.weight


# @MATCH_COST.register_module()
# class CrossEntropyLossCost:
#     """CrossEntropyLossCost.
#
#     Args:
#         weight (int | float, optional): loss weight. Defaults to 1.
#         use_sigmoid (bool, optional): Whether the prediction uses sigmoid
#                 of softmax. Defaults to True.
#     Examples:
#          >>> from core.bbox.match_costs import CrossEntropyLossCost
#          >>> import torch
#          >>> bce = CrossEntropyLossCost(use_sigmoid=True)
#          >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
#          >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
#          >>> print(bce(cls_pred, gt_labels))
#     """
#
#     def __init__(self, weight=1., use_sigmoid=True):
#         assert use_sigmoid, 'use_sigmoid = False is not supported yet.'
#         self.weight = weight
#         self.use_sigmoid = use_sigmoid
#
#     def _binary_cross_entropy(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
#                 (num_query, *).
#             gt_labels (Tensor): The learning label of prediction with
#                 shape (num_gt, *).
#
#         Returns:
#             Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
#         """
#         cls_pred = cls_pred.flatten(1).float()
#         gt_labels = gt_labels.flatten(1).float()
#         n = cls_pred.shape[1]
#         pos = F.binary_cross_entropy_with_logits(
#             cls_pred, torch.ones_like(cls_pred), reduction='none')
#         neg = F.binary_cross_entropy_with_logits(
#             cls_pred, torch.zeros_like(cls_pred), reduction='none')
#         cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
#             torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
#         cls_cost = cls_cost / n
#
#         return cls_cost
#
#     def __call__(self, cls_pred, gt_labels):
#         """
#         Args:
#             cls_pred (Tensor): Predicted classification logits.
#             gt_labels (Tensor): Labels.
#
#         Returns:
#             Tensor: Cross entropy cost matrix with weight in
#                 shape (num_query, num_gt).
#         """
#         if self.use_sigmoid:
#             cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
#         else:
#             raise NotImplementedError
#
#         return cls_cost * self.weight

@MATCH_COST.register_module()
class KptL1Cost(object):
    """KptL1Cost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import KptL1Cost
        >>> import torch
        >>> self = KptL1Cost()
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            torch.Tensor: kpt_cost value with weight.
        """
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1),
                gt_keypoints[i].reshape(-1).unsqueeze(0),
                p=1)
            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class KptL1V2Cost(object):
    """KptL1Cost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import KptL1Cost
        >>> import torch
        >>> self = KptL1Cost()
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            torch.Tensor: kpt_cost value with weight.
        """
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1),
                gt_keypoints[i].reshape(-1).unsqueeze(0),
                p=1)
            # avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            # print("valid_flag:",valid_flag.shape)
            # print("kpt_pred_tmp:",kpt_pred_tmp.shape)
            avg_factor = valid_flag.float().sum() * 2
            # print("avg_factor:",avg_factor)
            if avg_factor == 0:
                cost = cost * 0.0
            else:
                cost = cost / avg_factor
            # print("cost:",cost.min(),cost.max(),cost.shape)
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class OksCost(object):
    """OksCost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import OksCost
        >>> import torch
        >>> self = OksCost()
    """

    def __init__(self, num_keypoints=17, weight=1.0):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26,
                .25, .25,
                .35, .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89,
                .79, .79], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            torch.Tensor: oks_cost value with weight.
        """
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2) ** 2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2
            vis_flag = (valid_kpt_flag[i] > 0).int()
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            assert num_vis_kpt > 0
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)
            squared_distance0 = squared_distance0[:, vis_ind]
            squared_distance1 = torch.exp(-squared_distance0).sum(
                dim=1, keepdim=True)
            oks = squared_distance1 / num_vis_kpt
            # The 1 is a constant that doesn't change the matching, so omitted.
            oks_cost.append(-oks)
        oks_cost = torch.cat(oks_cost, dim=1)
        return oks_cost * self.weight


@MATCH_COST.register_module()
class OksV4Cost(object):
    """OksCost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import OksCost
        >>> import torch
        >>> self = OksCost()
    """

    def __init__(self, num_keypoints=17, weight=1.0, eps=1e-6):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26,
                .25, .25,
                .35, .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89,
                .79, .79], dtype=np.float32) / 10.0
        elif num_keypoints == 45:
            self.sigmas = np.array([1.0 for i in range(45)], dtype=np.float32) / 10.0
        elif num_keypoints == 54:
            self.sigmas = np.array([1.0 for i in range(54)], dtype=np.float32) / 10.0
        elif num_keypoints == 4:
            self.sigmas = np.array([1.0 for i in range(4)], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')
        self.eps = eps

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            torch.Tensor: oks_cost value with weight.
        """
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2) ** 2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2
            vis_flag = (valid_kpt_flag[i] > 0).int()
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)

            squared_distance0 = squared_distance0.clamp(max=100)
            squared_distance1 = torch.exp(-squared_distance0)
            squared_distance1 = squared_distance1 * vis_flag
            squared_distance1 = squared_distance1.sum(dim=1, keepdim=True)
            if num_vis_kpt > 0:
                oks = squared_distance1 / num_vis_kpt
                oks = oks.clamp(min=self.eps)
                oks = (-oks.log())
            else:
                oks = squared_distance1 * 0.0

            oks_cost.append(oks)
        oks_cost = torch.cat(oks_cost, dim=1)
        return oks_cost * self.weight
