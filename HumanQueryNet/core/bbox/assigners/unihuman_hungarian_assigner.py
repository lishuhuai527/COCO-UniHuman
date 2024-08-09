# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
from mmdet.core.bbox import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class UnihumanHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression oks cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt.
    - positive integer: positive sample, index (1-based) of assigned gt.

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        kpt_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        oks_weight (int | float, optional): The scale factor for regression
            oks cost. Default 1.0.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 kpt_cls_cost=None,
                 seg_cls_cost=None,
                 reg_cost=None,  # dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=None,  # dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 kpt_cost=None,  # dict(type='KptL1Cost', weight=1.0),
                 oks_cost=None,  # dict(type='OksCost', weight=1.0),
                 mask_cost=None,  # dict(type='FocalLossCost', weight=1.0, binary_input=True),
                 dice_cost=None,  # dict(type='DiceCost', weight=1.0),
                 smpl_pose_cost=None,
                 smpl_beta_cost=None,
                 smpl_3d_cost=None
                 ):
        self.cls_cost = build_match_cost(cls_cost)
        if reg_cost is not None:
            self.reg_cost = build_match_cost(reg_cost)
        if iou_cost is not None:
            self.iou_cost = build_match_cost(iou_cost)
        if kpt_cost is not None:
            self.kpt_cost = build_match_cost(kpt_cost)
        if oks_cost is not None:
            self.oks_cost = build_match_cost(oks_cost)
        if mask_cost is not None:
            self.mask_cost = build_match_cost(mask_cost)
        if dice_cost is not None:
            self.dice_cost = build_match_cost(dice_cost)
        if kpt_cls_cost is not None:
            self.kpt_cls_cost = build_match_cost(kpt_cls_cost)
        if seg_cls_cost is not None:
            self.seg_cls_cost = build_match_cost(seg_cls_cost)
        if smpl_pose_cost is not None:
            self.smpl_pose_cost = build_match_cost(smpl_pose_cost)
        if smpl_beta_cost is not None:
            self.smpl_beta_cost = build_match_cost(smpl_beta_cost)
        if smpl_3d_cost is not None:
            self.smpl_3d_cost = build_match_cost(smpl_3d_cost)

    def assign(self,
               cls_pred,
               bbox_pred,
               kpt_pred,
               mask_pred,
               gt_labels,
               gt_bboxes,
               gt_keypoints,
               gt_areas,
               gt_mask,
               img_meta,
               kpt_cls_pred=None,
               seg_cls_pred=None,
               gt_kpt_labels=None,
               gt_smpl_pose=None,
               gt_smpl_beta=None,
               gt_smpl_3d=None,
               smpl_pose_pred=None,
               smpl_beta_pred=None,
               smpl_3d_pred=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K*2].
            gt_labels (Tensor): Label of `gt_keypoints`, shape (num_gt,).
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v]. Shape [num_gt, K*3].
            gt_areas (Tensor): Ground truth mask areas, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_query = gt_labels.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_query,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_query,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_query == 0:
            # No ground truth or keypoints, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        if kpt_cls_pred is not None and hasattr(self, 'kpt_cls_cost'):
            kpt_cls_cost = self.kpt_cls_cost(kpt_cls_pred, gt_kpt_labels)
        else:
            kpt_cls_cost = 0

        if seg_cls_pred is not None and hasattr(self, 'seg_cls_cost'):
            seg_cls_cost = self.seg_cls_cost(seg_cls_pred, gt_labels)
        else:
            seg_cls_cost = 0

        reg_cost = 0
        iou_cost = 0
        if bbox_pred is not None:
            # regression L1 cost
            normalize_gt_bboxes = gt_bboxes / factor
            if hasattr(self, 'reg_cost'):
                reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
                # regression iou cost, defaultly giou is used in official DETR.
            bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            if hasattr(self, 'iou_cost'):
                iou_cost = self.iou_cost(bboxes, gt_bboxes)

        kpt_cost = 0
        oks_cost = 0
        # keypoint regression L1 cost
        if kpt_pred is not None:
            gt_keypoints_reshape = gt_keypoints.reshape(gt_keypoints.shape[0], -1,
                                                        3)
            valid_kpt_flag = gt_keypoints_reshape[..., -1]
            kpt_pred_tmp = kpt_pred.clone().detach().reshape(
                kpt_pred.shape[0], -1, 2)
            normalize_gt_keypoints = gt_keypoints_reshape[
                                     ..., :2] / factor[:, :2].unsqueeze(0)
            if hasattr(self, 'kpt_cost'):
                kpt_cost = self.kpt_cost(kpt_pred_tmp, normalize_gt_keypoints,
                                         valid_kpt_flag)
            # keypoint OKS cost
            kpt_pred_tmp = kpt_pred.clone().detach().reshape(
                kpt_pred.shape[0], -1, 2)
            kpt_pred_tmp = kpt_pred_tmp * factor[:, :2].unsqueeze(0)
            if hasattr(self, 'oks_cost'):
                oks_cost = self.oks_cost(kpt_pred_tmp, gt_keypoints_reshape[..., :2],
                                         valid_kpt_flag, gt_areas)

        # mask costs
        mask_cost = 0
        dice_cost = 0
        if mask_pred is not None:
            if hasattr(self, 'mask_cost'):
                if self.mask_cost.weight != 0:
                    # mask_pred shape = [num_query, h, w]
                    # gt_mask shape = [num_gt, h, w]
                    # mask_cost shape = [num_query, num_gt]
                    mask_cost = self.mask_cost(mask_pred, gt_mask)
            if hasattr(self, 'dice_cost'):
                if self.dice_cost.weight != 0:
                    dice_cost = self.dice_cost(mask_pred, gt_mask)

        smpl_pose_cost = 0
        smpl_beta_cost = 0
        smpl_3d_cost = 0
        if smpl_pose_pred is not None:
            if hasattr(self, 'smpl_pose_cost'):
                if self.smpl_pose_cost.weight != 0:
                    smpl_pose_cost = self.smpl_pose_cost(mask_pred, gt_mask)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost + kpt_cost + oks_cost + mask_cost + dice_cost + kpt_cls_cost + seg_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
