# Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.models.builder import DETECTORS
import numpy as np
import torch
from mmdet.core import bbox2result

from .dino import DINO
from ..builder import DETECTORS


@DETECTORS.register_module()
class HumanQueryNet(DINO):

    def __init__(self, *args, **kwargs):
        super(HumanQueryNet, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_attributes,
                      gt_keypoints,
                      gt_areas,
                      gt_masks,
                      gt_valids,
                      gt_smpl_pose=None,
                      gt_smpl_betas=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # This is a bug, super func should not be called here
        # super(MultiTaskDINO, self).forward_train(img, img_metas, gt_bboxes, gt_labels, *kwargs)
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        # import cv2
        # print(img.shape)
        # img0=img[0]
        # img1=img[1]
        # img0=img0.cpu().numpy().transpose(1,2,0)
        # img1=img1.cpu().numpy().transpose(1,2,0)
        # cv2.imshow("img0",img0)
        # cv2.imshow("img1", img1)
        # cv2.waitKey(0)
        # print("img.shape:", img.shape)
        # print("gt_masks:",gt_masks)
        # print(img_metas)
        # quit()
        # print("gt_smpl_params:",gt_smpl_params[0].shape)
        # print("gt_root_trans:",gt_root_trans[0].shape)
        # gt_smpl_pose=gt_smpl_params
        x = self.extract_feat(img)
        # for i in range(len(gt_labels)):
        #     print("gt_labels:",gt_labels[i].shape)
        #     print("gt_bboxes:",gt_bboxes[i].shape)
        #     print("gt_smpl_pose:",gt_smpl_pose[i].shape)
        #     print("gt_smpl_betas:",gt_smpl_betas[i].shape)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_attributes, gt_keypoints, gt_areas, gt_masks, gt_valids,
                                              gt_smpl_pose, gt_smpl_betas, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_dummy(self, img):
        n, c, h, w = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(h, w),
                scale_factor=[1, 1, 1, 1],
                ori_shape=(h, w, c),
                img_shape=(h, w, c)) for _ in range(n)
        ]
        feats = self.extract_feat(img)
        result = self.bbox_head.simple_test_bboxes(feats, img_metas=dummy_img_metas, rescale=False)
        return result

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        result = dict()
        if 'det' in results_list:
            bbox_results = [
                bbox2result(results_list['det'], results_list['label'], self.bbox_head.num_classes)
            ]
            result['det'] = bbox_results
        if 'seg' in results_list:
            result['ins_results'] = self.seg2result(results_list['seg'])
        return [result]

    def seg2result(self, seg_results):
        for i in range(len(seg_results)):
            if 'ins_results' in seg_results[i]:
                labels_per_image, bboxes, mask_pred_binary = seg_results[i]['ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image,
                                           self.bbox_head.num_classes)
                mask_results = [[] for _ in range(self.bbox_head.num_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                seg_results[i]['ins_results'] = bbox_results, mask_results
        result = [res['ins_results'] for res in seg_results]
        return result
