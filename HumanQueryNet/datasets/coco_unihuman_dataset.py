# Copyright (c) OpenMMLab. All rights reserved.

import contextlib
import io
import logging
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCOeval

from datasets.api_wrappers import AttrCOCOeval
from datasets.api_wrappers.eval_3D import py_3d_eval
from .builder import DATASETS


@DATASETS.register_module()
class CocoUnihumanDataset(CocoDataset):
    CLASSES = ('person',)

    PALETTE = [(220, 20, 60)]
    FLIP_PAIRS = [[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12],
                  [13, 14],
                  [15, 16]]

    def __init__(self,
                 ann_file,
                 pipeline,
                 num_keypoints_2d=17,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix, None, '.png',
                         proposal_file, test_mode, filter_empty_gt,
                         file_client_args
                         )
        self.keys = {'labels': 0, 'bboxes': 1, 'keypoints_2d': 2, 'keypoints_3d': 3,
                     'mask': 4, 'smpl': 5, 'gender': 6, 'age': 7}
        self.num_keypoints_2d = num_keypoints_2d

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        ann_info_parsed = self._parse_ann_info(self.data_infos[idx], ann_info)

        return ann_info_parsed

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_attributes = []
        gt_kp2ds = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_areas = []
        gt_smpl_pose = []
        gt_smpl_betas = []
        gt_valid = []
        # gt_smpl_global_orient = []
        # gt_smpl_transl = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            valid = ann['valid']
            list_valid = [0 for i in range(len(valid))]
            # valid_define = {'labels': 0, 'bboxes': 0, 'keypoints_2d': 0, 'keypoints_3d': 0,
            #                 'mask': 0, 'smpl': 0, 'gender': 0, 'age': 0}
            if valid['bboxes']:
                x1, y1, w, h = ann['bbox']
                list_valid[0] = 1
                list_valid[1] = 1
            else:
                continue
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if valid['keypoints_2d']:
                kp2d = ann['keypoints']
                list_valid[2] = 1
                # print("kp2d:",kp2d)
                # print("kp2d:", len(kp2d))
            else:
                kp2d = [0 for i in range(self.num_keypoints_2d * 3)]
            # if valid['keypoints_3d']:
            #     kp3d = ann['keypoints_3d']
            #     list_valid[3] = 1
            #     # print("kp3d:",kp3d)
            #     # print("kp3d0:", len(kp3d))
            # else:
            #     kp3d=[0 for i in range(self.num_keypoints_3d*3)]
            # print("kp3d1:", len(kp3d))
            if valid['gender']:
                list_valid[6] = 1
                gender = ann['gender']
            else:
                gender = -1
            if valid['age']:
                list_valid[7] = 1
                age = ann['age']
            else:
                age = -1
            if valid['smpl']:
                list_valid[5] = 1
                smpl_pose = ann['smpl']['pose']
                smpl_betas = ann['smpl']['betas']
            else:
                smpl_pose = [0 for i in range(72)]
                smpl_betas = [0 for i in range(10)]

            if valid['segmentation']:
                segmentation = ann['segmentation']
                list_valid[4] = 1
            else:
                segmentation = None
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_attributes.append([gender, age])
                gt_kp2ds.append(kp2d)
                gt_masks_ann.append(segmentation)
                gt_areas.append(ann['area'])
                gt_smpl_pose.append(smpl_pose)
                gt_smpl_betas.append(smpl_betas)
                gt_valid.append(list_valid)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_valid = np.array(gt_valid, dtype=np.int64)
            gt_attributes = np.array(gt_attributes, dtype=np.int64)
            gt_kp2ds = np.array(gt_kp2ds, dtype=np.float32)
            gt_areas = np.array(gt_areas, dtype=np.float32)
            gt_smpl_pose = np.array(gt_smpl_pose, dtype=np.float32)
            gt_smpl_betas = np.array(gt_smpl_betas, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_valid = np.array([0 for i in self.keys.keys()], dtype=np.int64)
            gt_kp2ds = np.zeros((0, self.num_keypoints_2d * 3), dtype=np.float32)
            gt_areas = np.array(gt_areas, dtype=np.float32)
            gt_smpl_pose = np.zeros((0, 72), dtype=np.float32)
            gt_smpl_betas = np.zeros((0, 10), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # seg_map = img_info['filename'].replace('jpg', 'png')
        # import cv2
        # print(img_info['filename'])
        # img=cv2.imread("/data/datasets/val2017/"+img_info['filename'])
        # for i in gt_bboxes:
        #     x0,y0,x1,y1=i
        #     cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,0,255),2)
        # for i in gt_kps:
        #     for j in range(17):
        #         cv2.circle(img, (int(i[j*3]),int(i[j*3+1])),2, (0, 0, 255), 2)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # quit()
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            attributes=gt_attributes,
            keypoints=gt_kp2ds,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            smpl_pose=gt_smpl_pose,
            smpl_betas=gt_smpl_betas,
            valids=gt_valid,
            areas=gt_areas,
            # seg_map=seg_map,
            flip_pairs=self.FLIP_PAIRS)
        return ann

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['keypoint_fields'] = []
        results['area_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_det_results = []
        json_kpt_results = []
        json_segm_result = []
        SMPL54toLSP = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            if 'det' in results[idx]:
                result = results[idx]['det']
                for label in range(len(result)):
                    bboxes = result[label][0]
                    for i in range(bboxes.shape[0]):
                        # print("i:",i)
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = self.cat_ids[label]
                        data['gender'] = float(bboxes[i][5])
                        data['age'] = float(bboxes[i][6])
                        json_det_results.append(data)
                        if len(bboxes[i]) < 10:
                            continue

                        kpt_data = dict()
                        kpt_data['image_id'] = img_id
                        kpt = bboxes[i][0 - (self.num_keypoints_2d * 2):]
                        new_kpt = [1 for i in range(self.num_keypoints_2d * 3)]
                        for kp_idx in range(self.num_keypoints_2d):
                            new_kpt[kp_idx * 3] = kpt[kp_idx * 2]
                            new_kpt[kp_idx * 3 + 1] = kpt[kp_idx * 2 + 1]
                        new_kpt_ = np.array(new_kpt)
                        # new_kpt_=new_kpt_.reshape((-1,3))
                        # new_kpt_14=new_kpt_[SMPL54toLSP]
                        # new_kpt_14=new_kpt_14.reshape((-1))
                        # new_kpt_14=new_kpt_14.tolist()
                        kpt_data['keypoints'] = new_kpt  # _14  # kpt.tolist()
                        if len(bboxes[i]) == 4 + 1 + 1 + 1 + 1 + self.num_keypoints_2d * 2:
                            # 4 bbox 1 score 1 gender 1 age 1 kpt score 34 kpt
                            kpt_data['score'] = float(bboxes[i][7])
                        else:
                            kpt_data['score'] = float(bboxes[i][4])
                        kpt_data['category_id'] = self.cat_ids[label]
                        kpt = kpt.reshape(-1, 2)
                        area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (
                                np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                        kpt_data['area'] = area
                        json_kpt_results.append(kpt_data)

            if 'ins_results' in results[idx]:
                det, seg = results[idx]['ins_results']
                for label in range(len(det)):
                    bboxes = det[label]
                    if isinstance(seg, tuple):
                        segms = seg[0][label]
                        mask_score = seg[1][label]
                    else:
                        segms = seg[label]
                        mask_score = [bbox[4] for bbox in bboxes]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(mask_score[i])
                        data['category_id'] = self.cat_ids[label]
                        if isinstance(segms[i]['counts'], bytes):
                            segms[i]['counts'] = segms[i]['counts'].decode()
                        data['segmentation'] = segms[i]
                        json_segm_result.append(data)

        return json_det_results, json_kpt_results, json_segm_result

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            result_files['keypoints'] = f'{outfile_prefix}.keypoints.json'
            mmcv.dump(json_results[1], result_files['keypoints'])
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[2], result_files['segm'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['keypoints'])
            mmcv.dump(json_results[2], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        elif isinstance(results[0], dict):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            result_files['keypoints'] = f'{outfile_prefix}.keypoints.json'
            mmcv.dump(json_results[1], result_files['keypoints'])
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[2], result_files['segm'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def evaluate_kp(self, result_files, iou_thrs=None):
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        cocoGt = self.coco

        predictions = mmcv.load(result_files['keypoints'])
        if len(predictions) == 0:
            print("No kp in the result")
            return
        cocoDt = cocoGt.loadRes(predictions)

        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        # cocoEval.params.kpt_oks_sigmas = np.array(
        #    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79])/10.0
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        # if metric != 'keypoints':
        #     cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_m': 3,
            'mAP_l': 4,
            'AR@100': 5,
            'AR@300': 6,
            'AR@1000': 7,
            'AR_s@1000': 8,
            'AR_m@1000': 9,
            'AR_l@1000': 10
        }
        cocoEval.evaluate()
        cocoEval.accumulate()

        # Save coco summarize print information to logger
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        print_log('\n' + redirect_string.getvalue(), logger=None)

    def evaluate_attr(self,
                      results,
                      result_files,
                      coco_gt,
                      metrics,
                      logger=None,
                      classwise=False,
                      proposal_nums=(100, 300, 1000),
                      iou_thrs=None,
                      metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            iou_type = 'bbox'
            # if metric not in result_files:
            #     raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files['bbox'])
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEvalAttr = AttrCOCOeval(coco_gt, coco_det, iou_type, metric)
            cocoEvalAttr.params.catIds = self.cat_ids
            cocoEvalAttr.params.imgIds = self.img_ids
            cocoEvalAttr.params.maxDets = list(proposal_nums)
            cocoEvalAttr.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            cocoEvalAttr.evaluate()
            cocoEvalAttr.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEvalAttr.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEvalAttr.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEvalAttr.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
        return eval_results

    def evaluate(self,
                 results,
                 metric=['bbox', 'kpt', 'attribute'],
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'kpt', 'attribute', '3D', 'all']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if metrics[0] == 'all':
            metrics = ['bbox', 'attribute', 'kpt', 'segm', '3D']
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}
        coco_gt = self.coco
        if 'attribute' in metrics:
            eval_attribute_results = self.evaluate_attr(results, result_files,
                                                        coco_gt, ['age'], logger,
                                                        classwise, **kwargs)
            eval_results.update(eval_attribute_results)

            eval_attribute_results = self.evaluate_attr(results, result_files,
                                                        coco_gt, ['gender'], logger,
                                                        classwise, **kwargs)
            eval_results.update(eval_attribute_results)
            metrics.remove('attribute')

        if 'kpt' in metrics:
            self.evaluate_kp(result_files)
            metrics.remove('kpt')

        if '3D' in metrics:
            eval_3d_result = py_3d_eval(results, coco_gt, "models/smpl/models/SMPL_NEUTRAL.pth")
            metrics.remove('3D')
            eval_results.update(eval_3d_result)
            # name_value_tuples = []
            # for _metric in metrics:
            #     if _metric == 'mpjpe':
            #         _nv_tuples = self._report_mpjpe(result_files)
            #     elif _metric == 'pa-mpjpe':
            #         _nv_tuples = self._report_mpjpe(result_files, metric='pa-mpjpe')
            #     elif _metric == '3dpck':
            #         _nv_tuples = self._report_3d_pck(result_files)
            #     elif _metric == 'pa-3dpck':
            #         _nv_tuples = self._report_3d_pck(result_files, metric='pa-3dpck')
            #     elif _metric == '3dauc':
            #         _nv_tuples = self._report_3d_auc(result_files)
            #     elif _metric == 'pa-3dauc':
            #         _nv_tuples = self._report_3d_auc(result_files, metric='pa-3dauc')
            #     elif _metric == 'pve':
            #         _nv_tuples = self._report_pve(result_files)
            #     elif _metric == 'ihmr':
            #         _nv_tuples = self._report_ihmr(result_files)
            #     else:
            #         raise NotImplementedError
            #     name_value_tuples.extend(_nv_tuples)
            #
            # name_value = OrderedDict(name_value_tuples)

        if (('bbox' in metrics) or ('segm' in metrics)
                or ('proposal' in metrics)):
            self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
            eval_ins_results = self.evaluate_det_segm(results, result_files,
                                                      coco_gt, metrics, logger,
                                                      classwise, **kwargs)
            eval_results.update(eval_ins_results)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        # print(eval_results)
        return eval_results
