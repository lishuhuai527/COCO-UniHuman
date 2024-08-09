import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, RandomCrop, Resize, RandomFlip
from mmdet.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module()
class LoadMultitaskInstanceAnnotations(LoadAnnotations):
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 extra_anno_list=[],
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        super(LoadMultitaskInstanceAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            denorm_bbox=denorm_bbox,
            file_client_args=file_client_args)
        self.extra_anno_list = extra_anno_list

    def _load_extra_anno(self, results):
        for i in self.extra_anno_list:
            results['gt_' + i] = results['ann_info'][i].copy()
            if i == "keypoints":
                results['keypoint_fields'].append('gt_keypoints')
            if i == "areas":
                results['area_fields'].append('gt_areas')
            if i == 'face_bboxes':
                results['bbox_fields'].append('gt_face_bboxes')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        # if self.with_seg or self.with_mask:
        #     results = self._load_masks_and_semantic_segs(results)
        # results = self._load_semantic_seg(results)
        self._load_extra_anno(results)

        return results


@PIPELINES.register_module()
class UniRandomCrop(RandomCrop):
    """Random crop the image & bboxes & masks & keypoints & mask areas.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        kpt_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - The keys for bboxes, keypoints and areas must be aligned. That is,
          `gt_bboxes` corresponds to `gt_keypoints` and `gt_areas`, and
          `gt_bboxes_ignore` corresponds to `gt_keypoints_ignore` and
          `gt_areas_ignore`.
    """

    def __init__(self,
                 *args,
                 kpt_clip_border=True,
                 crop_list=['gt_keypoints', 'gt_attributes', 'gt_areas', 'gt_labels', 'gt_masks'],
                 **kwargs):
        super(UniRandomCrop, self).__init__(*args, **kwargs)
        self.kpt_clip_border = kpt_clip_border
        # The key correspondence from bboxes to kpts and areas.
        self.bbox2others = {
            'gt_bboxes': crop_list,
            'gt_bboxes_ignore': ['gt_keypoints_ignore']
        }
        # self.bbox2area = {
        #     'gt_bboxes': 'gt_areas',
        #     'gt_bboxes_ignore': 'gt_areas_ignore'
        # }
        # self.bbox2attr = {
        #     'gt_bboxes': 'gt_attributes',
        #     'gt_bboxes_ignore': ''
        # }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps, keypoints, mask areas.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # for k,v in results.items():
        #     print("crop:",k)
        # quit()
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bbox_area_clip_before = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                bbox_area_clip_after = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                if key == 'gt_bboxes' and 'gt_areas' in results:
                    clip_ratio = bbox_area_clip_after / bbox_area_clip_before
                    results['gt_areas'] = results['gt_areas'] * clip_ratio
                    results['gt_areas'] = np.clip(results['gt_areas'], a_min=1.0, a_max=1e8)

            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            label_key = self.bbox2others.get(key)
            for key in label_key:
                if key in results:
                    if key == "gt_dense_pose":
                        # print("key:", key)
                        # print("results[key]:", results[key])
                        # num_instance = len(results['gt_dense_pose'])
                        valid_gt_dense_pose = []
                        # print("valid_inds:",valid_inds)
                        # idx=valid_inds.index(True)
                        for i in range(len(valid_inds)):
                            if valid_inds[i]:
                                valid_gt_dense_pose.append(results['gt_dense_pose'][i])
                            # num_dp = len(results['gt_dense_pose'][i])
                            # for j in range(num_dp):
                        results['gt_dense_pose'] = valid_gt_dense_pose
                    else:
                        results[key] = results[key][valid_inds]

            # label fields. e.g. gt_labels and gt_labels_ignore
            # label_key = self.bbox2label.get(key)
            # if label_key in results:
            #     results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if 0:  # mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

            # keypoint fields, e.g. gt_keypoints
        # crop masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].crop(
                np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # crop keypoints accordingly and clip to the image boundary

        for key in results.get('keypoint_fields', []):
            # e.g. gt_keypoints
            if len(results[key]) > 0:
                kpt_offset = np.array([offset_w, offset_h], dtype=np.float32)
                keypoints = results[key].copy()
                keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
                keypoints[..., :2] = keypoints[..., :2] - kpt_offset
                invalid_idx = \
                    (keypoints[..., 0] < 0).astype(np.int8) | \
                    (keypoints[..., 1] < 0).astype(np.int8) | \
                    (keypoints[..., 0] > img_shape[1]).astype(np.int8) | \
                    (keypoints[..., 1] > img_shape[0]).astype(np.int8) | \
                    (keypoints[..., 2] < 0.1).astype(np.int8)
                assert key == 'gt_keypoints'
                gt_valid = ~invalid_idx.all(1)
                # results['gt_bboxes'] = results['gt_bboxes'][gt_valid]
                # results['gt_areas'] = results['gt_areas'][gt_valid]
                # results['gt_labels'] = results['gt_labels'][gt_valid]
                keypoints[invalid_idx > 0, :] = 0
                # keypoints = keypoints[gt_valid]
                if len(keypoints) == 0:
                    return None
                keypoints = keypoints.reshape(keypoints.shape[0], -1)
                if self.kpt_clip_border:
                    keypoints[:, 0::3] = np.clip(keypoints[:, 0::3], 0,
                                                 img_shape[1])
                    keypoints[:, 1::3] = np.clip(keypoints[:, 1::3], 0,
                                                 img_shape[0])
                results[key] = keypoints
        # crop face bboxes
        if 'gt_face_bboxes' in results:
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            face_bboxes = results['gt_face_bboxes'] - bbox_offset
            if self.bbox_clip_border:
                # bbox_area_clip_before = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                face_bboxes[:, 0::2] = np.clip(face_bboxes[:, 0::2], 0, img_shape[1])
                face_bboxes[:, 1::2] = np.clip(face_bboxes[:, 1::2], 0, img_shape[0])
                # bbox_area_clip_after = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                # if key == 'gt_bboxes' and 'gt_areas' in results:
                #     clip_ratio = bbox_area_clip_after / bbox_area_clip_before
                #     results['gt_areas'] = results['gt_areas'] * clip_ratio
                #     results['gt_areas'] = np.clip(results['gt_areas'], a_min=1.0, a_max=1e8)
            valid_inds = (face_bboxes[:, 2] > face_bboxes[:, 0]) & (
                    face_bboxes[:, 3] > face_bboxes[:, 1])
            invalid_inds = [not _ for _ in valid_inds]
            # in this version, valid gt equals 0
            results['gt_face_labels'][invalid_inds] = 1
            results['gt_face_bboxes'] = face_bboxes
        if "gt_dense_pose" in results:
            num_instance = len(results['gt_dense_pose'])
            for i in range(num_instance):
                new_cur_dp = []
                num_dp = len(results['gt_dense_pose'][i])
                for j in range(num_dp):
                    if results['gt_dense_pose'][i][j]['x'] > crop_x1 and results['gt_dense_pose'][i][j]['x'] < crop_x2 \
                            and results['gt_dense_pose'][i][j]['y'] > crop_y1 and results['gt_dense_pose'][i][j][
                        'y'] < crop_y2:
                        results['gt_dense_pose'][i][j]['x'] -= crop_x1
                        results['gt_dense_pose'][i][j]['y'] -= crop_y1
                        new_cur_dp.append(results['gt_dense_pose'][i][j])
                results['gt_dense_pose'][i] = new_cur_dp

        for key in self.bbox2others['gt_bboxes']:
            assert len(results[key]) == len(results['gt_bboxes'])
        return results

    def __repr__(self):
        repr_str = super(UniRandomCrop, self).__repr__()[:-1] + ', '
        repr_str += f'kpt_clip_border={self.kpt_clip_border})'
        return repr_str


@PIPELINES.register_module()
class KPResize(Resize):
    """Resize images & bbox & mask & keypoint & mask area.

    Args:
        keypoint_clip_border (bool, optional): Whether to clip the objects
            outside the border of the image. Defaults to True.
    """

    def __init__(self,
                 *args,
                 keypoint_clip_border=True,
                 **kwargs):
        super(KPResize, self).__init__(*args, **kwargs)
        self.keypoint_clip_border = keypoint_clip_border

    def _resize_keypoints(self, results):
        """Resize keypoints with ``results['scale_factor']``."""
        for key in results.get('keypoint_fields', []):
            keypoints = results[key].copy()
            keypoints[:,
            0::3] = keypoints[:, 0::3] * results['scale_factor'][0]
            keypoints[:,
            1::3] = keypoints[:, 1::3] * results['scale_factor'][1]
            if self.keypoint_clip_border:
                img_shape = results['img_shape']
                keypoints[:, 0::3] = np.clip(keypoints[:, 0::3], 0,
                                             img_shape[1])
                keypoints[:, 1::3] = np.clip(keypoints[:, 1::3], 0,
                                             img_shape[0])
            results[key] = keypoints

    def _resize_areas(self, results):
        """Resize mask areas with ``results['scale_factor']``."""
        for key in results.get('area_fields', []):
            areas = results[key].copy()
            areas = areas * results['scale_factor'][0] * results[
                'scale_factor'][1]
            results[key] = areas

    def _resize_dp(self, results):
        if "gt_dense_pose" in results:
            num_instance = len(results['gt_dense_pose'])
            for i in range(num_instance):
                num_dp = len(results['gt_dense_pose'][i])
                for j in range(num_dp):
                    results['gt_dense_pose'][i][j]['x'] *= results['scale_factor'][0]
                    results['gt_dense_pose'][i][j]['y'] *= results['scale_factor'][1]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map, keypoints, mask areas.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        results = super(KPResize, self).__call__(results)
        self._resize_keypoints(results)
        self._resize_areas(results)
        self._resize_dp(results)
        return results

    def __repr__(self):
        repr_str = super(KPResize, self).__repr__()[:-1] + ', '
        repr_str += f'keypoint_clip_border={self.keypoint_clip_border})'
        return repr_str


@PIPELINES.register_module()
class KPRandomFlip(RandomFlip):
    """Flip the image & bbox & mask & keypoint.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    """

    def keypoint_flip(self, keypoints, img_shape, direction, flip_pairs):
        """Flip keypoints horizontally.

        Args:
            keypoints (numpy.ndarray): person's keypoints, shape (..., K*3).
            img_shape (tuple[int]): Image shape (height, width).
            direction (str): Flip direction. Only 'horizontal' is supported.
            flip_pairs (list): Flip pair indices.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        assert keypoints.shape[-1] % 3 == 0
        flipped = keypoints.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped = flipped.reshape(flipped.shape[0], flipped.shape[1] // 3,
                                      3)
            valid_idx = flipped[..., -1] > 0
            flipped[valid_idx, 0] = w - flipped[valid_idx, 0]
            for pair in flip_pairs:
                flipped[:, pair, :] = flipped[:, pair[::-1], :]
            flipped[..., 0] = np.clip(flipped[..., 0], 0, w)
            flipped = flipped.reshape(flipped.shape[0], keypoints.shape[1])
        elif direction == 'vertical':
            raise NotImplementedError
        elif direction == 'diagonal':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def keypoint_3d_flip(self, keypoints, img_shape, direction, flip_pairs):
        """Flip keypoints horizontally.

        Args:
            keypoints (numpy.ndarray): person's keypoints, shape (..., K*3).
            img_shape (tuple[int]): Image shape (height, width).
            direction (str): Flip direction. Only 'horizontal' is supported.
            flip_pairs (list): Flip pair indices.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """
        # print("call keypoint_3d_flip!!!")
        assert keypoints.shape[-1] % 3 == 0
        flipped = keypoints.copy()
        # print("before_flip:",flipped)
        if direction == 'horizontal':
            w = img_shape[1]
            flipped = flipped.reshape(flipped.shape[0], flipped.shape[1] // 3,
                                      3)
            # valid_idx = flipped[..., -1] > 0
            valid_idx = flipped[..., -1] != -2
            # flipped[valid_idx, 0] = w - flipped[valid_idx, 0]
            flipped[valid_idx, 0] = - flipped[valid_idx, 0]
            for pair in flip_pairs:
                flipped[:, pair, :] = flipped[:, pair[::-1], :]
            # flipped[..., 0] = np.clip(flipped[..., 0], 0, w)
            flipped = flipped.reshape(flipped.shape[0], keypoints.shape[1])
            # print("after_flip:", flipped)
        elif direction == 'vertical':
            raise NotImplementedError
        elif direction == 'diagonal':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def dense_pose_flip(self, dense_pose, img_shape, direction):
        # flipped = dense_pose.copy()
        num_instance = len(dense_pose)
        if direction == 'horizontal':
            w = img_shape[1]
            for i in range(num_instance):
                num_dp = len(dense_pose[i])
                for j in range(num_dp):
                    dense_pose[i][j]['x'] = min(max(w - dense_pose[i][j]['x'], 0), w)
        elif direction == 'vertical':
            raise NotImplementedError
        elif direction == 'diagonal':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return dense_pose

    def _flip_smpl_pose(self, pose_):
        """Flip SMPL pose parameters horizontally.

        Args:
            pose (np.ndarray([72])): SMPL pose parameters
        Returns:
            pose_flipped
        """
        # print("call flip")
        flippedParts = [
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
            20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
            38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
            59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
        ]
        # print(type(pose_))
        # print("pose:",pose_.shape)
        for i in range(len(pose_)):
            pose = pose_[i]
            # print(pose_[i])
            pose_flipped = pose[flippedParts]
            # Negate the second and the third dimension of the axis-angle
            pose_flipped[1::3] = -pose_flipped[1::3]
            pose_flipped[2::3] = -pose_flipped[2::3]
            pose_[i] = pose_flipped
            # print(pose_[i])
        return pose_

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps, keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        results = super(KPRandomFlip, self).__call__(results)
        # print("call flip!!!",results['flip'])
        if results['flip']:
            # flip keypoints
            for key in results.get('keypoint_fields', []):
                results[key] = self.keypoint_flip(
                    results[key], results['img_shape'],
                    results['flip_direction'],
                    results['ann_info']['flip_pairs'])
            if "gt_dense_pose" in results:
                results['gt_dense_pose'] = self.dense_pose_flip(
                    results['gt_dense_pose'], results['img_shape'],
                    results['flip_direction'],
                )
            if "gt_smpl_pose" in results:
                results['gt_smpl_pose'] = self._flip_smpl_pose(results['gt_smpl_pose'])
            if "gt_keypoints3d" in results:
                # print("call gt_keypoints3d!")
                results['gt_keypoints3d'] = self.keypoint_3d_flip(results['gt_keypoints3d'], results['img_shape'],
                                                                  results['flip_direction'],
                                                                  results['ann_info']['flip_pairs'])
        return results


@PIPELINES.register_module()
class MultiTaskFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255),
                 keys=['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_attributes', 'gt_keypoints']):
        self.img_to_float = img_to_float
        self.pad_val = pad_val
        self.keys = keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in self.keys:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_dense_pose' in results:
            results['gt_dense_pose'] = DC(
                results['gt_dense_pose'],
                # padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(img_to_float={self.img_to_float})'
