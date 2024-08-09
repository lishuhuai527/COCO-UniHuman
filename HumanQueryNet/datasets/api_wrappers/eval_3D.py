import numpy as np
import torch

from models.smpl import SMPLWrapper


def overlaps(target, sets):
    dets_np = np.array(sets)
    x1 = dets_np[:, 0]
    y1 = dets_np[:, 1]
    x2 = dets_np[:, 2]
    y2 = dets_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    area0 = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)

    xx1 = np.maximum(target[0], x1)
    yy1 = np.maximum(target[1], y1)
    xx2 = np.minimum(target[2], x2)
    yy2 = np.minimum(target[3], y2)  # 计算相交的面积,不重叠时面积为0
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h  # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
    ovr = inter / (area0 + areas - inter)
    return ovr


def pck(gt, preds):
    preds = np.array(preds)
    preds = preds[:, -54 * 3:]
    preds = preds.reshape((-1, 54, 3))[:, :, :2]  # .reshape((-1,54*2))

    gt = np.array(gt)
    gt = gt.reshape((-1, 54, 3))
    mask = gt[:, :, -1] > 0
    # print("gt:",gt)
    # print("mask:",mask)
    gt = gt[:, :, :2]  # .reshape((-1,54*2))
    # print(preds.shape)
    dists = np.sum(abs(preds - gt), axis=-1) * mask
    dists = np.sum(dists, axis=1)
    return dists


def kp2ds_to_bbox(kp2ds, mask):
    # print("mask:",mask.shape)
    # print("kp2ds:", kp2ds.shape)
    # mask=np.repeat(mask,(100),0)
    # print("mask:",mask.shape)
    index = np.where(mask > 0)[1]
    # print("index:",index)
    # print("kp2ds[:, :, 0]",kp2ds[:, :, 0].shape)
    kp2ds_xs = kp2ds[:, index, 0]  # [index]
    # print("kp2ds_xs:",kp2ds_xs.shape)
    # kp2ds_xs = kp2ds[:, :, 0][mask]
    kp2ds_ys = kp2ds[:, index, 1]  # [index]
    # print("kp2ds_xs:",kp2ds_xs.shape)
    min_x = np.min(kp2ds_xs, axis=1).reshape(-1, 1)
    max_x = np.max(kp2ds_xs, axis=1).reshape(-1, 1)
    min_y = np.min(kp2ds_ys, axis=1).reshape(-1, 1)
    max_y = np.max(kp2ds_ys, axis=1).reshape(-1, 1)
    # print("min_x:", min_x.shape)
    bbox = np.concatenate((min_x, min_y, max_x, max_y), axis=-1)
    # print("bbox:",bbox.shape)
    return bbox


def kp2bbox_ious(gt, preds):
    preds = np.array(preds)
    preds = preds[:, -54 * 3:]
    preds = preds.reshape((-1, 54, 3))[:, :, :2]  # .reshape((-1,54*2))
    gt = np.array(gt)
    gt = gt.reshape((-1, 54, 3))
    mask = gt[:, :, -1] > 0
    # preds = preds[mask]
    # print("gt:",gt)
    # print("mask:",mask)
    gt = gt[:, :, :2]  # .reshape((-1,54*2))
    # gt=gt[mask]
    preds_bbox = kp2ds_to_bbox(preds, mask)
    gt_bbox = kp2ds_to_bbox(gt, mask)[0]
    overlaps(gt_bbox, preds_bbox)
    return overlaps


def single_3D(pred3ds, gt3ds, mask, kp_mode="smpl54"):
    pred3ds = np.array(pred3ds).squeeze()
    if kp_mode == "smpl54":
        SMPL54toLSP = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]
        SMPL54toLSP = np.array(SMPL54toLSP)
        pred3ds = pred3ds[:, SMPL54toLSP, :]  # *1000
    pred_pelvis = (pred3ds[:, 2] + pred3ds[:, 3]) / 2
    pred3ds = (pred3ds - pred_pelvis[:, None, :]) * 1000
    mpjpe = keypoint_mpjpe(gt3ds, pred3ds, mask=mask)
    pa_mpjpe = keypoint_mpjpe(gt3ds, pred3ds, mask=mask, alignment="procrustes")
    return mpjpe, pa_mpjpe


def py_3d_eval(results, coco_gt, smpl_model_path="", metric=""):
    img_ids = coco_gt.get_img_ids()
    smpl_model_n = SMPLWrapper("models/smpl/models/SMPL_NEUTRAL.pth", rot_type='6D')
    smpl_model_m = SMPLWrapper("models/smpl/models/SMPL_MALE.pth", rot_type='6D')
    smpl_model_f = SMPLWrapper("models/smpl/models/SMPL_FEMALE.pth", rot_type='6D')
    idx = -1
    gt3ds_smpl54tolsp = []
    gt3ds_h36m = []
    pred3ds_smpl54tolsp_gender = []
    pred3ds_smpl54tolsp_neutral = []
    pred3ds_h36m17tolsp_gender = []
    pred3ds_h36m17tolsp_neutral = []
    SMPL54toLSP = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]
    SMPL54toLSP = np.array(SMPL54toLSP)
    # [ 8  5 45 46  4  7 21 19 17 16 18 20 47 48]

    for i in img_ids:
        ann_ids = coco_gt.get_ann_ids(img_ids=[i])
        ann_info = coco_gt.load_anns(ann_ids)
        gts = []
        for _, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if ann['iscrowd']:
                continue
            bbox = ann['bbox']
            tmp = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[3] + bbox[1]]
            # coco-unihuman 0:male 1:female
            # 3dpw 0:female 1:male -> coco format
            gender = -1
            if 'gender' in ann:
                if ann['gender'] == 1:
                    gender = 0
                else:
                    gender = 1
            elif 'gender_manual' in ann:
                gender = ann['gender_manual']
            cur_gt = {'bbox': tmp, 'gender': gender}
            if 'keypoints_3d' in ann:
                gt_keypoints_3d = ann['keypoints_3d']
                cur_gt['3d'] = gt_keypoints_3d
            elif 'smpl' in ann:
                smpl_pose = ann['smpl']['pose']
                smpl_betas = ann['smpl']['betas']
                smpl_pose = torch.Tensor(np.array(smpl_pose)).unsqueeze(0).cuda()
                smpl_betas = torch.Tensor(np.array(smpl_betas)).unsqueeze(0).cuda()
                smpl_outs_neutral = smpl_model_n(poses=smpl_pose, betas=smpl_betas, cams=None, rot_wrapper="3D")
                gt_keypoints_3d_h36m = smpl_outs_neutral['joints_h36m17'].cpu().numpy()
                gt_keypoints_3d = smpl_outs_neutral['j3d'].cpu().numpy()
                cur_gt['3d'] = gt_keypoints_3d
                cur_gt['3d_h36m'] = gt_keypoints_3d_h36m
            # else:
            #     continue
            gts.append(cur_gt)
        idx += 1
        result = results[idx]  # ['bbox_results']
        qnum = result['det'][0][0].shape[0]
        preds = []
        smpls = []
        for q_idx in range(qnum):
            re = result['det'][0][0][q_idx]
            pred = []
            # print("re:",re.shape)
            for item in re:
                pred.append(item.item())
            preds.append(pred)
            # smpl = re[7:7+144+10+3]
            # smpls.append(smpl)
        for gt in gts:
            if not '3d' in gt:
                continue
            gt_bbox = gt['bbox']
            gt_keypoints_3d = gt['3d']
            gt3ds_smpl54tolsp.append(np.array(gt_keypoints_3d))
            if '3d_h36m' in gt:
                gt_keypoints_3d_h36m = gt['3d_h36m']
                gt3ds_h36m.append(np.array(gt_keypoints_3d_h36m))
            # gt_keypoints_2d = gt['2d']
            gender = gt['gender']
            overlaps_ = overlaps(gt_bbox, preds)
            argmax = np.argmax(overlaps_, 0)

            # pcks = pck(gt_keypoints_2d, preds)
            # argmin = np.argmin(pcks, 0)

            # kp_overlaps = kp2bbox_ious(gt_keypoints_2d, preds)
            # argmax_kp = np.argmax(kp_overlaps, 0)

            # match_kp = preds[argmin]
            match_det = preds[argmax]
            pred_gender = (match_det[5] > 0.5)
            # match_kp_box = preds[argmax_kp]
            # iou_ = overlaps_[argmax]
            # if iou_<0.5:
            #    continue
            # TODO use pred gender to eval
            match_pose = torch.Tensor(np.array(match_det[7:7 + 144])).unsqueeze(0).cuda()
            match_beta = torch.Tensor(np.array(match_det[7 + 144:7 + 144 + 10])).unsqueeze(0).cuda()
            match_cam = torch.Tensor(np.array(match_det[7 + 144 + 10:7 + 144 + 10 + 3])).unsqueeze(0).cuda()
            # coco-unihuman 0:male 1:female
            # 3dpw 0:female 1:male
            # if gender==0:
            #     smpl_model = smpl_model_m
            # elif gender==1:
            #     smpl_model = smpl_model_f
            # else:
            #     smpl_model = smpl_model_n
            if pred_gender:
                smpl_model = smpl_model_f
            else:
                smpl_model = smpl_model_m
            smpl_outs_with_gender = smpl_model(poses=match_pose, betas=match_beta, cams=match_cam)
            pred3ds_smpl54tolsp_gender.append(smpl_outs_with_gender['j3d'].cpu().numpy())
            pred3ds_h36m17tolsp_gender.append(smpl_outs_with_gender['joints_h36m17'][:, :14].cpu().numpy())

            smpl_outs_neutral = smpl_model_n(poses=match_pose, betas=match_beta, cams=match_cam)
            pred3ds_smpl54tolsp_neutral.append(smpl_outs_neutral['j3d'].cpu().numpy())
            pred3ds_h36m17tolsp_neutral.append(smpl_outs_neutral['joints_h36m17'][:, :14].cpu().numpy())

    gt3ds_smpl54tolsp = np.array(gt3ds_smpl54tolsp)
    gt3ds_smpl54tolsp = gt3ds_smpl54tolsp.reshape((-1, 54, 3))
    gt3ds_smpl54tolsp = gt3ds_smpl54tolsp[:, SMPL54toLSP, :]  # *1000
    gt_pelvis_smpl54 = (gt3ds_smpl54tolsp[:, 2] + gt3ds_smpl54tolsp[:, 3]) / 2
    gt3ds_smpl54tolsp = (gt3ds_smpl54tolsp - gt_pelvis_smpl54[:, None, :]) * 1000
    mask = np.ones(gt3ds_smpl54tolsp.shape).astype(bool)
    mask[gt3ds_smpl54tolsp == -2] = 0
    mask = mask[:, :, 0]
    mpjpe_gt_smpl54_pred_smpl54_with_gender, pa_mpjpe_gt_smpl54_pred_smpl54_with_gender = single_3D(
        pred3ds_smpl54tolsp_gender, gt3ds_smpl54tolsp, mask, kp_mode="smpl54")
    mpjpe_gt_smpl54_pred_h36m_with_gender, pa_mpjpe_gt_smpl54_pred_h36m_with_gender = single_3D(
        pred3ds_h36m17tolsp_gender, gt3ds_smpl54tolsp, mask, kp_mode="h36m")
    mpjpe_gt_smpl54_pred_smpl54_neutral, pa_mpjpe_gt_smpl54_pred_smpl54_neutral = single_3D(pred3ds_smpl54tolsp_neutral,
                                                                                            gt3ds_smpl54tolsp, mask,
                                                                                            kp_mode="smpl54")
    mpjpe_gt_smpl54_pred_h36m_neutral, pa_mpjpe_gt_smpl54_pred_h36m_neutral = single_3D(pred3ds_h36m17tolsp_neutral,
                                                                                        gt3ds_smpl54tolsp, mask,
                                                                                        kp_mode="h36m")

    result_dict = {'mpjpe_gt_smpl54_pred_smpl54_with_gender': mpjpe_gt_smpl54_pred_smpl54_with_gender,
                   'pa_mpjpe_gt_smpl54_pred_smpl54_with_gender': pa_mpjpe_gt_smpl54_pred_smpl54_with_gender,
                   'mpjpe_gt_smpl54_pred_h36m_with_gender': mpjpe_gt_smpl54_pred_h36m_with_gender,
                   'pa_mpjpe_gt_smpl54_pred_h36m_with_gender': pa_mpjpe_gt_smpl54_pred_h36m_with_gender,
                   'mpjpe_gt_smpl54_pred_smpl54_neutral': mpjpe_gt_smpl54_pred_smpl54_neutral,
                   'pa_mpjpe_gt_smpl54_pred_smpl54_neutral': pa_mpjpe_gt_smpl54_pred_smpl54_neutral,
                   'mpjpe_gt_smpl54_pred_h36m_neutral': mpjpe_gt_smpl54_pred_h36m_neutral,
                   'pa_mpjpe_gt_smpl54_pred_h36m_neutral': pa_mpjpe_gt_smpl54_pred_h36m_neutral}
    if len(gt3ds_h36m) > 0:
        gt3ds_h36m = np.array(gt3ds_h36m)
        gt3ds_h36m = gt3ds_h36m.reshape((-1, 17, 3))[:, :14, :]
        gt_pelvis_h36m = (gt3ds_h36m[:, 2] + gt3ds_h36m[:, 3]) / 2
        gt3ds_h36m = (gt3ds_h36m - gt_pelvis_h36m[:, None, :]) * 1000
        mpjpe_gt_h36m_pred_smpl54_with_gender, pa_mpjpe_gt_h36m_pred_smpl54_with_gender = single_3D(
            pred3ds_smpl54tolsp_gender, gt3ds_h36m, mask, kp_mode="smpl54")
        mpjpe_gt_h36m_pred_h36m_with_gender, pa_mpjpe_gt_h36m_pred_h36m_with_gender = single_3D(
            pred3ds_h36m17tolsp_gender, gt3ds_h36m, mask, kp_mode="h36m")
        mpjpe_gt_h36m_pred_smpl54_neutral, pa_mpjpe_gt_h36m_pred_smpl54_neutral = single_3D(pred3ds_smpl54tolsp_neutral,
                                                                                            gt3ds_h36m, mask,
                                                                                            kp_mode="smpl54")
        mpjpe_gt_h36m_pred_h36m_neutral, pa_mpjpe_gt_h36m_pred_h36m_neutral = single_3D(pred3ds_h36m17tolsp_neutral,
                                                                                        gt3ds_h36m, mask,
                                                                                        kp_mode="h36m")

        result_dict['mpjpe_gt_h36m_pred_smpl54_with_gender'] = mpjpe_gt_h36m_pred_smpl54_with_gender
        result_dict['pa_mpjpe_gt_h36m_pred_smpl54_with_gender'] = pa_mpjpe_gt_h36m_pred_smpl54_with_gender
        result_dict['mpjpe_gt_h36m_pred_h36m_with_gender'] = mpjpe_gt_h36m_pred_h36m_with_gender
        result_dict['pa_mpjpe_gt_h36m_pred_h36m_with_gender'] = pa_mpjpe_gt_h36m_pred_h36m_with_gender
        result_dict['mpjpe_gt_h36m_pred_smpl54_neutral'] = mpjpe_gt_h36m_pred_smpl54_neutral
        result_dict['pa_mpjpe_gt_h36m_pred_smpl54_neutral'] = pa_mpjpe_gt_h36m_pred_smpl54_neutral
        result_dict['mpjpe_gt_h36m_pred_h36m_neutral'] = mpjpe_gt_h36m_pred_h36m_neutral
        result_dict['pa_mpjpe_gt_h36m_pred_h36m_neutral'] = pa_mpjpe_gt_h36m_pred_h36m_neutral
    print("")
    for k, v in result_dict.items():
        if k[0] == "m":
            print(k.ljust(40, " "), ":{:.2f}  {:.2f}".format(v, result_dict['pa_' + k]))
    return result_dict


def py_3d_eval_old(results, coco_gt, smpl_model_path="", metric=""):
    img_ids = coco_gt.get_img_ids()
    smpl_model_m = SMPLWrapper("models/smpl/models/SMPL_MALE.pth", rot_type='6D')
    smpl_model_f = SMPLWrapper("models/smpl/models/SMPL_FEMALE.pth", rot_type='6D')
    idx = -1
    gt3ds = []
    pred3ds = []
    pred3ds_kp_match = []
    pred3ds_kp_box_match = []
    SMPL54toLSP = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]
    for i in img_ids:
        ann_ids = coco_gt.get_ann_ids(img_ids=[i])
        ann_info = coco_gt.load_anns(ann_ids)
        gts = []
        for _, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if ann['iscrowd']:
                continue
            bbox = ann['bbox']
            gt_keypoints_3d = ann['keypoints_3d']
            gt_keypoints_2d = ann['keypoints_2d']
            # for k,v in ann.items():
            #     print(k,v)
            # quit()
            tmp = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[3] + bbox[1]]
            # gt_attrs = [-1 for _ in range(num_attributes)]
            # if 'attributes' in ann:
            #     gt_attrs=ann['attributes']
            # for item in gt_attrs:
            #     tmp.append(item)
            gender = ann['gender']
            gts.append({'bbox': tmp, '3d': gt_keypoints_3d, '2d': gt_keypoints_2d, 'gender': gender})
        idx += 1
        result = results[idx]  # ['bbox_results']
        qnum = result['det'][0][0].shape[0]
        preds = []
        smpls = []
        for q_idx in range(qnum):
            re = result['det'][0][0][q_idx]
            pred = []
            # print("re:",re.shape)
            for item in re:
                pred.append(item.item())
            preds.append(pred)
            # smpl = re[7:7+144+10+3]
            # smpls.append(smpl)
        for gt in gts:
            gt_bbox = gt['bbox']
            gt_keypoints_3d = gt['3d']
            gt_keypoints_2d = gt['2d']
            gender = gt['gender']
            overlaps_ = overlaps(gt_bbox, preds)
            argmax = np.argmax(overlaps_, 0)

            pcks = pck(gt_keypoints_2d, preds)
            argmin = np.argmin(pcks, 0)

            kp_overlaps = kp2bbox_ious(gt_keypoints_2d, preds)
            argmax_kp = np.argmax(kp_overlaps, 0)

            match_kp = preds[argmin]
            match_det = preds[argmax]
            match_kp_box = preds[argmax_kp]
            # iou_ = overlaps_[argmax]
            # if iou_<0.5:
            #    continue

            match_pose = torch.Tensor(np.array(match_det[7:7 + 144])).unsqueeze(0)
            match_beta = torch.Tensor(np.array(match_det[7 + 144:7 + 144 + 10])).unsqueeze(0)
            match_cam = torch.Tensor(np.array(match_det[7 + 144 + 10:7 + 144 + 10 + 3])).unsqueeze(0)
            if gender == 1:
                smpl_model = smpl_model_m
            else:
                smpl_model = smpl_model_f
            smpl_outs = smpl_model(poses=match_pose, betas=match_beta, cams=match_cam)
            pred3ds.append(smpl_outs['j3d'].numpy())

            match_pose_kp = torch.Tensor(np.array(match_kp[7:7 + 144])).unsqueeze(0)
            match_beta_kp = torch.Tensor(np.array(match_kp[7 + 144:7 + 144 + 10])).unsqueeze(0)
            match_cam_kp = torch.Tensor(np.array(match_kp[7 + 144 + 10:7 + 144 + 10 + 3])).unsqueeze(0)

            smpl_outs_kp = smpl_model(poses=match_pose_kp, betas=match_beta_kp, cams=match_cam_kp)
            pred3ds_kp_match.append(smpl_outs_kp['j3d'].numpy())

            match_pose_kp_box = torch.Tensor(np.array(match_kp_box[7:7 + 144])).unsqueeze(0)
            match_beta_kp_box = torch.Tensor(np.array(match_kp_box[7 + 144:7 + 144 + 10])).unsqueeze(0)
            match_cam_kp_box = torch.Tensor(np.array(match_kp_box[7 + 144 + 10:7 + 144 + 10 + 3])).unsqueeze(0)

            smpl_outs_kp_box = smpl_model(poses=match_pose_kp_box, betas=match_beta_kp_box, cams=match_cam_kp_box)
            pred3ds_kp_box_match.append(smpl_outs_kp_box['j3d'].numpy())

            gt3ds.append(np.array(gt_keypoints_3d))

    SMPL54toLSP = np.array(SMPL54toLSP)
    gt3ds = np.array(gt3ds)
    gt3ds = gt3ds.reshape((-1, 54, 3))
    pred3ds = np.array(pred3ds).squeeze()

    gt3ds = gt3ds[:, SMPL54toLSP, :]  # *1000
    # print(gt3ds)
    mask = np.ones(gt3ds.shape).astype(bool)
    mask[gt3ds == -2] = 0
    pred3ds = pred3ds[:, SMPL54toLSP, :]  # *1000
    pred_pelvis = (pred3ds[:, 2] + pred3ds[:, 3]) / 2
    gt_pelvis = (gt3ds[:, 2] + gt3ds[:, 3]) / 2
    pred3ds = (pred3ds - pred_pelvis[:, None, :]) * 1000
    gt3ds = (gt3ds - gt_pelvis[:, None, :]) * 1000

    pred3ds_kp_match = np.array(pred3ds_kp_match).squeeze()
    pred3ds_kp_match = pred3ds_kp_match[:, SMPL54toLSP, :]  # *1000
    pred_pelvis = (pred3ds_kp_match[:, 2] + pred3ds_kp_match[:, 3]) / 2
    pred3ds_kp_match = (pred3ds_kp_match - pred_pelvis[:, None, :]) * 1000

    pred3ds_kp_box_match = np.array(pred3ds_kp_box_match).squeeze()
    pred3ds_kp_box_match = pred3ds_kp_box_match[:, SMPL54toLSP, :]  # *1000
    pred_pelvis = (pred3ds_kp_box_match[:, 2] + pred3ds_kp_box_match[:, 3]) / 2
    pred3ds_kp_box_match = (pred3ds_kp_box_match - pred_pelvis[:, None, :]) * 1000

    mask = mask[:, :, 0]
    mpjpe = keypoint_mpjpe(gt3ds, pred3ds, mask=mask)
    pa_mpjpe = keypoint_mpjpe(gt3ds, pred3ds, mask=mask, alignment="procrustes")

    mpjpe_kp = keypoint_mpjpe(gt3ds, pred3ds_kp_match, mask=mask)
    pa_mpjpe_kp = keypoint_mpjpe(gt3ds, pred3ds_kp_match, mask=mask, alignment="procrustes")

    mpjpe_kp_box = keypoint_mpjpe(gt3ds, pred3ds_kp_box_match, mask=mask)
    pa_mpjpe_kp_box = keypoint_mpjpe(gt3ds, pred3ds_kp_box_match, mask=mask, alignment="procrustes")

    return {'mpjpe': mpjpe, 'pa_mpjpe': pa_mpjpe, 'mpjpe_kp': mpjpe_kp, 'pa_mpjpe_kp': pa_mpjpe_kp,
            'mpjpe_kp_box': mpjpe_kp_box, 'pa_mpjpe_kp_box': pa_mpjpe_kp_box}


# def py_3d_eval(results, coco_gt, smpl_model_path="", metric=""):
#     img_ids = coco_gt.get_img_ids()
#     smpl_model = SMPLWrapper(smpl_model_path, rot_type='6D')
#     idx=-1
#     gt3ds=[]
#     pred3ds=[]
#     pred3ds_kp_match=[]
#     pred3ds_kp_box_match=[]
#     SMPL54toLSP=[8,5,45,46,4,7,21,19,17,16,18,20,47,48]
#     for i in img_ids:
#         ann_ids = coco_gt.get_ann_ids(img_ids=[i])
#         ann_info = coco_gt.load_anns(ann_ids)
#         gts=[]
#         for _, ann in enumerate(ann_info):
#             if ann.get('ignore', False):
#                 continue
#             if ann['iscrowd']:
#                 continue
#             bbox=ann['bbox']
#             gt_keypoints_3d=ann['keypoints_3d']
#             gt_keypoints_2d=ann['keypoints_2d']
#             # for k,v in ann.items():
#             #     print(k,v)
#             # quit()
#             tmp = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[3]+bbox[1]]
#             # gt_attrs = [-1 for _ in range(num_attributes)]
#             # if 'attributes' in ann:
#             #     gt_attrs=ann['attributes']
#             # for item in gt_attrs:
#             #     tmp.append(item)
#             gts.append({'bbox':tmp, '3d':gt_keypoints_3d, '2d':gt_keypoints_2d})
#         idx += 1
#         result = results[idx] #['bbox_results']
#         qnum = result['det'][0][0].shape[0]
#         preds = []
#         smpls = []
#         for q_idx in range(qnum):
#             re = result['det'][0][0][q_idx]
#             pred = []
#             # print("re:",re.shape)
#             for item in re:
#                 pred.append(item.item())
#             preds.append(pred)
#             # smpl = re[7:7+144+10+3]
#             # smpls.append(smpl)
#         for gt in gts:
#             gt_bbox=gt['bbox']
#             gt_keypoints_3d=gt['3d']
#             gt_keypoints_2d = gt['2d']
#             overlaps_ = overlaps(gt_bbox, preds)
#             argmax = np.argmax(overlaps_, 0)
#
#             pcks=pck(gt_keypoints_2d, preds)
#             argmin = np.argmin(pcks, 0)
#
#             kp_overlaps = kp2bbox_ious(gt_keypoints_2d,preds)
#             argmax_kp = np.argmax(kp_overlaps, 0)
#
#             match_kp = preds[argmin]
#             match_det = preds[argmax]
#             match_kp_box = preds[argmax_kp]
#             # iou_ = overlaps_[argmax]
#             # if iou_<0.5:
#             #    continue
#
#             match_pose = torch.Tensor(np.array(match_det[7:7+144])).unsqueeze(0)
#             match_beta = torch.Tensor(np.array(match_det[7 + 144:7+144+10])).unsqueeze(0)
#             match_cam = torch.Tensor(np.array(match_det[7 + 144+10:7+144+10+3])).unsqueeze(0)
#
#             smpl_outs = smpl_model(poses=match_pose, betas=match_beta, cams=match_cam)
#             pred3ds.append(smpl_outs['j3d'].numpy())
#
#             match_pose_kp = torch.Tensor(np.array(match_kp[7:7+144])).unsqueeze(0)
#             match_beta_kp = torch.Tensor(np.array(match_kp[7 + 144:7+144+10])).unsqueeze(0)
#             match_cam_kp = torch.Tensor(np.array(match_kp[7 + 144+10:7+144+10+3])).unsqueeze(0)
#
#             smpl_outs_kp = smpl_model(poses=match_pose_kp, betas=match_beta_kp, cams=match_cam_kp)
#             pred3ds_kp_match.append(smpl_outs_kp['j3d'].numpy())
#
#             match_pose_kp_box = torch.Tensor(np.array(match_kp_box[7:7+144])).unsqueeze(0)
#             match_beta_kp_box = torch.Tensor(np.array(match_kp_box[7 + 144:7+144+10])).unsqueeze(0)
#             match_cam_kp_box = torch.Tensor(np.array(match_kp_box[7 + 144+10:7+144+10+3])).unsqueeze(0)
#
#             smpl_outs_kp_box = smpl_model(poses=match_pose_kp_box, betas=match_beta_kp_box, cams=match_cam_kp_box)
#             pred3ds_kp_box_match.append(smpl_outs_kp_box['j3d'].numpy())
#
#             gt3ds.append(np.array(gt_keypoints_3d))
#
#     SMPL54toLSP=np.array(SMPL54toLSP)
#     gt3ds=np.array(gt3ds)
#     gt3ds=gt3ds.reshape((-1,54,3))
#     pred3ds=np.array(pred3ds).squeeze()
#
#     gt3ds = gt3ds[:, SMPL54toLSP, :] #*1000
#     # print(gt3ds)
#     mask=np.ones(gt3ds.shape).astype(np.bool)
#     mask[gt3ds==-2]=0
#     pred3ds = pred3ds[:,SMPL54toLSP,:] #*1000
#     pred_pelvis = (pred3ds[:, 2] + pred3ds[:, 3]) / 2
#     gt_pelvis = (gt3ds[:, 2] + gt3ds[:, 3]) / 2
#     pred3ds = (pred3ds - pred_pelvis[:, None, :]) * 1000
#     gt3ds = (gt3ds - gt_pelvis[:, None, :]) * 1000
#
#     pred3ds_kp_match = np.array(pred3ds_kp_match).squeeze()
#     pred3ds_kp_match = pred3ds_kp_match[:, SMPL54toLSP, :]  # *1000
#     pred_pelvis = (pred3ds_kp_match[:, 2] + pred3ds_kp_match[:, 3]) / 2
#     pred3ds_kp_match = (pred3ds_kp_match - pred_pelvis[:, None, :]) * 1000
#
#     pred3ds_kp_box_match = np.array(pred3ds_kp_box_match).squeeze()
#     pred3ds_kp_box_match = pred3ds_kp_box_match[:, SMPL54toLSP, :]  # *1000
#     pred_pelvis = (pred3ds_kp_box_match[:, 2] + pred3ds_kp_box_match[:, 3]) / 2
#     pred3ds_kp_box_match = (pred3ds_kp_box_match - pred_pelvis[:, None, :]) * 1000
#
#     mask=mask[:,:,0]
#     mpjpe=keypoint_mpjpe(gt3ds, pred3ds, mask=mask)
#     pa_mpjpe=keypoint_mpjpe(gt3ds, pred3ds, mask=mask,alignment="procrustes")
#
#     mpjpe_kp=keypoint_mpjpe(gt3ds, pred3ds_kp_match, mask=mask)
#     pa_mpjpe_kp=keypoint_mpjpe(gt3ds, pred3ds_kp_match, mask=mask,alignment="procrustes")
#
#     mpjpe_kp_box=keypoint_mpjpe(gt3ds, pred3ds_kp_box_match, mask=mask)
#     pa_mpjpe_kp_box=keypoint_mpjpe(gt3ds, pred3ds_kp_box_match, mask=mask,alignment="procrustes")
#
#     return {'mpjpe':mpjpe,'pa_mpjpe':pa_mpjpe,'mpjpe_kp':mpjpe_kp, 'pa_mpjpe_kp':pa_mpjpe_kp,
#             'mpjpe_kp_box':mpjpe_kp_box, 'pa_mpjpe_kp_box':pa_mpjpe_kp_box}
#     # return result_dict

def compute_similarity_transform(source_points,
                                 target_points,
                                 return_tform=False):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale.

    And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.
    Notes:
        Points number: N
    Args:
        source_points (np.ndarray([N, 3])): Source point set.
        target_points (np.ndarray([N, 3])): Target point set.
        return_tform (bool) : Whether return transform
    Returns:
        source_points_hat (np.ndarray([N, 3])): Transformed source point set.
        transform (dict): Returns if return_tform is True.
            Returns rotation: r, 'scale': s, 'translation':t.
    """

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.dot(source_points) + t

    source_points_hat = source_points_hat.T

    if return_tform:
        return source_points_hat, {
            'rotation': R,
            'scale': scale,
            'translation': t
        }

    return source_points_hat


def keypoint_mpjpe(pred, gt, mask, alignment='none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (PA-MPJPE).
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
    Returns:
        tuple: A tuple containing joint position errors
        - mpjpe (float|np.ndarray[N]): mean per-joint position error.
        - pa-mpjpe (float|np.ndarray[N]): mpjpe after rigid alignment with the
            ground truth
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)[mask].mean()

    return error
