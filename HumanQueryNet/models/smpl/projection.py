import cv2
import numpy as np
import torch


# from .utils import rot6D_to_angular, batch_orth_proj, estimate_translation

def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512., 512.]), proj_mat=None,
                             cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0, 0], camK[1, 1] = focal_length, focal_length
        camK[:2, 2] = img_size // 2
    else:
        camK = proj_mat
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist, \
                                                  flags=cv2.SOLVEPNP_EPNP, reprojectionError=20, iterationsCount=100)

    if inliers is None:
        return None
    else:
        tra_pred = tvec[:, 0]
        return tra_pred


def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512., 512.]),
                            proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        # focal length
        f = np.array([focal_length, focal_length])
        # optical center
        center = img_size / 2.
    else:
        f = np.array([proj_mat[0, 0], proj_mat[1, 1]])
        center = proj_mat[:2, 2]

    # transformations
    Z = np.reshape(np.tile(joints_3d[:, 2], (2, 1)).T, -1)
    XY = np.reshape(joints_3d[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
                  O - np.reshape(joints_2d, -1)]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(joints_3d, joints_2d, pts_mnum=4, focal_length=600, proj_mats=None, cam_dists=None,
                         img_size=np.array([512., 512.])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()

    if joints_2d.shape[-1] == 2:
        joints_conf = joints_2d[:, :, -1] > -2.
    elif joints_2d.shape[-1] == 3:
        joints_conf = joints_2d[:, :, -1] > 0
    joints3d_conf = joints_3d[:, :, -1] != -2.

    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float32)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i, :, :2]
        valid_mask = joints_conf[i] * joints3d_conf[i]
        if valid_mask.sum() < pts_mnum:
            trans[i] = None
            continue
        if len(img_size.shape) == 1:
            imgsize = img_size
        elif len(img_size.shape) == 2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask],
                                                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i],
                                                cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask],
                                               valid_mask[valid_mask].astype(np.float32),
                                               focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])

    return torch.from_numpy(trans).float()


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    print("camera:", camera.shape)
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def convert_cam_to_3d_trans2(j3ds, pj3d):
    predicts_j3ds = j3ds[:, :24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:, :, :2][:, :24].detach().cpu().numpy() + 1) * 256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                     focal_length=443.4, img_size=np.array([512, 512])).to(j3ds.device)
    return cam_trans


def convert_proejection_from_input_to_orgimg(kps, offsets):
    top, bottom, left, right, h, w = offsets
    img_pad_size = max(h, w)
    kps[:, :, 0] = (kps[:, :, 0] + 1) * img_pad_size / 2 - left
    kps[:, :, 1] = (kps[:, :, 1] + 1) * img_pad_size / 2 - top
    if kps.shape[-1] == 3:
        kps[:, :, 2] = (kps[:, :, 2] + 1) * img_pad_size / 2
    return kps


def body_mesh_projection2image(j3d_preds, cam_preds, vertices=None, input2org_offsets=None):
    pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
    pred_cam_t = convert_cam_to_3d_trans2(j3d_preds, pj3d)
    projected_outputs = {'pj2d': pj3d[:, :, :2], 'cam_trans': pred_cam_t}
    if vertices is not None:
        projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d', keep_dim=True)

    if input2org_offsets is not None:
        projected_outputs['pj2d_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['pj2d'],
                                                                                 input2org_offsets)
        projected_outputs['verts_camed_org'] = convert_proejection_from_input_to_orgimg(
            projected_outputs['verts_camed'], input2org_offsets)
    return projected_outputs
