import numpy as np
import torch
from einops.einops import rearrange
from torch.nn import functional as F


# def vertices_kp3d_projection(j3d_preds, cam_preds, joints_h36m17_preds=None, vertices=None, input2orgimg_offsets=None, presp=False):
#     # if presp:
#     #     pred_cam_t = denormalize_cam_params_to_trans(cam_preds, positive_constrain=False)
#     #     pj3d = perspective_projection(j3d_preds,translation=pred_cam_t,focal_length=args().focal_length, normalize=True)
#     #     projected_outputs = {'cam_trans':pred_cam_t, 'pj2d': pj3d[:,:,:2].float()}
#     #     if joints_h36m17_preds is not None:
#     #         pj3d_h36m17 = perspective_projection(joints_h36m17_preds,translation=pred_cam_t,focal_length=args().focal_length, normalize=True)
#     #         projected_outputs['pj2d_h36m17'] = pj3d_h36m17[:,:,:2].float()
#     #     if vertices is not None:
#     #         projected_outputs['verts_camed'] = perspective_projection(vertices.clone().detach(),translation=pred_cam_t,focal_length=args().focal_length, normalize=True, keep_dim=True)
#     #         projected_outputs['verts_camed'][:,:,2] = vertices[:,:,2]
#     # else:
#
#     pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
#     # print("j3d_preds:",j3d_preds)
#     # print("pj3d:",pj3d)
#     # print("cam_preds:",cam_preds)
#     pred_cam_t = convert_cam_to_3d_trans(cam_preds)
#     projected_outputs = {'pj2d': pj3d[:,:,:2], 'cam_trans':pred_cam_t}
#     if vertices is not None:
#         projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d',keep_dim=True)
#
#     if input2orgimg_offsets is not None:
#         projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], input2orgimg_offsets)
#         projected_outputs['verts_camed_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['verts_camed'], input2orgimg_offsets)
#         if 'pj2d_h36m17' in projected_outputs:
#             projected_outputs['pj2d_org_h36m17'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d_h36m17'], input2orgimg_offsets)
#     return projected_outputs

def vertices_kp3d_projection(j3d_preds, cam_preds, joints_h36m17_preds=None, vertices=None, input2orgimg_offsets=None,
                             presp=False):
    pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
    # print("j3d_preds:",j3d_preds)
    # print("pj3d:",pj3d)
    # print("cam_preds:",cam_preds)
    pred_cam_t = convert_cam_to_3d_trans(cam_preds)
    projected_outputs = {'pj2d': pj3d[:, :, :2], 'cam_trans': pred_cam_t}
    if vertices is not None:
        projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d', keep_dim=True)

    if input2orgimg_offsets is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'],
                                                                          input2orgimg_offsets)
        projected_outputs['verts_camed_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['verts_camed'],
                                                                                 input2orgimg_offsets)
        if 'pj2d_h36m17' in projected_outputs:
            projected_outputs['pj2d_org_h36m17'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d_h36m17'],
                                                                                     input2orgimg_offsets)
    return projected_outputs


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def convert_cam_to_3d_trans(cams, weight=2.):
    (s, tx, ty) = cams[:, 0], cams[:, 1], cams[:, 2]
    depth, dx, dy = 1. / s, tx / s, ty / s
    trans3d = torch.stack([dx, dy, depth], 1) * weight
    return trans3d


def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10]
    leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
    kp2ds_on_orgimg = (kp2ds[:, :, :2] + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    if kp2ds.shape[-1] == 3:
        kp2ds_on_orgimg = torch.cat(
            [kp2ds_on_orgimg, (kp2ds[:, :, [2]] + 1) * img_pad_size.unsqueeze(1)[:, :, [0]] / 2], -1)
    return kp2ds_on_orgimg


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    if x.shape[-1] == 6:
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            num = x.shape[1]
            x = rearrange(x, 'b n d -> (b n) d', d=6)
        else:
            num = 1
        x = rearrange(x, 'b (k l) -> b k l', k=3, l=2)
        # x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 -
                         torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)

        mat = torch.stack((b1, b2, b3), dim=-1)
        if num > 1:
            mat = rearrange(
                mat, '(b n) h w-> b n h w', b=batch_size, n=num, h=3, w=3)
    else:
        if isinstance(x, torch.Tensor):
            x = x.view(-1, 3, 2)
        elif isinstance(x, np.ndarray):
            x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 -
                         torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        mat = torch.stack((b1, b2, b3), dim=-1)
    return mat


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.

    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w = norm_quat[:, 0]
    x = norm_quat[:, 1]
    y = norm_quat[:, 2]
    z = norm_quat[:, 3]
    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(B, 3, 3)
    return rotMat
