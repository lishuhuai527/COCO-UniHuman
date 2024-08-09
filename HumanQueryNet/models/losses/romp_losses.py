import numpy as np
import torch

joint2D_tree = {
    'L_Shoulder': ['Jaw', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'], \
    'L_Ankle': ['L_BigToe', 'L_SmallToe', 'L_Heel', 'L_Toe_SMPL'], \
    'R_Ankle': ['R_BigToe', 'R_SmallToe', 'R_Heel', 'R_Toe_SMPL'], \
    }
SMPL_ALL_54 = {
    'Pelvis_SMPL': 0, 'L_Hip_SMPL': 1, 'R_Hip_SMPL': 2, 'Spine_SMPL': 3, 'L_Knee': 4, 'R_Knee': 5,
    'Thorax_SMPL': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Thorax_up_SMPL': 9, \
    'L_Toe_SMPL': 10, 'R_Toe_SMPL': 11, 'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'Jaw': 15, 'L_Shoulder': 16,
    'R_Shoulder': 17, \
    'L_Elbow': 18, 'R_Elbow': 19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23, 'Nose': 24,
    'R_Eye': 25, 'L_Eye': 26, 'R_Ear': 27, 'L_Ear': 28, \
    'L_BigToe': 29, 'L_SmallToe': 30, 'L_Heel': 31, 'R_BigToe': 32, 'R_SmallToe': 33, 'R_Heel': 34, \
    'L_Hand_thumb': 35, 'L_Hand_index': 36, 'L_Hand_middle': 37, 'L_Hand_ring': 38, 'L_Hand_pinky': 39, \
    'R_Hand_thumb': 40, 'R_Hand_index': 41, 'R_Hand_middle': 42, 'R_Hand_ring': 43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip': 46, 'Neck_LSP': 47, 'Head_top': 48, 'Pelvis': 49, 'Thorax_MPII': 50, \
    'Spine_H36M': 51, 'Jaw_H36M': 52, 'Head': 53
}


def batch_kp_2d_l2_loss(real, pred):
    """
    Directly supervise the 2D coordinates of global joints, like torso
    While supervise the relative 2D coordinates of part joints, like joints on face, feets
    """
    # invisible joints have been set to -2. in data pre-processing
    vis_mask = ((real > -1.99).sum(-1) == real.shape[-1]).float()

    for parent_joint, leaf_joints in joint2D_tree.items():
        parent_id = SMPL_ALL_54[parent_joint]
        leaf_ids = np.array([SMPL_ALL_54[leaf_joint] for leaf_joint in leaf_joints])
        vis_mask[:, leaf_ids] = vis_mask[:, [parent_id]] * vis_mask[:, leaf_ids]
        real[:, leaf_ids] -= real[:, [parent_id]]
        pred[:, leaf_ids] -= pred[:, [parent_id]]
    bv_mask = torch.logical_and(vis_mask.sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    vis_mask = vis_mask[bv_mask]
    loss = 0
    if vis_mask.sum() > 0:
        # diff = F.mse_loss(real[bv_mask], pred[bv_mask]).sum(-1)
        diff = torch.norm(real[bv_mask] - pred[bv_mask], p=2, dim=-1)
        loss = (diff * vis_mask).sum(-1) / (vis_mask.sum(-1) + 1e-4)
        # loss = (torch.norm(real[bv_mask]-pred[bv_mask],p=2,dim=-1) * vis_mask).sum(-1) / (vis_mask.sum(-1)+1e-4)

        if torch.isnan(loss).sum() > 0 or (loss > 1000).sum() > 0:
            return 0
            print('CAUTION: meet nan of pkp2d loss again!!!!')
            non_position = torch.isnan(loss)
            print('batch_kp_2d_l2_loss, non_position:', non_position, \
                  'diff results', diff, \
                  'real kp 2d vis', real[bv_mask][non_position][vis_mask[non_position].bool()], \
                  'pred kp 2d vis', pred[bv_mask][non_position][vis_mask[non_position].bool()])
            return 0
    return loss


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def compute_mpjpe(predicted, target, valid_mask=None, pck_joints=None, sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape, print(predicted.shape, target.shape)
    mpjpe = torch.norm(predicted - target, p=2, dim=-1)

    if pck_joints is None:
        if sample_wise:
            mpjpe_batch = (mpjpe * valid_mask.float()).sum(-1) / valid_mask.float().sum(
                -1) if valid_mask is not None else mpjpe.mean(-1)
        else:
            mpjpe_batch = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_batch
    else:
        mpjpe_pck_batch = mpjpe[:, pck_joints]
        return mpjpe_pck_batch


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:, :, 0] != -2.
    if align_inds is not None:
        pred_aligned = align_by_parts(pred, align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real, align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    if return_org:
        return mpjpe_each, (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each


def calc_pampjpe(real, pred, sample_wise=True, return_transform_mat=False):
    real, pred = real.float(), pred.float()
    # extracting the keypoints that all samples have the annotations
    vis_mask = (real[:, :, 0] != -2.).sum(0) == len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:, vis_mask], real[:, vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:, vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each
