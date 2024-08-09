import torch
import torch.nn as nn
# from utils.projection import vertices_kp3d_projection
# from utils.rot_6D import rot6D_to_angular
from torch.nn import functional as F

# import config
# from config import args
# import constants
from .smpl import SMPL
from .utils import vertices_kp3d_projection


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 3)
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose


class SMPLWrapper(nn.Module):
    def __init__(self,
                 smpl_model_path,
                 cam_dim=3,
                 rot_dim=3,
                 smpl_joint_num=22,
                 smpl_mesh_root_align=True,
                 rot_type='3D',
                 perspective_proj=False
                 ):
        super(SMPLWrapper, self).__init__()
        self.smpl_model_path = smpl_model_path
        self.cam_dim = cam_dim
        self.rot_dim = rot_dim
        self.smpl_joint_num = smpl_joint_num
        self.smpl_mesh_root_align = smpl_mesh_root_align
        self.rot_type = rot_type
        self.perspective_proj = perspective_proj
        # self.smpl_model = smpl_model.create(self.smpl_model_path, J_reg_extra9_path=self.smpl_J_reg_extra_path, J_reg_h36m17_path=self.smpl_J_reg_h37m_path, \
        #    batch_size=self.batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False).cuda()
        self.smpl_model = SMPL(self.smpl_model_path, model_type='smpl').cuda()
        # self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        # self.part_idx = [self.cam_dim, self.rot_dim,  (self.smpl_joint_num-1)*self.rot_dim,       10]

        # self.unused_part_name = ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
        # self.unused_part_idx = [        15,                  15,           3,          3,            3,          10]

        # self.kps_num = 25 # + 21*2
        # self.params_num = np.array(self.part_idx).sum()
        # global_orient_nocam = np.array([0, 0, np.pi])
        # self.global_orient_nocam = torch.from_numpy(global_orient_nocam).unsqueeze(0)
        # self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()
        # self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def forward(self, poses, betas, cams, rot_wrapper="6D"):
        # print("call smpl forward!!!!!!!!!!!!!!!!!!")
        # idx_list, params_dict = [0], {}
        # for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
        #     idx_list.append(idx_list[i] + idx)
        #     # if name=="betas":
        #     #     print(idx_list[i],idx_list[i+1])
        #     params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i+1]].contiguous()

        if self.rot_type == '6D' and rot_wrapper == '6D':
            # params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            # params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
            poses = rot6D_to_angular(poses)
        # N = params_dict['body_pose'].shape[0]
        # print("poses",poses.shape)
        # N = poses.shape[0]
        # params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)
        # params_dict['poses'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)
        # print("params_dict['betas']:",params_dict['betas'])
        vertices, joints54_17 = self.smpl_model(betas=betas, poses=poses, root_align=self.smpl_mesh_root_align)

        outputs = {}
        outputs.update({'verts': vertices, 'j3d': joints54_17[:, :54], 'joints_h36m17': joints54_17[:, 54:]})

        # outputs.update(vertices_kp3d_projection(outputs['j3d'], cams, joints_h36m17_preds=outputs['joints_h36m17'], \
        #             vertices=outputs['verts'], input2orgimg_offsets=meta_data['offsets'], presp=self.perspective_proj))
        # for k,v in outputs.items():
        #     print("before:",k)
        if cams != None:
            outputs.update(vertices_kp3d_projection(outputs['j3d'], cams, joints_h36m17_preds=outputs['joints_h36m17'], \
                                                    vertices=outputs['verts'], presp=self.perspective_proj))
        # for k,v in outputs.items():
        #     print("after:",k)
        return outputs
