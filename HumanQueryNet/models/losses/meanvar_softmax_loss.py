import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class MeanVarianceSoftmaxLoss(nn.Module):
    def __init__(self, weight_mean, weight_var, reduction='mean', loss_weight=1.0):
        super(MeanVarianceSoftmaxLoss, self).__init__()

        self.weight_mean = weight_mean
        self.weight_var = weight_var
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        batch_size = pred.shape[0]
        class_num = pred.shape[-1]
        age = [i for i in range(class_num)]
        age_all = []
        for j in range(batch_size):
            age_all.append(age)
        age_numpy = np.array(age)
        age_all_numpy = np.array(age_all)
        age_tensor = torch.from_numpy(age_numpy).float().cuda()
        age_all_tensor = torch.from_numpy(age_all_numpy).float().cuda()
        out_softmax = F.softmax(pred, dim=1)
        mean = torch.mul(out_softmax, age_tensor).sum(dim=1)
        criterion_mean = nn.MSELoss(reduction='none')
        mean_loss = criterion_mean(mean, target.float())

        mean_all = mean.reshape([batch_size, 1]).expand(batch_size, class_num)
        cha = age_all_tensor - mean_all
        variance_loss = torch.mul(out_softmax, torch.mul(cha, cha)).sum(dim=1)  # .mean()

        criterion_softmax = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        softmax_loss = criterion_softmax(pred, target.long())

        loss = self.weight_mean * mean_loss + self.weight_var * variance_loss + softmax_loss
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight
        return loss
