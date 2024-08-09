# Copyright (c) OpenMMLab. All rights reserved.

from .center_focal_loss import CenterFocalLoss
from .focal_loss import OnehotFocalLoss, OnehotSoftFocalLoss, NoActFocalLoss
from .meanvar_softmax_loss import MeanVarianceSoftmaxLoss
from .oks_loss import OKSLoss
from .pose_prior_loss import MaxMixturePrior

__all__ = [

    'OnehotFocalLoss', 'OnehotSoftFocalLoss', 'NoActFocalLoss',
    'MeanVarianceSoftmaxLoss', 'OKSLoss', 'CenterFocalLoss', 'MaxMixturePrior'
]
