from typing import Literal

import torch.nn as nn
from mmengine import MODELS

from .depth_completion_loss import DepthCompletionLoss
from .video_depth_loss import VideoDepthLoss


@MODELS.register_module()
class loss_fn(nn.Module):
    def __init__(
        self,
        mode: Literal['video_depth', 'depth_completion'] = 'video_depth',
        losses: dict | None = None,
    ):
        super().__init__()

        if mode == 'video_depth':
            self.loss = VideoDepthLoss(losses=losses)
        elif mode == 'depth_completion':
            self.loss = DepthCompletionLoss(losses=losses)
        else:
            raise KeyError(f'Unsupported loss mode: {mode}')

    @property
    def loss_keys(self):
        return self.loss.loss_keys

    def forward(self, sample_pred, sample_target):
        return self.loss(sample_pred, sample_target)
