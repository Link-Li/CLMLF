"""
Name: label_smooth
Date: 2020/10/19 下午10:21
Version: 1.0
"""

import torch.nn.modules as nn
import torch

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    来自英伟达，这个计算的时候，smooth值包含了正确的类
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingPure(nn.Module):
    """
    来自https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631，
    这个计算的时候，smooth值没有包含正确的类
    """
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingPure, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))