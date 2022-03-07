import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):

    def __init__(self, classes, gamma=2.0, alpha=0.75, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classes = classes
        self.alpha = alpha

    def forward(self, inputs, targets):
        # y = Variable(to_one_hot(target_.data, self.classes))
        # targets_ = label_to_one_hot_label(targets, num_classes=2)
        targets = F.one_hot(targets, num_classes=2)
        targets = targets.to(dtype=torch.float16)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        loss = torch.mean(loss)

        return loss