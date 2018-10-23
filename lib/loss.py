import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from lib.lovasz import lovasz_hinge


class BCELoss(_Loss):
    def __init__(self, per_image=True, from_logits=True):
        super().__init__()
        self.per_image = per_image
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_pred.size(0)
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        if self.per_image:
            return loss.view(batch_size, -1).mean(dim=1)
        return loss.mean()


class FocalLoss(_Loss):
    def __init__(self, gamma, per_image=True, from_logits=True):
        super().__init__()
        self.gamma = float(gamma)
        self.per_image = per_image
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

            # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
            invprobs = F.logsigmoid(-y_pred * (y_true * 2 - 1))
            loss = (invprobs * self.gamma).exp() * loss
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

            # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
            invprobs = -y_pred * (y_true * 2 - 1)
            loss = (invprobs * self.gamma).exp() * loss

        if self.per_image:
            batch_size = y_pred.size(0)
            return loss.view(batch_size, -1).mean(dim=1)

        return loss.mean()


class JaccardLoss(_Loss):
    def __init__(self, per_image=True, from_logits=True, smooth=10):
        super().__init__()
        self.from_logits = from_logits
        self.per_image = per_image
        self.smooth = float(smooth)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_pred.size(0)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        if self.per_image:
            y_pred = y_pred.view(batch_size, -1)
            y_true = y_true.view(batch_size, -1)

            intersection = torch.sum(y_pred * y_true, dim=1)
            union = torch.sum(y_pred, dim=1) + torch.sum(y_true, dim=1) - intersection
        else:
            intersection = torch.sum(y_pred * y_true, dim=None)
            union = torch.sum(y_pred, dim=None) + torch.sum(y_true, dim=None) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class BCEAndJaccardLoss(_Loss):
    def __init__(self, bce_weight=1, jaccard_weight=1, per_image=True, from_logits=True):
        super().__init__()
        self.bce = BCELoss(per_image=per_image, from_logits=from_logits)
        self.bce_weight = float(bce_weight)
        self.jaccard = JaccardLoss(per_image=per_image, from_logits=from_logits)
        self.jaccard_weight = float(jaccard_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        bce_loss = self.bce(y_pred, y_true)
        iou_loss = self.jaccard(y_pred, y_true)
        return (bce_loss * self.bce_weight + iou_loss * self.jaccard_weight) / (self.bce_weight + self.jaccard_weight)


class FocalAndJaccardLoss(_Loss):
    def __init__(self, focal_weight=1, jaccard_weight=1, per_image=True, from_logits=True):
        super().__init__()
        self.focal = FocalLoss(per_image=per_image, from_logits=from_logits, gamma=2)
        self.focal_weight = float(focal_weight)
        self.jaccard = JaccardLoss(per_image=per_image, from_logits=from_logits)
        self.jaccard_weight = float(jaccard_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        foc_loss = self.focal(y_pred, y_true)
        iou_loss = self.jaccard(y_pred, y_true)
        return (foc_loss * self.focal_weight + iou_loss * self.jaccard_weight) / (self.focal_weight + self.jaccard_weight)


class LovaszHingeLoss(_Loss):
    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, output: Tensor, target: Tensor):
        return lovasz_hinge(output, target, self.per_image, self.ignore)


class BCEAndLovaszLoss(_Loss):
    def __init__(self, bce_weight=1, lovasz_weight=1, per_image=True, from_logits=True):
        super().__init__()
        if not from_logits:
            raise ValueError("This loss operates only on logits")

        self.bce = BCELoss(per_image=per_image, from_logits=from_logits)
        self.bce_weight = float(bce_weight)
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        bce_loss = self.bce(y_pred, y_true)
        lov_loss = self.lovasz(y_pred, y_true)
        return (bce_loss * self.bce_weight + lov_loss * self.lovasz_weight) / (self.bce_weight + self.lovasz_weight)
