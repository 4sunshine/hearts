import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, target, mask):
        return self.criterion(output[mask], target[mask])
        # OUTPUT: BATCH x TIMES, TARGET: BATCH x TIMES


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target, mask):
        return self.criterion(output[mask].float(), target[mask].float())


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, output, target, mask):
        return self.criterion(output[mask], target[mask])


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1.e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        #self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, target, mask):
        output = output.sigmoid()
        output = output * mask
        intersection = torch.sum(output * target, -1)
        fps = torch.sum(output * (-target + 1.0), -1)
        fns = torch.sum((-output + 1.0) * target, -1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)

        return torch.mean(-tversky_loss + 1.0)
        #return self.criterion(output, target)
        # OUTPUT: BATCH x TIMES, TARGET: BATCH x TIMES


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets, mask):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs[mask], targets[mask], reduction='none')
        targets = targets[mask].type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets, mask):
        boundary_pred = torch.abs(torch.diff(inputs))
        boundary_gt = torch.abs(torch.diff(targets))
        return F.binary_cross_entropy_with_logits(boundary_pred[mask[:, :-1]], boundary_gt[mask[:, :-1]], reduction='none')


class Boundary_BCE(nn.Module):
    def __init__(self, alpha=1.):
        super(Boundary_BCE, self).__init__()
        self.bce = BCELoss()
        self.bound = BoundaryLoss()
        self.alpha = alpha

    def forward(self, inputs, targets, mask):
        l1 = self.bce(inputs, targets, mask)
        l2 = self.bound(inputs, targets, mask)
        return l1 + self.alpha * l2

        # targets = targets[mask].type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        # pt = torch.exp(-BCE_loss)
        # F_loss = at*(1-pt)**self.gamma * BCE_loss
        # return F_loss.mean()
