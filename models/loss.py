import torch
import torch.nn as nn


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