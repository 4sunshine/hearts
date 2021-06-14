import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.criterion(output, target)
        # OUTPUT: BATCH x TIMES, TARGET: BATCH x TIMES
