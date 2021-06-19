from models.CRNN import CRNN
from models.unet import UNet
import torch
from torch import nn


class EnsembleU(nn.Module):
    def __init__(self, u_path, cr_path, num_class=1, # len features
                 rnn_hidden=256, epoch=0):
        super(EnsembleU, self).__init__()
        self.unet = UNet(n_channels=1, n_classes=1)
        self.crnn = CRNN()
        self.crnn.load_state_dict(torch.load(cr_path))
        self.unet.load_state_dict(torch.load(u_path))
        self.act = nn.ELU()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        if epoch == 0:
            self.epoch = 0
            self.freeze_back()
        else:
            self.epoch = 0

    def freeze_back(self):
        for child in self.crnn.children():
            for param in child.parameters():
                param.requires_grad = False
        for child in self.unet.children():
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze_back(self):
        for child in self.crnn.children():
            for param in child.parameters():
                param.requires_grad = True
        for child in self.unet.children():
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        self.epoch += 1
        if self.epoch == 40:
            self.unfreeze_back()
        crnn_x = self.crnn(x).unsqueeze_(1)
        unet_x = self.unet(x).unsqueeze_(1)
        x = torch.cat([crnn_x, unet_x], dim=1)
        x = self.act(x)
        return self.conv(x).squeeze_(1)

