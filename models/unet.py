import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = torch.nn.functional.pad(x1, [diff // 2, diff - diff // 2, ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = x[:, 1: , :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x[:, 0, :]
        return x

#
# import random
# from tqdm import tqdm
# import numpy as np
# import torch
#
#
# if __name__ == '__main__':
#     np.random.seed(42)
#     random.seed(1001)
#     torch.manual_seed(1002)
#
#     net = UNet(n_channels=1, n_classes=5, bilinear=True)
#     optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     dataset = WheatDS('dataset.json')
#     train_size = 0.6
#     batch_size = 1
#     train_loader, val_data, test_data = splitted_loaders(dataset, batch_size=1,
#                                                              train_size=train_size, val_size=0.2)
#
#     n_train = int(len(dataset) * 0.6) / batch_size
#
#     epochs = 40
#     test_accuracy_history = []
#     test_loss_history = []
#
#     for epoch in range(epochs):
#         net.train()
#         epoch_loss = 0
#         for batch in tqdm(train_loader):
#             imgs = batch[0]
#             masks = batch[1].type(torch.long)
#
#             pred = net.forward(imgs)
#             loss = criterion(pred, masks)
#             epoch_loss += loss.item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         for batch in val_data:
#             imgs = batch[0]
#             masks = torch.tensor(batch[1], dtype=torch.long)
#
#             pred = net.forward(imgs)
#             loss = criterion(pred, masks)
#             test_loss_history.append(loss.item())
#
#             accuracy = (pred.argmax(dim=1) == masks).float().mean()
#             test_accuracy_history.append(accuracy)
#
#     plotfig(test_loss_history, 'val_loss.png')
#     plotfig(test_accuracy_history, 'val_acc.png')
#
#     torch.save(net.state_dict(), 'unet.pth')
#     print('COMPLETE')