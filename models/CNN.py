from torch import nn


class CNN(nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv1d(1, 128, 5, 1, padding=2)
        self.BN1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.Conv2 = nn.Conv1d(128, 256, 3, 1, padding=1)
        self.BN2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(2, 2)

        self.Conv3 = nn.Conv1d(80, 512, 3, 1, padding=1)
        self.BN3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(2, 2)

        self.lin1 = nn.Linear(512, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 1)

        self.up = nn.ConvTranspose1d(1, 1, 4, 4)

    def forward(self, x):
        x = x[:, 1:, :]
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.act1(x)

        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.act2(x)
        x = self.mp1(x)

        x = self.Conv(x)
        x = self.BN3(x)
        x = self.act3(x)
        x = self.mp2(x)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        x = self.up(x)
        return x