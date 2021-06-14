import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_class, map_to_seq_hidden=64, rnn_hidden=256):
        super(CRNN, self).__init__()
        self.Conv1 = nn.Conv1d(1, 60, 5, 1)
        self.act1 = nn.ReLU()
        self.Conv2 = nn.Conv1d(60, 80, 3, 1)
        self.act2 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(2, 2)
        self.do1 = nn.Dropout(0.05)
        self.Conv3 = nn.Conv1d(80, 128, 3, 1)
        self.act3 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(2, 2)
        self.do2 = nn.Dropout(0.15)
        #Masking / Batch Norm
        self.map_to_seq = nn.Linear(128 * 912, map_to_seq_hidden) # c * l
        self.lstm = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, x):
        x = x[:, 1:, :, 0]
        x = self.Conv1(x)
        x = self.act1(x)
        x = self.Conv2(x)
        x = self.act2(x)
        x = self.mp1(x)
        x = self.do1(x)
        x = self.Conv3(x)
        x = self.act3(x)
        x = self.mp2(x)
        x = self.do2(x)

        b, c, l = x.size()
        x = torch.reshape(x, (b, c*l))
        x = self.map_to_seq(x)
        x = torch.unsqueeze(x, 0)
        x, _ = self.lstm(x)
        x = self.dense(x)
        return x
