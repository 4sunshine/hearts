import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_class=1, # len features
                 rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()
        self.Conv1 = nn.Conv1d(1, 60, 5, 1, padding=2)
        self.act1 = nn.ReLU()
        self.Conv2 = nn.Conv1d(60, 80, 3, 1, padding=1)
        self.act2 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(2, 2)
        #self.do1 = nn.Dropout(0.05)
        self.Conv3 = nn.Conv1d(80, 128, 3, 1, padding=1)
        self.act3 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(2, 2)
        #self.do2 = nn.Dropout(0.15)
        #Masking / Batch Norm
        #self.map_to_seq = nn.Linear(128 * 912, map_to_seq_hidden) # c * l
        self.bi_lstm_1 = nn.LSTM(128, rnn_hidden, bidirectional=True)
        self.bi_lstm_2 = nn.LSTM(2 * rnn_hidden, 128, bidirectional=True)
        self.dense = nn.Linear(2 * 128, num_class)
        self.up = nn.ConvTranspose1d(1, 1, 4, 4)

    def forward(self, x):
        x = x[:, 1:, :]
        x = self.Conv1(x)
        x = self.act1(x)
        x = self.Conv2(x)
        x = self.act2(x)
        x = self.mp1(x)
        #x = self.do1(x)
        x = self.Conv3(x)
        x = self.act3(x)
        x = self.mp2(x)
        #x = self.do2(x)

        x = x.permute(2, 0, 1)
        # print(x.shape)
        # b, c, l = x.size()
        # x = torch.reshape(x, (b, c*l))
        # print(x.shape)
        #x = self.map_to_seq(x)
        #x = torch.unsqueeze(x, 0)
        x, _ = self.bi_lstm_1(x)
        x, _ = self.bi_lstm_2(x)
        # x: SEQ_LEN, BATCH, N_FEATURES -> BATCH, SEQ_LEN, N_CLASS = (1)
        x = x.permute(1, 0, 2)
        x = self.dense(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1)
        x = self.up(x)
        x.squeeze_(1)
        return x


if __name__ == '__main__':
    model = CRNN(num_class=1)
    dummy_input = torch.rand((4, 2, 1024))
    output = model(dummy_input)
