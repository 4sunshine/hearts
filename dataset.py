from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from dynamic_ecg import plot_ecg
from models.CRNN import CRNN

RR_MEAN = 641.282
RR_STD = 121.321
TIME_MEAN = 162468.
TIME_STD = 235341.
MAX_LEN = 3661         # max n_measures
TRAIN = 0.7
VAL = 0.3


class BaseDataset(Dataset):
    def __init__(self, file, transform):
        self.data = self.__load_data__(file)
        self.transform = torch.jit.script(transform)

    @staticmethod
    def __load_data__(file):
        data = pd.read_csv(file)
        data = np.array(data.values).astype(np.int32)
        ids = data[:, 0]
        diffs = np.diff(ids)
        changes_ids = np.argwhere(diffs != 0).squeeze(1) + 1
        personal_data = np.split(data, changes_ids)
        return personal_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        person = self.data[item]
        person = np.pad(person, ((0, MAX_LEN - len(person)), (0, 0)))
        labels = person[:, -1]
        person = person[:, 1:-1]

        # C x H x W
        person = np.transpose(person)
        person = np.expand_dims(person, 2)

        person = torch.Tensor(person)
        labels = torch.Tensor(labels)

        person = person.to(torch.float32)
        labels = labels.to(torch.float32)

        # person = transform(person)
        return person, labels


transform = torch.nn.Sequential(
    transforms.Normalize([RR_MEAN, TIME_MEAN], [RR_STD, TIME_STD]),)
dataset = BaseDataset('data\\train.csv', transform)

indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = {
    'train': indices[:int(0.7 * len(dataset))],
    'val': indices[int(0.7 * len(dataset)): int(0.9 * len(dataset))],
    'test': indices[int(0.9 * len(dataset)):]
}
torch.save(split, 'data\\indices.pth')


train_sampler = SubsetRandomSampler(split['train'])
val_sampler = SubsetRandomSampler(split['val'])

train_loader = DataLoader(dataset, batch_size=30, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=30, sampler=val_sampler)

model = CRNN(MAX_LEN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.BCEWithLogitsLoss()
all_accuracy = []
n_epoch = 250
for epoch in range(n_epoch):
    accuracy = []
    for p, l in train_loader:
        optimizer.zero_grad()

        pred = model(p)[0]
        loss_val = loss(pred, l)
        loss_val.backward()
        optimizer.step()

        a = pred.sigmoid().detach().numpy().astype(float).flatten()
        b = l.int().detach().numpy().flatten()
        a[a >= 0.5] = 1
        a[a < 0.5] = 0
        # a = int(a > 0.5)
        # a = torch.nn.Sigmoid(a)
        # a[a < 0.5] = 0
        # a[a > 0.5] = 1
        # print(len(a[a > 0.5]))
        # print(a, b)
        time = p[0, 0,:,:]
        RR = p[0, 1,:,:]
        target = l[0]
        predi = pred.sigmoid()[0]
        predi[predi <= 0.75] = 0
        predi[predi > 0.75] = 1
        plot(time, RR, target, predi)
        accuracy += [f1_score(a, b, average='micro')]
    all_accuracy += [np.mean(accuracy)]
    print(epoch, np.mean(accuracy))

plt.plot(all_accuracy)