from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
from data.statistics import personalize_data, load_data
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import json

from models.CRNN import CRNN


class BaseDataset(Dataset):
    def __init__(self, is_train, transform, cfg, file='data/train.csv', split_file=None):
        self.is_train = is_train
        self.cfg = cfg
        self.data = self.load_data(file, split_file)
        self.anomalies_masks = torch.load('data/anomalies_masks.pth')
        self.transform = transform

    def load_data(self, file, split_file):
        all_data = load_data(file)
        personal_data = personalize_data(all_data)
        if split_file is not None:
            with open(split_file, 'r') as f:
                splits = json.load(f)
            personal_data = personal_data[splits['train']] if self.is_train else personal_data[splits['val']]
        else:
            indices = np.arange(len(personal_data), dtype=np.int32)
            np.random.shuffle(indices)
            split = int(self.cfg.TRAIN * len(personal_data))
            train_indices, val_indices = indices[:split].copy(), indices[split:].copy()
            #print(train_indices)
            if self.is_train:
                personal_data = list(personal_data[i] for i in train_indices)
            else:
                personal_data = list(personal_data[i] for i in val_indices)
        return personal_data

    def person_labels(self, item):
        person = self.data[item]
        labels = person[:, -1].astype(np.float64)
        person = person[:, 1:-1]
        person = np.transpose(person).astype(np.float64)  # CHANNELS: [TIME, RR], TICKS
        return person, labels

    def get_training_data(self, item):
        person, labels = self.person_labels(item)
        anomaly_starts = self.anomalies_masks['anomaly_start_indices'][item]
        anomaly_ends = self.anomalies_masks['anomaly_end_indices'][item]
        sample = {
            'person': person,
            'labels': labels,
            # 'anomalies_starts': anomaly_starts,
            # 'anomalies_ends': anomaly_ends,
            'start_pos': 0,
            'end_pos': len(labels)
        }
        if self.transform:
            sample = self.transform(sample)

        print(sample['person'].shape)
        print(sample['labels'].shape)
        return sample

    def get_test_data(self, item):
        person, labels = self.person_labels(item)
        sample = {
            'person': person,
            'labels': labels,
            'start_pos': 0,
            'end_pos': len(labels)
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, item):
        if self.is_train:
            return self.get_training_data(item)
        else:
            return self.get_test_data(item)

    def __len__(self):
        return len(self.data)


# transform = torch.nn.Sequential(
#     transforms.Normalize([RR_MEAN, TIME_MEAN], [RR_STD, TIME_STD]),)
# dataset = BaseDataset('train.csv', transform)
#
# indices = np.arange(len(dataset))
# np.random.shuffle(indices)
# split = int(TRAIN * len(dataset))
# train_indices, val_indices = indices[:split], indices[split:]
#
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
#
# train_loader = DataLoader(dataset, batch_size=30, sampler=train_sampler)
# val_loader = DataLoader(dataset, batch_size=30, sampler=val_sampler)
#
# model = CRNN(MAX_LEN)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss = torch.nn.BCEWithLogitsLoss()
# all_accuracy = []
# n_epoch = 250
# for epoch in range(n_epoch):
#     accuracy = []
#     for p, l in train_loader:
#         optimizer.zero_grad()
#
#         pred = model(p)[0]
#         loss_val = loss(pred, l)
#         loss_val.backward()
#         optimizer.step()
#
#         a = pred.detach().numpy().astype(float).flatten()
#         b = l.detach().numpy().astype(float).flatten()
#
#         # a[a < 0.5] = 0
#         # a[a > 0.5] = 1
#         # print(len(a[a > 0.5]))
#         print(a, b)
#         accuracy += [f1_score(a, b)]
#     all_accuracy += [np.mean(accuracy)]
#     print(epoch, np.mean(accuracy))
#
# plt.plot(all_accuracy)
