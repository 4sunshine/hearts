from torch.utils.data import Dataset
from data.statistics import personalize_data, load_data
import numpy as np
import torch


class BaseDataset(Dataset):
    def __init__(self, is_train, transform, cfg, file='data/train.csv', split_file='data/indices.pth'):
        self.is_train = is_train
        self.cfg = cfg
        self.data = self.load_data(file, split_file)
        self.anomalies_masks = torch.load('data/anomalies_masks.pth')
        self.transform = transform

    def load_data(self, file, split_file):
        all_data = load_data(file)
        personal_data = personalize_data(all_data)
        if split_file is not None:
            splits = torch.load(split_file)
            personal_data = np.array(personal_data)[splits['train']] if self.is_train \
                else np.array(personal_data)[splits['val']]
        else:
            indices = np.arange(len(personal_data), dtype=np.int32)
            np.random.shuffle(indices)
            split = int(self.cfg.TRAIN * len(personal_data))
            train_indices, val_indices = indices[:split].copy(), indices[split:].copy()
            if self.is_train:
                personal_data = list(personal_data[i] for i in train_indices)
            else:
                personal_data = list(personal_data[i] for i in val_indices)
        return personal_data

    def person_labels(self, item):
        person = self.data[item]
        person_id = person[0, 0]
        labels = person[:, -1].astype(np.float64)
        person = person[:, 1:-1]
        person = np.transpose(person).astype(np.float64)  # CHANNELS: [TIME, RR], TICKS
        return person, labels, person_id

    def get_training_data(self, item):
        person, labels, person_id = self.person_labels(item)
        anomaly_starts = self.anomalies_masks['anomaly_start_indices'][item]
        anomaly_ends = self.anomalies_masks['anomaly_end_indices'][item]
        sample = {
            'person': person,
            'labels': labels,
            # 'anomalies_starts': anomaly_starts,
            # 'anomalies_ends': anomaly_ends,
            'start_pos': 0,
            'end_pos': len(labels),
            'person_id': person_id
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_test_data(self, item):
        person, labels, person_id = self.person_labels(item)
        sample = {
            'person': person,
            'labels': labels,
            'start_pos': 0,
            'end_pos': len(labels),
            'person_id': person_id
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

