from train import get_model
from data.transforms import get_test_sequence_transform
from data.dataset import TestDataset
from torch.utils.data import DataLoader
from options import get_config
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np


def get_test_padded_sample(sample):
    person = [s['person'] for s in sample]
    seq_lens = [s['end_pos'] for s in sample]
    time = [s['time'] for s in sample]
    person_id = sample[0]['person_id']
    max_seq_len = int(max(seq_lens))
    person = pad_sequence(person, batch_first=True)
    if max_seq_len % 4:
        person = F.pad(person, pad=(0, 0, 0, 4 - max_seq_len % 4), mode='constant', value=0)
    person = person.permute(0, 2, 1)
    return person, time[0].astype(np.int32), max_seq_len, person_id


class OutputWriter:
    def __init__(self, cfg, name='test_submission.csv'):
        self.all_data = {'id': [], 'time': [], 'y': []}
        os.makedirs(os.path.join('experiments', cfg.experiment_name), exist_ok=True)
        self.save_path = os.path.join('experiments', cfg.experiment_name, name)

    def append(self, output, time, person_id):
        assert len(output) == len(time)
        self.all_data['id'] += [person_id] * len(time)
        self.all_data['time'] += list(time)
        self.all_data['y'] += output.tolist()

    def write_output(self):
        all_data = pd.DataFrame(self.all_data)
        all_data.to_csv(self.save_path, index=False)
        self.all_data = {'id': [], 'time': [], 'y': []}


if __name__ == '__main__':
    cfg = get_config()
    cfg.device = torch.device('cuda' if torch.cuda.is_available() and cfg.cuda else 'cpu')

    with torch.no_grad():
        writer = OutputWriter(cfg)
        model = get_model(cfg)
        model.eval()
        val_transform = get_test_sequence_transform(cfg)
        test_set = TestDataset(transform=val_transform, cfg=cfg)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            # BATCH CROP LEN
            person, time, max_seq_len, person_id = get_test_padded_sample(sample)
            # FORWARD
            output = model(person.float().to(cfg.device))[0, :max_seq_len]

            output = (output.sigmoid() > cfg.threshold).int().cpu().numpy()

            writer.append(output, time, person_id)

        writer.write_output()




