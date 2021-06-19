import os
from pathlib import Path
from numpy import genfromtxt

import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':
    files = os.listdir('data\\new_data')
    id = 1000
    all_data = {'id': [], 'time': [], 'x': [], 'y': []}

    for f in files:
        with open(f'data\\new_data\\{f}', 'r') as f:
            data = f.read()

        data = data.split('\n')
        data = np.array([int(1000 * float(d.strip())) for d in data[:-1]])
        data = data / 1000 if data[0] > 1e3 else data
        RR = data.astype(int)
        time = [0, ] + list(np.cumsum(RR))
        for i, t, x in zip(np.full(len(RR), id), time, RR):
            all_data['id'] += [i]
            all_data['time'] += [t]
            all_data['x'] += [x]
            all_data['y'] += [0]
        id += 1

    all_data = pd.DataFrame(all_data)
    print(all_data.head())
    all_data.to_csv('train_healthy.csv')