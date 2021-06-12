import numpy as np
import pandas as pd


def load_data(filepath):
    # 0 - ID, 1 - CUMULATIVE TIME, 2 - LOCAL INTERVAL, 3 - LABEL
    # SORTING CHECK
    df = pd.read_csv(filepath, sep=',', header=0)
    df_copy = df.copy()
    df_sorted = df_copy.sort_values(['id', 'time'], ascending=(True, True))
    sorted_data = np.array(df_sorted.values).astype(np.int32)
    all_data = np.array(df.values).astype(np.int32)
    max_diff = np.max(all_data - sorted_data)
    min_diff = np.min(all_data - sorted_data)
    assert max_diff == min_diff == 0, 'UNSORTED DATA'
    return all_data


def personalize_data(data):
    # DATA SHOULD BE TIME SORTED
    ids = np.squeeze(data[:, :1], axis=-1)
    diffs = np.diff(ids)
    # changes = np.where(diffs != 0)
    # NEXT ELEMENT IS STARTING POSITION
    changes_ids = np.argwhere(diffs != 0).squeeze(1) + 1
    # DON'T FORGET LAST ELEMENT IN DATA
    changes_ids = np.concatenate([changes_ids, [len(data)]])
    personalized = np.split(data, changes_ids)
    print(personalized)



if __name__ == '__main__':
    all_data = load_data('train.csv')
    personalize_data(all_data)