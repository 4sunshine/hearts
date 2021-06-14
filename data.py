import matplotlib.pyplot as plt
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
    personal_data = np.split(data, changes_ids)[:-1]  # LAST ELEMENT IS EMPTY
    # LIST OF NP ARRAYS
    return personal_data


def extract_observation_data(person):
    data = person[:, 1:]
    n_measures = len(data)
    times = data[:, 0]
    rr = data[:, 1]
    labels = data[:, -1]
    observation_time = times[-1]
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    anomal_rr = rr[labels.astype(np.bool)]
    mean_anomal = np.mean(anomal_rr)
    std_anomal = np.std(anomal_rr)

    diffs = np.diff(labels)

    starts_ids = np.argwhere(diffs > 0).squeeze(1) + 1

    if labels[0] > 0:
        starts_ids = np.concatenate([[0], starts_ids])

    ends_ids = np.argwhere(diffs < 0).squeeze(1) + 1
    if labels[-1] > 0:
        ends_ids = np.concatenate([ends_ids, [n_measures - 1]])

    assert len(starts_ids) == len(ends_ids), 'FAIL ANOMALY MASKING'

    anomaly_measures = ends_ids - starts_ids
    anomaly_durations = times[ends_ids] - times[starts_ids]
    intra_measures = starts_ids[1:] - ends_ids[:-1]
    intra_durations = times[starts_ids[1:]] - times[ends_ids[:-1]]
    return observation_time, n_measures, mean_rr, std_rr, mean_anomal, std_anomal, starts_ids, ends_ids,\
           anomaly_measures, anomaly_durations, intra_measures, intra_durations


def stats_from_list_of_numpy(data_list):
    n_elements = 0
    values = []
    for data in data_list:
        if len(data) > 0:
            values.append(np.mean(data))
            n_elements += 1
    values = np.array(values)
    return np.mean(values), np.std(values), np.min(values), np.max(values)


def measure_stats(personal_data):
    # SEQUENCES LENGTHS
    count_measures = []
    # OBS TIMES
    observation_times = []
    # M, STD_RR & FOR ANOMALY
    means_rr, stds_rr = [], []
    means_anomal, stds_anomal = [], []
    # MASKS FOR ANOMALY
    anomalies_starts, anomalies_ends = [], []
    # ANOMALIES DURATIONS
    anomalies_ticks, anomalies_durations, intras_ticks, intras_durations = [], [], [], []
    for p in personal_data:
        observation_time, n_measures, mean_rr, std_rr, mean_anomal, std_anomal, starts_ids, ends_ids, \
        anomaly_measures, anomaly_durations, intra_measures, intra_durations = extract_observation_data(p)

        count_measures.append(n_measures)
        observation_times.append(observation_time)
        means_rr.append(mean_rr)
        stds_rr.append(std_rr)
        means_anomal.append(mean_anomal)
        stds_anomal.append(std_anomal)
        anomalies_starts.append(starts_ids)
        anomalies_ends.append(ends_ids)
        anomalies_ticks.append(anomaly_measures)
        anomalies_durations.append(anomaly_durations)
        intras_ticks.append(intra_measures)
        intras_durations.append(intra_durations)

    count_measures = np.array(count_measures)
    observation_times = np.array(observation_times)
    means_rr = np.array(means_rr)
    stds_rr = np.array(stds_rr)
    means_anomal = np.array(means_anomal)
    stds_anomal = np.array(stds_anomal)

    mean_rr = np.mean(means_rr)  ## TBD
    std_rr = np.mean(stds_rr)  # TBD
    mean_anomal_rr = np.mean(means_anomal)
    std_anomal_rr = np.mean(stds_anomal)
    mean_obs_time = np.mean(observation_times)
    mean_ticks = np.mean(count_measures)

    mean_an_ticks, std_an_ticks, min_an_ticks, max_an_ticks = stats_from_list_of_numpy(anomalies_ticks)
    mean_an_dur, std_an_dur, min_an_dur, max_an_dur = stats_from_list_of_numpy(anomalies_durations)
    mean_in_ticks, std_in_ticks, min_in_ticks, max_in_ticks = stats_from_list_of_numpy(intras_ticks)
    mean_in_dur, std_in_dur, min_in_dur, max_in_dur = stats_from_list_of_numpy(intras_durations)


def measure_len_analys(personal_data):
    count_measures = [len(p) for p in personal_data]
    plt.hist(count_measures, bins=50, density=True)
    median = np.median(count_measures)
    plt.ylabel('Вероятность')
    plt.xlabel('Число измерений на человека')
    plt.show()


if __name__ == '__main__':
    all_data = load_data('data\\train.csv')
    # LIST OF NP ARRAYS
    personal_data = personalize_data(all_data)
    measure_stats(personal_data)
    measure_len_analys(personal_data)
