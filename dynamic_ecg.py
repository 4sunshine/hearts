import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def plot_ecg(time, RR, target, prediction):
    RR = RR[time != 0]
    target = target[time != 0]
    prediction = prediction[time != 0]
    time = time[time != 0]

    plt.plot(time, RR)
    [plt.fill_between([time[i], ], 0, [RR[i], ], color='red', alpha=0.2) for i in range(len(time)) if target[i] == 1]
    [plt.fill_between([time[i], ], [RR[i], ], [max(RR), ], color='green', alpha=0.2) for i in range(len(time)) if prediction[i] == 1]
    plt.xlabel("Время, мин")
    plt.ylabel("R-R интервал")
    plt.title(f"Real-time ритмограмма пациента N{1}")
    plt.tight_layout()
    plt.savefig('image.png')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic ECG')
    parser.add_argument('-id', action='store', help='People ID', default=1)
    # # parser.add_argument('-speed', action='store', help='Speed', default=1)
    # parser.add_argument('-start', action='store', help='start', default=0)
    # parser.add_argument('-end', action='store', help='end', default=1e9)
    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    person = df.loc[df['id'] == args.id]
    person['time'] = person['time'] / 60000

    time, RR, label = person['time'].to_numpy(), person['x'].to_numpy(), person['y'].to_numpy()
    delta = np.asarray([time[i] - time[i-1] for i in range(1, len(time))])
    mean_delta, std_delta = delta.mean(), delta.std()

    plt.rcParams['animation.html'] = 'jshtml'
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    plt.xlabel("Время, мин")
    plt.ylabel("R-R интервал")
    plt.title(f"Real-time ритмограмма пациента N{1}")
    plt.tight_layout()
    fig.show()

    n_pionts_min = int(len(time) / max(time))
    n_min_show = 3
    n_points_show = n_min_show * n_pionts_min

    time = np.concatenate([-1 * n_min_show * np.arange(0, 1, 1 / n_points_show)[::-1], time])
    RR = np.concatenate([np.full(n_points_show, RR.mean()), RR])
    label = np.concatenate([np.zeros(n_points_show), label])
    plt.plot(time, RR)
    [plt.fill_between([time[i], ], 0, [RR[i], ], color='red', alpha=0.2) for i in range(len(time)) if label[i] == 1]

    for i in range(n_points_show, len(time)):
        plt.xlim(left=time[i - n_points_show], right=time[i])
        print()
        fig.canvas.draw()
        fig.canvas.flush_events()
