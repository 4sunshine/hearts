import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class FigPlotter:
    @staticmethod
    def plot_ecg(time, RR, target, prediction, person_id):
        plt.figure()
        RR = RR[RR != 0]
        target = target[RR != 0]
        prediction = prediction[RR != 0]
        time = time[RR != 0]

        # mean, std = RR[target == 1].mean(), RR[target == 1].std()
        # mean, std = RR.mean(), RR.std()
        # RR[RR > mean + std] = mean + std
        # RR[RR < mean - std] = mean - std

        # if len(person > 200): # mediana
        #     mean, std = person[1].mean(), person[1].std()
        #     person[1][person[1] > (mean + 4 * std)] = mean + 4 * std
        #     person[1][person[1] < (mean - 4 * std)] = mean - 4 * std

        plt.plot(time, RR)
        [plt.fill_between([time[i], ], [min(RR), ], [RR[i], ], color='red', alpha=0.2) for i in range(len(time)) if target[i] == 1]
        [plt.fill_between([time[i], ], [RR[i], ], [max(RR), ], color='green', alpha=0.2) for i in range(len(time)) if prediction[i] == 1]
        # [plt.fill_between([time[i], ], [min(RR), ], [max(RR), ], color='green', alpha=0.2) for i in range(len(time)) if RR[i] > (mean + 4*std)]
        # [plt.fill_between([time[i], ], [min(RR), ], [max(RR), ], color='green', alpha=0.2) for i in range(len(time)) if RR[i] < (mean - 4*std)]
        plt.xlabel("Время, мин")
        plt.ylabel("R-R интервал")
        plt.title(f"Ритмограмма пациента N{person_id}")
        plt.savefig(f'{person_id}')
        plt.tight_layout()

    @staticmethod
    def plot_scatter_ecg(RR, target, person_id):
        plt.figure()
        RR = RR[RR != 0]
        target = target[RR != 0]

        norm, covid = RR[target == 0], RR[target == 1]
        plt.scatter(norm[1:], norm[:len(norm) - 1], color='b')
        plt.scatter(covid[1:], covid[:len(covid) - 1], color='r')

        plt.xlabel("RR(n)")
        plt.ylabel("RR(n-1)")
        plt.title(f"Скатерограмма пациента №{person_id}")
        plt.tight_layout()
        plt.savefig(f'scatter {id}')

    @staticmethod
    def plot_covid(time, RR, target, person_id):
        RR = RR[RR != 0]
        target = target[RR != 0]
        time = time[RR != 0]
        time = time / 60000

        args = np.squeeze(np.argwhere(target == 1))
        RR = RR[args]
        time = time[args]

        dif = np.diff(args)
        dif[dif <= 1] = 0
        dif[dif > 1] = 1

        num_covid = sum(dif) + 1    # first covid
        fig, axs = plt.subplots(3, num_covid)
        fig.set_size_inches(7, 3)

        changes_covids = np.argwhere(dif != 0).squeeze(1) + 1
        times = np.split(time, changes_covids)
        RRs = np.split(RR, changes_covids)

        for i in range(num_covid):
            t, rr = times[i], RRs[i]
            axs[0, i].plot(t, rr)

            derivative_1 = [(rr[j + 1] - rr[j - 1]) / (2 * (t[j + 1] - t[j - 1])) for j in range(1, len(rr) - 1)]
            axs[1, i].plot(t[1:len(rr) - 1], derivative_1)

            derivative_2 = [(rr[j + 1] - 2 * rr[j] + rr[j - 1]) / ((t[j + 1] - t[j - 1]) ** 2) for j in range(1, len(rr) - 1)]
            axs[2, i].plot(t[1:len(rr) - 1], derivative_2)

        plt.tight_layout()
        plt.savefig(f'covid {id}')

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def refresh():
        plt.close('all')

    @staticmethod
    def get_figures():
        return [plt.figure(i) for i in plt.get_fignums()]


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')

    # for id in [4, ]:
    for id in set(df['id']):
        person = df.loc[df['id'] == id]
        time, RR, label = person['time'].to_numpy(), person['x'].to_numpy(), person['y'].to_numpy()
        FigPlotter.plot_ecg(time, RR, label, np.zeros(len(RR)), id)
        # FigPlotter.show()
        FigPlotter.refresh()
        # FigPlotter.plot_covid(time, RR, label, id)
        # FigPlotter.show()
        # FigPlotter.refresh()
        # print(id)

    # parser = argparse.ArgumentParser(description='Dynamic ECG')
    # parser.add_argument('-id', action='store', help='People ID', default=1)
    # # # parser.add_argument('-speed', action='store', help='Speed', default=1)
    # # parser.add_argument('-start', action='store', help='start', default=0)
    # # parser.add_argument('-end', action='store', help='end', default=1e9)
    # args = parser.parse_args()
    #
    # df = pd.read_csv('data/train.csv')
    # person = df.loc[df['id'] == args.id]
    # person['time'] = person['time'] / 60000
    #
    # time, RR, label = person['time'].to_numpy(), person['x'].to_numpy(), person['y'].to_numpy()
    # delta = np.asarray([time[i] - time[i-1] for i in range(1, len(time))])
    # mean_delta, std_delta = delta.mean(), delta.std()
    #
    # plt.rcParams['animation.html'] = 'jshtml'
    # fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(111)
    # plt.xlabel("Время, мин")
    # plt.ylabel("R-R интервал")
    # plt.title(f"Real-time ритмограмма пациента N{1}")
    # plt.tight_layout()
    # fig.show()
    #
    # n_pionts_min = int(len(time) / max(time))
    # n_min_show = 3
    # n_points_show = n_min_show * n_pionts_min
    #
    # time = np.concatenate([-1 * n_min_show * np.arange(0, 1, 1 / n_points_show)[::-1], time])
    # RR = np.concatenate([np.full(n_points_show, RR.mean()), RR])
    # label = np.concatenate([np.zeros(n_points_show), label])
    # plt.plot(time, RR)
    # [plt.fill_between([time[i], ], 0, [RR[i], ], color='red', alpha=0.2) for i in range(len(time)) if label[i] == 1]
    #
    # for i in range(n_points_show, len(time)):
    #     plt.xlim(left=time[i - n_points_show], right=time[i])
    #     print()
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
