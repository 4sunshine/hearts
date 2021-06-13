import matplotlib.pyplot as plt
import pandas as pd


def plot_person_ecg(df, id):
    person = df.loc[df['id'] == id]
    time = person['time'].to_numpy() * 1e-3
    RR = person['x'].to_numpy()
    label = person['y'].to_numpy()

    plt.figure(figsize=(15, 5))
    plt.plot(time, RR)
    plt.xlim(time.min(), time.max())
    [plt.fill_between([time[i], ], 0, [RR[i], ], color='red') for i in range(len(person)) if label[i] == 1]

    plt.title(f'Ритмограмма {id} пациента')
    plt.ylabel('R-R интервал')
    plt.xlabel('Время, c')
    plt.show()




if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    plot_person_ecg(df, 1)
