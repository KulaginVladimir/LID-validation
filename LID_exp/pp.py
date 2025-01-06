import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

pd.options.mode.chained_assignment = None


def get_leak(leak_file):
    data = pd.read_csv(leak_file, skiprows=328, header=None, sep=",").dropna(axis=1)

    data[0] = pd.to_datetime(data[0])
    data[0] = (data[0] - data[0][0]).dt.total_seconds()

    mass3 = data[data[1] == 3.0].reset_index(drop=True).drop(columns=1)
    mass4 = data[data[1] == 4.0].reset_index(drop=True).drop(columns=1)
    mass3 = mass3.rename(columns={0: "Time, s", 2: f"Sygnal, A"})
    mass4 = mass4.rename(columns={0: "Time, s", 2: f"Sygnal, A"})
    # mass3.to_csv("M3.csv")
    # mass4.to_csv("M4.csv")

    plt.plot(mass3["Time, s"], mass3["Sygnal, A"])
    plt.plot(mass4["Time, s"], mass4["Sygnal, A"])
    plt.yscale("log")
    leak = mass4["Sygnal, A"][mass4["Time, s"] > 80].mean()

    print(leak)
    return leak


def pp(datas, leak, is_plot=True):
    period = 10
    k = 1.57005e14
    # leak = 9.47e-10

    cutted_data = []
    fluences = []
    max_t = datas[1]["Time, s"][
        datas[1]["Sygnal, A"] == max(datas[1]["Sygnal, A"])
    ].iloc[0]

    for i, data in enumerate(datas):
        data = data.rolling(10).mean()

        data["Sygnal, A"] = (
            data["Sygnal, A"] - data["Sygnal, A"][data["Time, s"] <= 1].mean()
        )
        noise_max = data["Sygnal, A"][data["Time, s"] <= 2].max()

        cutted = data[
            (data["Time, s"] <= max_t + period * 2 / 3) & (data["Time, s"] > max_t - 2)
        ]

        if is_plot:
            (l1,) = plt.plot(data["Time, s"], data["Sygnal, A"], label=f"Масса: {i+3}")
            # plt.axhline(noise_max, ls="dashed", color=l1.get_color())

        if noise_max * 5 >= cutted["Sygnal, A"].max():
            fluences.append(0)
        else:
            left_tzero = cutted["Time, s"][
                (cutted["Time, s"] < max_t) & (cutted["Sygnal, A"] <= 0)
            ].iloc[-1]
            try:
                right_tzero = cutted["Time, s"][
                    (cutted["Time, s"] > max_t) & (cutted["Sygnal, A"] <= 0)
                ].iloc[0]
            except:
                right_tzero = cutted["Time, s"].max()

            fluences.append(
                np.trapz(
                    y=cutted["Sygnal, A"][
                        (cutted["Time, s"] >= left_tzero)
                        & (cutted["Time, s"] <= right_tzero)
                    ],
                    x=cutted["Time, s"][
                        (cutted["Time, s"] >= left_tzero)
                        & (cutted["Time, s"] <= right_tzero)
                    ],
                )
                * k
                / leak
                / 2
                * (i + 1)
            )
            # if is_plot:
            # plt.axvline(left_tzero, ls="dashed", color=l1.get_color())
            # plt.axvline(right_tzero, ls="dashed", color=l1.get_color())
    if is_plot:
        # plt.axhline(0, ls="dashed", color="black")
        plt.xlabel("Время, с")
        plt.ylabel("Сигнал")
        plt.legend()
        plt.show()
        print(fluences)
    return cutted_data, fluences


def get_fluences(folder, name, leak):
    for file in sorted(os.listdir(folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith(name):
            data = pd.read_csv(
                folder + filename, skiprows=328, header=None, sep=","
            ).dropna(axis=1)

            data[0] = pd.to_datetime(data[0])
            data[0] = (data[0] - data[0][0]).dt.total_seconds()

            mass3 = data[data[1] == 3.0].reset_index(drop=True).drop(columns=1)
            mass4 = data[data[1] == 4.0].reset_index(drop=True).drop(columns=1)
            mass3 = mass3.rename(columns={0: "Time, s", 2: f"Sygnal, A"})
            mass4 = mass4.rename(columns={0: "Time, s", 2: f"Sygnal, A"})

            separated_data = [mass3, mass4]

            separated_data, fluence_1 = pp(separated_data, leak, True)

            F = 0
            for i, fl in enumerate(fluence_1):
                if fl >= 0:
                    F += fl

            fluences.append(F)
    return fluences


folder = "./250 mks/"
name = "50824"
fluences = []

leak = get_leak(folder + "leak.csv")
fluences = get_fluences(folder, name, leak)
energies = [0.054, 0.078, 0.124, 0.171, 0.216, 0.261, 0.306, 0.351]  # 250 us
# fluences = [0, 0, 5133716691344.515, 18532887681986.27, 10540481234968.29, 60643751401553.06, 59098320981498.01, 81554457549307.39]

plt.errorbar(
    energies,
    np.array(fluences),
    yerr=np.array(fluences) * 0.16,
    fmt="o",
    markersize=5,
    capsize=6,
    label="250 us",
    # s=25,
)
plt.show()

folder = "./1 ms/"
name = "60824"
fluences = []

leak = get_leak(folder + "leak.csv")
fluences = get_fluences(folder, name, leak)
energies = [0.066, 0.16, 0.255, 0.34951, 0.444, 0.53771, 0.636, 0.823, 1.003]  # 1 ms
# fluences = [0, 0, 1054478097062.3062, 2401476299662.052, 1507355677569.9465, 6853539178657.508, 12628691889725.508, 64106569074151.78, 239789819543843.44]

plt.errorbar(
    energies,
    np.array(fluences),
    yerr=np.array(fluences) * 0.16,
    fmt="o",
    markersize=5,
    capsize=6,
    label="1 ms",
    # s=25,
)

# energies = [1.2987984, 2.06995995, 2.06995995, 2.06995995, 2.83714393, 3.60416556, 4.364855548, 5.16272364, 6.68069427, 8.14184247]


plt.legend()
# plt.yscale("log")
plt.show()
