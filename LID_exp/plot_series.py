import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

pd.options.mode.chained_assignment = None

folder = "./250 mks IPG/"
fluences = []

for file in sorted(os.listdir(folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".csv") and filename.startswith("50824"):
        data = pd.read_csv(
            folder + filename, skiprows=328, header=None, sep=","
        ).dropna(axis=1)

        data[0] = pd.to_datetime(data[0])
        data[0] = (data[0] - data[0][0]).dt.total_seconds()

        mass3 = data[data[1] == 3.0].reset_index(drop=True).drop(columns=1)
        mass4 = data[data[1] == 4.0].reset_index(drop=True).drop(columns=1)

        separated_data = [mass3, mass4]
        fluence_1 = []

        for i, data in enumerate(separated_data):
            separated_data[i] = data.rename(columns={0: "Time, s", 2: f"Sygnal, A"})
            plt.plot(separated_data[i]["Time, s"], separated_data[i]["Sygnal, A"])

        plt.show()
