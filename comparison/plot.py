import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(8, 6))


T_1ms = np.loadtxt("./T_2D/res_1ms/Trz_E1.003.csv", skiprows=1, delimiter=",")
r = np.linspace(0, 1.5e-3, 50, endpoint=True)

for i in [225, 250, 300, 350]:
    axs[0][0].plot(r / 1e-3, T_1ms[i, 1::], label=f"t={T_1ms[i, 0]*1e3:.2f} мс")

axs[0][0].set_xlim(0, 1.5)
axs[0][0].legend()

ret_1ms = np.loadtxt(
    "./results_1ms_new1/profiles_1ms_E1.003.txt", skiprows=1, delimiter=","
)
axs[1, 0].plot(ret_1ms[:, 0] / 1e-3, ret_1ms[:, 1])

axs[1][0].set_xlim(0, 1.5)
axs[1][0].set_ylim(bottom=1.5e22)

T_250us = np.loadtxt("./T_2D/res_250us/Trz_E0.351.csv", skiprows=1, delimiter=",")
r = np.linspace(0, 1.5e-3, 50, endpoint=True)

for i in [75, 100, 150, 200]:
    axs[0][1].plot(r / 1e-3, T_250us[i, 1::], label=f"t={T_250us[i, 0]*1e3:.2f} мс")

axs[0][1].set_xlim(0, 1.5)
axs[0][1].legend()

ret_250us = np.loadtxt(
    "./results_250us_new1/profiles_250us_E0.351.txt", skiprows=1, delimiter=","
)
axs[1, 1].plot(ret_250us[:, 0] / 1e-3, ret_250us[:, 1])

axs[1][1].set_xlim(0, 1.5)
axs[1][1].set_ylim(1.5e22, 1.58e22)

axs[0, 0].set_ylabel("Температура, К")
axs[1, 0].set_ylabel(r"Содержание D, м$^{-2}$")
axs[1, 0].set_xlabel("r, мм")
axs[1, 1].set_xlabel("r, мм")

axs[0, 0].set_title(r"1 мс: $E=1,003$ Дж")
axs[0, 1].set_title(r"170 мкс: $E=0,351$ Дж")
plt.tight_layout()

plt.savefig("1.png")
plt.show()
