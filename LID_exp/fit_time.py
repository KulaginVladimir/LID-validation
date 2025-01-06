import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt("./1ms.csv", delimiter=",")


def fit_func(t, t1, t2, dt1, delta, dt2):
    return np.piecewise(
        t,
        [t <= t1, (t > t1) & (t <= t2), t > t2],
        [
            lambda t: 1 / (1 + np.exp(-(t - 0.5 * t1) / dt1)),
            lambda t: 1
            / (1 + np.exp(-(t1 - 0.5 * t1) / dt1))
            * (1 - delta * (t - t1) / (t2 - t1)),
            lambda t: 1
            / (1 + np.exp(-(t1 - 0.5 * t1) / dt1))
            * (1 - delta)
            * np.exp(-(t - t2) / dt2),
        ],
    )


# popt, pcov = curve_fit(fit_func, data[:,0], data[:, 1], p0 = [1e-4, 1e-3, 2e-6, 30, 2e-4]) # for 10 ms
popt, pcov = curve_fit(
    fit_func, data[:, 0], data[:, 1], p0=[1e-4, 1e-3, 2e-6, 30, 2e-4]
)  # for 1 ms


print(popt)
p_us = [
    2.00546391e-04,
    170e-6,
    3.38692875e-07,
    3.23973615e-02,
    1.97812839e-04,
]

fig = plt.figure(figsize=(5, 4))
plt.plot(data[:, 0], data[:, 1], label="Experiment")
plt.plot(data[:, 0], fit_func(data[:, 0], *popt), label="Fit")
plt.plot(
    data[:, 0],
    fit_func(data[:, 0], *p_us),
    label="Fit1",
)
plt.xlabel("Time, s")
plt.ylabel("Sygnal, a.u.")
plt.legend()
plt.savefig("./10ms_time.png")

plt.xlim(0, 2.5e-3)
plt.show()

times = np.linspace(0, 12e-3, 100000)

int = np.trapz(fit_func(times, *popt), x=times)
print(f"Integral = {int:.7f}")
int1 = np.trapz(fit_func(times, *popt) / int, x=times)
print(f"Integral = {int1:.7f}")

int = np.trapz(fit_func(times, *p_us), x=times)
print(f"Integral = {int:.7f}")
int1 = np.trapz(fit_func(times, *p_us) / int, x=times)
print(f"Integral = {int1:.7f}")
