import numpy as np
import matplotlib.pyplot as plt

FWHM = 1e-3
sigma_r = FWHM / 2 / np.sqrt(2*np.log(2))

profile = lambda r: 1 * np.exp(-r**2 / 2 / sigma_r**2)

r = np.linspace(0,2e-3, num=10000)

plt.plot(r/1e-3, profile(r))
plt.xlabel("r, mm")
plt.ylabel("Heat flux, a.u.")

plt.show()