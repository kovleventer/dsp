import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg

from fxpmath import Fxp

x = np.random.uniform(0, 1, 1000)
N = 64
for j in range(3, 16):
    fxp_ref = Fxp(None, signed=True, n_word=32, n_frac=j)
    x_fp = Fxp(x, like=fxp_ref)

    res_f64 = []
    res_fxp = []
    for i in range(1000-N):
        res_f64.append(np.fft.fft(x[i:i+N]))
        res_fxp.append(np.fft.fft(x_fp[i:i + N]))
    res_f64_a = np.array(res_f64)
    res_fxp_a = np.array(res_fxp).astype(complex)
    #dif = np.sum(np.abs(res_f64_a[-1] - res_fxp_a[-1])**2, axis=1)
    dif = np.abs(res_f64_a[-1] - res_fxp_a[-1])**2

    print(j, np.mean(dif), (fxp_ref.precision**2/12*2)*(np.log(N)-2))
    #plt.plot(dif)
plt.show()
