import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg

from fxpmath import Fxp

np.random.seed(0)

length = 2
sample_rate = 1000
f = 1
A = 1  # 2**31
time = np.arange(0, length, 1 / sample_rate)
x = A * np.sin(2 * np.pi * f * time)

#plt.plot(time, x)
#plt.show()

snrs = []
snr_refs = []

for n_frac in range(3, 33):
    fxp_ref = Fxp(None, signed=True, n_word=n_frac + 15, n_frac=n_frac)
    x_q = Fxp(x, like=fxp_ref)
    x_qq = x_q.get_val()
    #x_qq = x

    N = 64
    output_ref = np.zeros(len(x))
    output_fxp = np.zeros(len(x))
    state_ref = np.zeros(len(x)+1)
    state_fxp = np.zeros(len(x)+1)
    a = 0.9
    for i in range(len(x)):

        output_ref[i] = state_ref[i] * a + x_qq[i]
        state_ref[i+1] = output_ref[i]

        output_fxp[i] = Fxp(Fxp(state_fxp[i] * a, like=fxp_ref).get_val() + x_qq[i], like=fxp_ref).get_val()
        state_fxp[i+1] = output_fxp[i]

    """plt.plot(x_qq, label="x")
    plt.plot(output_ref, label="reference filter output")
    plt.plot(output_fxp, label="fxp filter output")
    plt.legend()
    plt.savefig("iir_" + str(n_frac) + ".png")
    plt.show()"""

    Ps = np.mean(np.abs(output_ref) ** 2)
    Pn = np.mean(np.abs(output_ref - output_fxp) ** 2)
    Psfxp = np.mean(np.abs(output_fxp) ** 2) - Pn
    Pnexp = (fxp_ref.precision ** 2) / 6 / (1-a**2)

    snr = 10 * np.log10(Psfxp / Pn)
    snr_ref = 10 * np.log10(Ps / Pnexp)
    snrs.append(snr)
    snr_refs.append(snr_ref)
    print(Ps, Pn, Pnexp, snr, snr_ref)


plt.plot(np.arange(3, 33), snrs, label="measured")
plt.plot(np.arange(3, 33), snr_refs, label="reference")
plt.legend()
plt.savefig("iir.png")
plt.show()
