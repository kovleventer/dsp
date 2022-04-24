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
A = 1/20  # 2**31
time = np.arange(0, length, 1 / sample_rate)
x = A * np.sin(2 * np.pi * f * time)

#plt.plot(time, x)
#plt.show()

snrs = []
snr_refs = []
snr_dithers = []

for n_frac in range(3, 33):
    fxp_ref = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='around')

    def Q(x):
        return Fxp(x, like=fxp_ref).get_val()

    q = fxp_ref.precision
    x_q = Fxp(x, like=fxp_ref)
    x_qq = x_q.get_val()
    #x_qq = x

    N = 64
    output_ref = np.zeros(len(x), dtype=np.cdouble)
    output_fxp = np.zeros(len(x), dtype=np.cdouble)
    output_fxp_d = np.zeros(len(x), dtype=np.cdouble)
    state_ref = np.zeros(len(x)+1, dtype=np.cdouble)
    state_fxp = np.zeros(len(x)+1, dtype=np.cdouble)
    state_fxp_d = np.zeros(len(x) + 1, dtype=np.cdouble)

    a = 0.85 * np.exp(1j*67/180 * np.pi)

    ar = a.real
    ai = a.imag
    for i in range(len(x)):

        output_ref[i] = state_ref[i] * a + x_qq[i]
        state_ref[i+1] = output_ref[i]

        re = Q( Q( Q(state_fxp[i].real * ar) - Q(state_fxp[i].imag * ai) ) + x_qq[i] )
        im = Q( Q( Q(state_fxp[i].imag * ar) + Q(state_fxp[i].real * ai) ) )
        output_fxp[i] = re + 1j * im
        state_fxp[i+1] = output_fxp[i]

        #d1 = np.random.uniform(-q/2, q/2)
        #d2 = np.random.uniform(-q/2, q/2)
        #output_fxp_d[i] = Q(Q(state_fxp_d[i] * a + d1) + x_qq[i] + d2)
        #state_fxp_d[i + 1] = output_fxp_d[i]

    """plt.plot(x_qq, label="x")
    plt.plot(output_ref, label="reference filter output")
    plt.plot(output_fxp, label="fxp filter output")
    plt.plot(output_fxp_d, label="fxp dither filter output")
    plt.legend()
    plt.savefig("iir_" + str(n_frac) + "_dither.png")
    plt.show()"""

    Ps = np.mean(np.abs(output_ref) ** 2)
    Pn = np.mean(np.abs(output_ref - output_fxp) ** 2)
    Pndither = np.mean(np.abs(output_ref - output_fxp_d) ** 2)
    Psfxp = np.mean(np.abs(output_fxp) ** 2) - Pn
    Psdfxp = np.mean(np.abs(output_fxp_d) ** 2) - Pn
    Pnexp = (q ** 2) / 12 * 2 * 2 / (1-np.abs(a)**2)


    snr = 10 * np.log10(Psfxp / Pn)
    snr_ref = 10 * np.log10(Ps / Pnexp)
    snr_dither = 10 * np.log10(Psdfxp / Pndither)
    snrs.append(snr)
    snr_refs.append(snr_ref)
    snr_dithers.append(snr_dither)
    print(Ps, Pn, Pnexp, snr, snr_ref)
    #print(71+10*np.log10(1-a**2), q)

    #break


plt.plot(np.arange(3, 33), snrs, label="measured")
plt.plot(np.arange(3, 33), snr_refs, label="reference")
#plt.plot(np.arange(3, 33), snr_dithers, color="r", label="dither")
plt.legend()
plt.savefig("ciir.png")
plt.show()
