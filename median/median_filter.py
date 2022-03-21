import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg

from fxpmath import Fxp
np.random.seed(0)

length = 2
sample_rate = 1000
f = 5
A = 1#2**31
time = np.arange(0, length, 1/sample_rate)
x = A * np.sin(2 * np.pi * f * time)

#plt.plot(time, x)
#plt.show()

snrs = []
snr_refs = []

for n_frac in range(3, 33):

    fxp_ref = Fxp(None, signed=True, n_word=48, n_frac=n_frac)
    x_q = Fxp(x, like=fxp_ref)

    N = 64
    output_ref = np.zeros(len(x)-N)
    output_fxp = Fxp(output_ref, like=fxp_ref)
    for i in range(len(x)-N):
        #output_ref[i] = np.mean(x[i:i+N])
        #output_fxp[i] = np.mean(x_q[i:i+N])

        output_ref[i] = np.sum(x[i:i+N] /N)
        output_fxp[i] = np.sum(x_q[i:i+N] /N)

        #for j in range(N):
        #    output_ref[i] += x[i+j] / N
        #    output_fxp[i] += x_q[i+j] / N

    output_fxp = output_fxp.get_val()
    #plt.plot(output_ref, alpha=0.5)
    #plt.plot(output_fxp, alpha=0.5)
    #plt.show()

    Ps = np.mean(np.abs(output_ref)**2)
    Pn = np.mean(np.abs(output_ref - output_fxp)**2)
    Pnexp = (fxp_ref.precision**2)/12 * (N+1+1/N)
    
    snr = 10 * np.log10(Ps/Pn)
    snr_ref = 10 * np.log10(Ps/Pnexp)
    snrs.append(snr)
    snr_refs.append(snr_ref)
    print(Ps, Pn, Pnexp, snr, snr_ref)
    #break


plt.plot(np.arange(3, 33), snrs)
plt.plot(np.arange(3, 33), snr_refs)
#plt.show()
plt.savefig("manual.png")
