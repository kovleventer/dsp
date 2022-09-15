import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg
from fxpmath import Fxp

np.random.seed(0)

x = np.arange(16)/16

N = 8

n_frac = 6
fxp_ref = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='around')

def WK():
    return np.exp(1j * 2 * np.pi * np.arange(N) / N)

def Q(x):
    d = 0#np.random.uniform(-fxp_ref.precision / 2, fxp_ref.precision / 2)
    ret = Fxp(x + d, like=fxp_ref).get_val()
    print("Before quantization: " + str(x) + ", after quantization: " + str(ret) + ", difference: " + str(x-ret))
    return Fxp(x + d, like=fxp_ref).get_val()

buffer = collections.deque(maxlen=N)
for i in range(N):
    buffer.append(0)

wk = WK()
fxp_ref_2 = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='trunc')
wk = Fxp(wk, like=fxp_ref_2).get_val()

res_sdft = np.zeros((len(x), N), dtype=np.cdouble)
res_movdft_f64 = np.zeros((len(x), N), dtype=np.cdouble)
sdft_last_res = np.zeros(N)

for i, xx in enumerate(x):
    lastx = buffer.popleft()
    buffer.append(xx)

    res_sdft[i] = Q(wk * Q(sdft_last_res + (xx - lastx)))
    #res_sdft[i] = Q(wk * Q(sdft_last_res + (xx - Q(np.sum(sdft_last_res)/N))))

    #res_sdft[i] = wk * (sdft_last_res + (xx - lastx))
    sdft_last_res = res_sdft[i]
    sdft_fft_res = np.fft.fft(np.array(buffer))
    res_movdft_f64[i] = sdft_fft_res
    #if i % 10000 == 0: print(i)

r = np.abs(res_movdft_f64 - res_sdft)
y = np.mean(r, axis=1)
