from matplotlib import pyplot as plt
import numpy as np
from fxpmath import Fxp
import collections

np.random.seed(0)
#x = np.random.uniform(0, 1, 100000).astype(np.single)
x = np.tile(np.random.randn(1000), 100)
N = 64

def WK():
    return np.exp(1j * 2 * np.pi * np.arange(N) / N)

def WK2():
    t = np.exp(1j * 2 * np.pi * (np.arange(N//2-1)+1) / N)
    return np.concatenate([np.array([1]), t, np.array([-1]), t[::-1].conj()])

def WK4():
    t = np.exp(1j * 2 * np.pi * (np.arange(N // 4 - 1) + 1) / N)
    return np.concatenate([np.array([1]), t, np.array([1j]), -t[::-1].conj(), np.array([-1]), -t, np.array([-1j]), t[::-1].conj()])

n_frac = 16
fxp_ref = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='around')

def Q(x):
    return Fxp(x, like=fxp_ref).get_val()

x = Q(x)

res_sdft = np.zeros((len(x), N), dtype=np.cdouble)
res_movdft_f64 = np.zeros((len(x), N), dtype=np.cdouble)

sdft_last_res = np.zeros(N)
buffer = collections.deque(maxlen=N)
for i in range(N):
    buffer.append(0)

wk = WK()
wk2 = wk
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
plt.scatter(wk.real, wk.imag, label="reference")
fxp_ref_2 = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='trunc')
wk = Fxp(wk, like=fxp_ref_2).get_val()
plt.scatter(wk.real, wk.imag, label="quantized")
l = 2**(n_frac+1)+1
if l < 1000:
    plt.hlines(np.linspace(-1, 1, l), -1, 1, color="black", alpha=0.5)
    plt.vlines(np.linspace(-1, 1, l), -1, 1, color="black", alpha=0.5)
plt.legend()
plt.show()


for i, xx in enumerate(x):
    lastx = buffer.popleft()
    buffer.append(xx)

    res_sdft[i] = Q(wk * Q(sdft_last_res + (xx - lastx)))
    #res_sdft[i] = wk * (sdft_last_res + (xx - lastx))
    sdft_last_res = res_sdft[i]
    sdft_fft_res = np.fft.fft(np.array(buffer))
    res_movdft_f64[i] = sdft_fft_res
    if i % 1000 == 0: print(i)

r = np.abs(res_movdft_f64 - res_sdft)
l = []
for i in range(N):
    res = r[-1, i]
    plt.plot(r[:, i], alpha=0.5)
    print(i, res)
    l.append(res)

plt.savefig("REPEAT_BINS_Q" + str(n_frac) + "_N" + str(N) + ".png")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylabel("SDFT bin difference", color='b')
ax1.plot(l, color="b")
ax2.set_ylabel("Wk quant. error", color='r')
ax2.plot(np.abs(wk-wk2), color="r")
fig.tight_layout()
plt.savefig("REPEAT_ERRORS_Q" + str(n_frac) + "_N" + str(N) + ".png")
plt.show()

