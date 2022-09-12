import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg

from fxpmath import Fxp

def get_range(k, mul):
    return np.exp(-1j * 2 * np.pi * k * mul)
    firsthalf = np.exp(-1j * 2 * np.pi * k[:len(k) // 2 + 1] * mul)
    if len(k) % 2 == 0:
        retval = np.concatenate((firsthalf, np.flip(firsthalf[1:-1].conj())))
    else:
        retval = np.concatenate((firsthalf, np.flip(firsthalf[1:].conj())))

    return retval

x = np.random.uniform(0, 1, 100000).astype(np.single)
windowsize = 64
class SDFT():
    def __init__(self, size=10, nfrac=8):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.last_result = np.zeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)
        self.wk = get_range(Fxp(np.arange(self.size), signed=True, n_word=32, n_frac=nfrac), 1 / self.size).conj()

    def process_point(self, x):
        x = Fxp(x, like=self.wk)
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        new = self.wk * (self.last_result + (x - lastx) * (1 / self.size))
        self.last_result = new
        return self.last_result

class MOVDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.buffer.popleft()
        self.buffer.append(x)

        return np.fft.fft(np.array(self.buffer)) / self.size



sdft = SDFT(windowsize)
res_sdft = np.zeros((len(x), windowsize), dtype=np.cdouble)
movdft_f64 = MOVDFT(windowsize)
res_movdft_f64 = np.zeros((len(x), windowsize), dtype=np.cdouble)


for i, point in enumerate(x):
    res_sdft[i] = sdft.process_point(point)
    res_movdft_f64[i] = movdft_f64.process_point(point)

dif_sdft = np.mean(np.abs(res_movdft_f64 - res_sdft), axis=1)
plt.plot(dif_sdft)
plt.show()
