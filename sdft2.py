import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections


np.random.seed(0)
x = np.random.randn(32000).astype(np.single)
#x = np.tile(np.random.randn(1000), 100)



def get_range(k, mul):
    #return np.exp(-1j * 2 * np.pi * k * mul).astype(np.csingle)
    firsthalf = np.exp(-1j * 2 * np.pi * k[:len(k) // 2 + 1] * mul).astype(np.csingle)
    if len(k) % 2 == 0:
        retval = np.concatenate((firsthalf, np.flip(firsthalf[1:-1]).conj().astype(np.csingle))).astype(np.csingle)
    else:
        retval = np.concatenate((firsthalf, np.flip(firsthalf[1:]).conj().astype(np.csingle))).astype(np.csingle)

    return retval

class SDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.last_result = np.zeros(size, dtype=np.csingle)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        new = get_range(np.arange(self.size), 1 / self.size).conj() * (self.last_result + x - lastx)

        self.last_result = new.astype(np.csingle)

        return self.last_result

class mSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.last_result = np.zeros(size, dtype=np.csingle)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        c = get_range(np.arange(self.size), ((self.cnt) % self.size) / self.size).conj()
        g = c.conj()

        new = self.last_result + g * (x - lastx)
        self.last_result = new.astype(np.csingle)

        self.cnt += 1
        return (new * get_range(np.arange(self.size), (self.cnt % self.size) / self.size).conj()).astype(np.csingle)

class oSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.internal = np.zeros(size, dtype=np.csingle)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        c = get_range(np.arange(self.size), ((self.cnt-1) % self.size) / self.size).conj().astype(np.csingle)
        g = c.conj().astype(np.csingle)

        y = (np.sum(c * self.internal) / self.size).astype(np.csingle)
        new = self.internal + (g * (x - y)).astype(np.csingle)


        self.internal = new.astype(np.csingle)

        return (self.internal * get_range(np.arange(self.size), (self.cnt % self.size) / self.size).conj()).astype(np.csingle)

class resoSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.internal = np.zeros(size, dtype=np.csingle)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        y = (np.sum(self.internal) / self.size).astype(np.csingle)
        new = get_range(np.arange(self.size), 1 / self.size).conj().astype(np.csingle) * (self.internal + x - y).astype(np.csingle)


        self.internal = new.astype(np.csingle)

        return (self.internal * get_range(np.arange(self.size), (self.cnt % self.size) / self.size).conj().astype(np.csingle)).astype(np.csingle)


class MOVDFT():
    def __init__(self, size=10, type=np.csingle):
        self.buffer = collections.deque(maxlen=size)
        self.type=type
        self.size = size
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.buffer.popleft()
        self.buffer.append(x)

        return np.fft.fft(np.array(self.buffer).astype(self.type)).astype(self.type)

windowsize = 25

sdft = SDFT(windowsize)
msdft = mSDFT(windowsize)
osdft = oSDFT(windowsize)
resosdft = resoSDFT(windowsize)
movdft_f32 = MOVDFT(windowsize)
movdft_f64 = MOVDFT(windowsize, type=np.cdouble)
res_sdft = np.zeros((len(x), windowsize), dtype=np.csingle)
res_msdft = np.zeros((len(x), windowsize), dtype=np.csingle)
res_osdft = np.zeros((len(x), windowsize), dtype=np.csingle)
res_resosdft = np.zeros((len(x), windowsize), dtype=np.csingle)
res_movdft_f32 = np.zeros((len(x), windowsize), dtype=np.csingle)
res_movdft_f64 = np.zeros((len(x), windowsize), dtype=np.cdouble)

for i, point in enumerate(x):
    res_sdft[i] = sdft.process_point(point)
    res_msdft[i] = msdft.process_point(point)
    res_osdft[i] = osdft.process_point(point)
    res_resosdft[i] = resosdft.process_point(point)
    res_movdft_f32[i] = movdft_f32.process_point(point)
    res_movdft_f64[i] = movdft_f64.process_point(point)

dif_sdft = np.mean(np.abs(res_movdft_f64 - res_sdft), axis=1)
dif_msdft = np.mean(np.abs(res_movdft_f64 - res_msdft), axis=1)
dif_osdft = np.mean(np.abs(res_movdft_f64 - res_osdft), axis=1)
dif_resosdft = np.mean(np.abs(res_movdft_f64 - res_resosdft), axis=1)
dif_movdft = np.mean(np.abs(res_movdft_f64 - res_movdft_f32), axis=1)


def plot_x():
    plt.plot(x, color="y")


def plot_dfts(bin=3):
    #plt.plot(np.abs(res_sdft[:,bin].real), color="r", alpha=0.5)
    plt.plot(np.abs(res_msdft[:,bin].real), color="g", alpha=0.5)
    #plt.plot(np.abs(res_movdft_f32[:,bin].real), color="b", alpha=0.5)
    plt.plot(np.abs(res_movdft_f64[:,bin].real), color="y", alpha=0.5)


def plot_diffs():
    #plt.plot(dif_sdft, color="r", alpha=0.5, label="SDFT error")
    #plt.plot(dif_msdft, color="g", alpha=0.5, label="mSDFT error")
    plt.plot(dif_osdft, color="y", alpha=0.5, label="oSDFT error")
    plt.plot(dif_osdft, color="orange", alpha=0.5, label="Resonator oSDFT error")
    plt.plot(dif_movdft, color="b", alpha=0.5, label="Moving DFT error")
    plt.legend()


plot_diffs()
#plt.savefig("mo-comp-norepeat.png")
plt.show()
