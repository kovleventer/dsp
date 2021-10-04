import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections

t = np.linspace(0, 10, 2000)
x = np.concatenate([
    np.sin(2*np.pi*t),
    scipy.signal.square(2 * np.pi * t),
    scipy.signal.sawtooth(2 * np.pi * t)
])

#x = np.arange(10)+1

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
        if self.cnt == self.size:
            new = np.fft.fft(np.array(self.buffer))
        else:
            new = np.exp(1j * 2 * np.pi * np.arange(self.size) / self.size) * (self.last_result + x - lastx)

        self.last_result = new

        return new

class mSDFT():
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
        if self.cnt == self.size:
            new = np.fft.fft(np.array(self.buffer))
        else:
            new = self.last_result + np.exp(-1j * 2 * np.pi * np.arange(self.size) * (self.cnt-1) / self.size) * (x - lastx)

        self.last_result = new

        return new * np.exp(1j * 2 * np.pi * np.arange(self.size) * (self.cnt-1) / self.size)

class MOVDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.buffer.popleft()
        self.buffer.append(x)

        return np.fft.fft(np.array(self.buffer))

windowsize = 5
plotted_bin = 3

sdft = SDFT(windowsize)
msdft = mSDFT(windowsize)
movdft = MOVDFT(windowsize)
res_sdft = np.zeros_like(x, dtype=np.csingle)
res_msdft = np.zeros_like(x, dtype=np.csingle)
res_movdft = np.zeros_like(x, dtype=np.csingle)

for i, point in enumerate(x):
    res_sdft[i] = sdft.process_point(point)[plotted_bin]
    res_msdft[i] = msdft.process_point(point)[plotted_bin]
    res_movdft[i] = movdft.process_point(point)[plotted_bin]

dif_sdft_movdft = np.abs(res_sdft - res_movdft)
dif_sdft_msdft = np.abs(res_sdft - res_msdft)
dif_msdft_movdft = np.abs(res_msdft - res_movdft)

dif_alter_sdft_movdft = np.abs(np.abs(res_sdft) - np.abs(res_movdft))
dif_alter_sdft_msdft = np.abs(np.abs(res_sdft) - np.abs(res_msdft))
dif_alter_msdft_movdft = np.abs(np.abs(res_msdft) - np.abs(res_movdft))

def plot_x():
    plt.plot(x, color="y")

def plot_dfts():
    plt.plot(np.abs(res_sdft), color="r", alpha=0.5)
    plt.plot(np.abs(res_msdft), color="g", alpha=0.5)
    plt.plot(np.abs(res_movdft), color="b", alpha=0.5)

def plot_diffs():
    plt.plot(dif_msdft_movdft, color="r", alpha=0.5)
    plt.plot(dif_sdft_movdft, color="g", alpha=0.5)
    plt.plot(dif_sdft_msdft, color="b", alpha=0.5)

def plot_alter_diffs():
    plt.plot(dif_alter_msdft_movdft, color="r", alpha=0.5)
    plt.plot(dif_alter_sdft_movdft, color="g", alpha=0.5)
    plt.plot(dif_alter_sdft_msdft, color="b", alpha=0.5)

plot_alter_diffs()
plt.show()