import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections

from forcedarray import csZeros, csArray, csExp, csSum

np.random.seed(0)
x = np.random.randn(32000).astype(np.single)
#x = np.tile(np.random.randn(1000), 100)



def get_range(k, mul):
    #return csExp(-1j * 2 * np.pi * k * mul)
    firsthalf = csExp(-1j * 2 * np.pi * csArray(k.array[:len(k.array) // 2 + 1]) * mul)
    if len(k.array) % 2 == 0:
        retval = csArray(np.concatenate((firsthalf.array, np.flip(firsthalf.array[1:-1].conj()))))
    else:
        retval = csArray(np.concatenate((firsthalf.array, np.flip(firsthalf.array[1:].conj()))))

    return retval

def fix_array(a):
    return csArray(np.concatenate((a.array[:len(a.array)//2+1], np.flip(a.array[1:len(a.array)//2].conj()))))

def fix_array2(a):
    return np.concatenate((a[:len(a)//2+1], np.flip(a[1:len(a)//2].conj())))

class SDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.last_result = csZeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)

    def process_point(self, x):
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        new = get_range(csArray(np.arange(self.size)), 1 / self.size).conj() * (self.last_result + x - lastx)
        new = fix_array(new)
        self.last_result = new

        return self.last_result.array

class mSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.internal = csZeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0.)

    def process_point(self, x):
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        g = get_range(csArray(np.arange(self.size)), ((self.cnt) % self.size) / self.size)
        new = self.internal + g * (x - lastx)
        new = fix_array(new)
        self.internal = new

        self.cnt += 1
        return fix_array((self.internal * get_range(csArray(np.arange(self.size)), (self.cnt % self.size) / self.size).conj())).array

class oSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.internal = csZeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0.)

    def process_point(self, x):
        self.cnt += 1
        self.buffer.popleft()
        self.buffer.append(x)

        g = get_range(csArray(np.arange(self.size)), ((self.cnt-1) % self.size) / self.size)
        c = g.conj()

        y = (csSum(c * self.internal) / self.size)
        new = self.internal + (g * (x - y))
        new = fix_array(new)
        self.internal = new

        return fix_array((self.internal * get_range(csArray(np.arange(self.size)), (self.cnt % self.size) / self.size).conj())).array

class resoSDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.internal = csZeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0.)

    def process_point(self, x):
        self.cnt += 1
        self.buffer.popleft()
        self.buffer.append(x)

        y = (csSum(self.internal) / self.size)
        new = get_range(csArray(np.arange(self.size)), 1 / self.size).conj() * (self.internal + x - y)
        new = fix_array(new)
        self.internal = new

        return fix_array(self.internal).array


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

        return fix_array2(np.fft.fft(np.array(self.buffer).astype(self.type)).astype(self.type))

windowsize = 64

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

dif_selt_movdft64 = np.mean(np.abs(res_movdft_f64[:, 1:windowsize//2] - np.flip(res_movdft_f64[:, windowsize//2+1:].conj(), axis=1)), axis=1)
dif_selt_movdft32 = np.mean(np.abs(res_movdft_f32[:, 1:windowsize//2] - np.flip(res_movdft_f32[:, windowsize//2+1:].conj(), axis=1)), axis=1)
dif_selt_sdft = np.mean(np.abs(res_sdft[:, 1:windowsize//2] - np.flip(res_sdft[:, windowsize//2+1:].conj(), axis=1)), axis=1)
dif_selt_msdft = np.mean(np.abs(res_msdft[:, 1:windowsize//2] - np.flip(res_msdft[:, windowsize//2+1:].conj(), axis=1)), axis=1)
dif_selt_osdft = np.mean(np.abs(res_osdft[:, 1:windowsize//2] - np.flip(res_osdft[:, windowsize//2+1:].conj(), axis=1)), axis=1)
dif_selt_resosdft = np.mean(np.abs(res_resosdft[:, 1:windowsize//2] - np.flip(res_resosdft[:, windowsize//2+1:].conj(), axis=1)), axis=1)

#a = res_movdft_f64[:, 1:windowsize//2]
#b = np.flip(res_movdft_f64[:, windowsize//2+1:].conj(), axis=1)
def plot_x():
    plt.plot(x, color="y")


def plot_dfts(bin=3):
    #plt.plot(np.abs(res_sdft[:,bin].real), color="r", alpha=0.5)
    plt.plot(np.abs(res_msdft[:,bin].real), color="g", alpha=0.5)
    #plt.plot(np.abs(res_movdft_f32[:,bin].real), color="b", alpha=0.5)
    plt.plot(np.abs(res_movdft_f64[:,bin].real), color="y", alpha=0.5)


def plot_diffs():
    #plt.plot(dif_sdft, color="r", alpha=0.5, label="SDFT error")
    plt.plot(dif_msdft, color="g", alpha=0.5, label="mSDFT error")
    plt.plot(dif_osdft, color="y", alpha=0.5, label="oSDFT error")
    plt.plot(dif_resosdft, color="orange", alpha=0.5, label="Resonator oSDFT error")
    plt.plot(dif_movdft, color="b", alpha=0.5, label="Moving DFT error")
    plt.legend()

def plot_selfdiffs():
    #plt.plot(dif_selt_sdft, color="r", alpha=1, label="SDFT error")
    plt.plot(dif_selt_msdft, color="g", alpha=1, label="mSDFT error")
    plt.plot(dif_selt_osdft, color="y", alpha=1, label="oSDFT error")
    plt.plot(dif_selt_resosdft, color="orange", alpha=1, label="Resonator oSDFT error")
    plt.plot(dif_selt_movdft32, color="b", alpha=1, label="Moving DFT error")
    plt.legend()

plot_diffs()
#plt.savefig("forced_csingle_repeat.png")
plt.show()
