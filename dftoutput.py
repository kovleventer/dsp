import collections

from matplotlib import pyplot as plt

from forcedarray import csZeros, csArray, csExp, csSum, csDot

import numpy as np

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

np.random.seed(0)
x = np.random.uniform(0, 1, 100000).astype(np.single)
#x = np.tile(np.random.randn(1000), 100)
N = 256

class SDFT():
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.last_result = csZeros(size)
        self.cnt = 0
        for i in range(size):
            self.buffer.append(0)
        self.wk = get_range(csArray(np.arange(self.size)), 1 / self.size).conj()

    def process_point(self, x):
        self.cnt += 1
        lastx = self.buffer.popleft()
        self.buffer.append(x)

        new = self.wk * (self.last_result + (x - lastx) * (1 / self.size))
        new = fix_array(new)
        self.last_result = new

        return self.last_result.array



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

        return fix_array2(np.fft.fft(np.array(self.buffer).astype(self.type)).astype(self.type) / self.size)

sdft = SDFT(N)
movdft_f64 = MOVDFT(N, type=np.cdouble)
res_sdft = np.zeros((len(x), N), dtype=np.csingle)
res_movdft_f64 = np.zeros((len(x), N), dtype=np.cdouble)

for i, point in enumerate(x):
    res_sdft[i] = sdft.process_point(point)
    res_movdft_f64[i] = movdft_f64.process_point(point)

r = np.abs(res_movdft_f64 - res_sdft)
l = []
for i in range(N):
    res = r[-1, i]
    plt.plot(r[:, i], alpha=0.5)
    print(i, res)
    l.append(res)
    #plt.show()

plt.show()
plt.plot(l)
plt.show()

