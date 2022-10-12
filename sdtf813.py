import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import collections
import scipy.linalg
from fxpmath import Fxp

#np.random.seed(0)
#x_orig = np.sin(np.arange(100)/8 * 2 * np.pi)
x_orig = np.random.random(100000)

#N = 63

#for N in [7, 8, 9, 16, 20, 21, 32, 63, 64, 96, 99, 102, 108, 111, 128]:
for N in [99]:

    if N % 4 == 0:
        Nq = N-4
    elif N % 2 == 0:
        Nq = N-2
    else:
        Nq = N-1

    n_frac = 8
    #for n_frac in range(10, 17):
    fxp_ref = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='around')
    x = Fxp(x_orig, like=fxp_ref).get_val()

    def WK():
        return np.exp(1j * 2 * np.pi * np.arange(N) / N)

    q1s = []
    def Q1(x):
        d = np.random.uniform(-fxp_ref.precision / 2, fxp_ref.precision / 2)
        ret = Fxp(x + d, like=fxp_ref).get_val()

        #print("Q1 Before quantization: " + str(x) + ", after quantization: " + str(ret) + ", difference: " + str(x-ret))
        #print("Q1 difference: " + str(x-ret), np.mean(x-ret))
        q1s.append(x-ret)
        return ret, x-ret

    q2s = []
    def Q2(x):
        d = np.random.uniform(-fxp_ref.precision / 2, fxp_ref.precision / 2)
        ret = Fxp(x + d, like=fxp_ref).get_val()

        #print("Q2 Before quantization: " + str(x) + ", after quantization: " + str(ret) + ", difference: " + str(x - ret))
        q2s.append(x - ret)
        return ret

    buffer = collections.deque(maxlen=N)
    for i in range(N):
        buffer.append(0)

    wk = WK()
    fxp_ref_2 = Fxp(None, signed=True, n_word=n_frac + 10, n_frac=n_frac, rounding='trunc')
    #wk = Fxp(wk, like=fxp_ref_2).get_val()

    res_sdft = np.zeros((len(x), N), dtype=np.cdouble)
    res_movdft_f64 = np.zeros((len(x), N), dtype=np.cdouble)
    sdft_last_res = np.zeros(N)

    ss = []
    qus = []
    diffs = []

    a = 0

    for i, xx in enumerate(x):
        #if i == 32: break
        lastx = buffer.popleft()
        buffer.append(xx)

        #print("i, xx, lastx: ", i,  xx-lastx)

        s = (np.sum((res_sdft[i-1])))/N
        diff = s-lastx
        diffs.append(diff)
        #s = lastx



        #temp = wk * (sdft_last_res + (xx - lastx))

        if i % 100 == 1:
            temp = wk * Q2(sdft_last_res + (xx - Q2(s)))
            res, qu = Q1(temp)
            #res -= wk * diff
        else:
            temp = wk * (sdft_last_res + (xx - lastx))
            res = temp
            res -= wk * diff

        sdft_fft_res = np.fft.fft(np.array(buffer))
        res_sdft[i] = res

        #ss.append(s)
        #qus.append(qu)
        #a += qu-wk*diff
        if i % 1000 == 0:
            print(i)
        #res_sdft[i] = Q1(wk * Q2(sdft_last_res + (xx - lastx)))
        #res_sdft[i] = Q(wk * Q(sdft_last_res + (xx - Q(np.sum(sdft_last_res)/N))))

        #res_sdft[i] = wk * (sdft_last_res + (xx - lastx))
        sdft_last_res = res_sdft[i]

        res_movdft_f64[i] = sdft_fft_res

        def plot():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')
            plt.scatter(res_sdft[i].real, res_sdft[i].imag, color="r", alpha=0.5)
            plt.scatter(res_movdft_f64[i].real, res_movdft_f64[i].imag, color="b", alpha=0.5)
            plt.scatter(temp.real, temp.imag, color="g", alpha=0.5)
            l = 2 ** (n_frac + 1)
            if l < 1000:
                plt.hlines(np.linspace(-4, 4, l*4+1), -4, 4, color="black", alpha=0.2)
                plt.vlines(np.linspace(-4, 4, l*4+1), -4, 4, color="black", alpha=0.2)
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.show()
        #if i % 10000 == 0: print(i)

    r = np.power(np.abs(res_movdft_f64 - res_sdft), 2)
    y = np.mean(r, axis=1)


    plts = []
    for j in range(25):
        plts.append(0)
    for xx in range(len(x)//100):
        for j in range(25):
            plts[j] += y[xx*100+j] / (len(x)//100)
            #plts[j] = diffs[xx*100+j]


    #plt.plot(plts)

    plt.hlines(fxp_ref.precision**2/12*6*Nq/N, 0, 25, 'r')
    plt.hlines(np.mean(y), 0, 25, 'g')
    #plt.plot(y, alpha=0.5)
    plt.plot(plts, alpha=0.5)


    #plt.plot(np.abs(res_movdft_f64[:, 1]), label="MOVDFTn_frac=" + str(n_frac))
    #plt.savefig("average_variance7.png")
    plt.savefig(f"plt_{N}.png")
    plt.clf()
    #plt.show()

    #plt.show()
    print(y)
