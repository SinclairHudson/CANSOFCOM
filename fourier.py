import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from signalgenerator import psigenerator, generateData
from configurations import *

sample_length = 1.05 # in seconds
f_s = 100_000  # sample frequency, in hz

config = dict(scenario, **djimavicair2)
# config = paperconfig1
p = psigenerator(**config)
xs, ys = generateData(p, f_s, sample_length)



## nperseg is the number of frequency ranges per sample
# f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=False)


fig, axs = plt.subplots(2)
f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=128, noverlap=64, return_onesided=False)
axs[0].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
print(Zxx.shape)
# axs[0].xlabel("time (s)")
# axs[0].ylabel("frequency (1/s)")

axs[1].plot(xs, ys)
# axs[1].xlabel("time (s)")
# axs[1].ylabel("intensity")
# plt.specgram(ys)
plt.show()
