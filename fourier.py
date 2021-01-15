import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from signalgenerator import psigenerator, generateData
from configurations import *

sample_length = 0.10 # in seconds
Wf_s = 26_000  # sample frequency for W band, in hz
Xf_s = 10_000  # sample frequency for X band, in hz

# config = dict(scenarioWband, **djimatrice300rtk)
config = paperconfig1
p = psigenerator(**config)
xs, ys = generateData(p, Wf_s, sample_length)



## nperseg is the number of frequency ranges per sample
# f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=False)


fig, axs = plt.subplots(3)
# time-domain
axs[0].set_title("time domain signal")
# axs[0].ylabel("Voltage (V)")
axs[0].plot(xs, ys)

# short-window STFT
axs[1].set_title("short-window STFT")
f, t, Zxx = signal.stft(ys, Wf_s, window='hamming', nperseg=16, noverlap=8, return_onesided=False)
axs[1].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
axs[1].axis([0,0.1,-10000, 10000])

# long-window STFT
axs[2].set_title("long-window STFT")
f, t, Zxx = signal.stft(ys, Wf_s, window='hamming', nperseg=256, noverlap=64, return_onesided=False)
axs[2].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
axs[2].axis([0,0.1,-10000, 10000])

plt.xlabel("time (s)")
plt.show()
