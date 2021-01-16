import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



## nperseg is the number of frequency ranges per sample
# f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=False)


def plotSTFT(xs, ys, f_s, long_window_size=512, short_window_size=16):
    fig, axs = plt.subplots(3)
    # time-domain
    axs[0].set_title("time domain signal")
    axs[0].plot(xs, ys)

    axs[1].set_title("short-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=short_window_size,
                            noverlap=short_window_size//2, return_onesided=False)
    axs[1].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
    # axs[1].axis([0,0.1,-40000, 40000])

    axs[2].set_title("long-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=long_window_size,
                            noverlap=long_window_size//2, return_onesided=False)
    axs[2].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
    # axs[2].axis([0,0.1,-40000, 40000])

    plt.xlabel("time (s)")
    plt.show()
