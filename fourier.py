import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



## nperseg is the number of frequency ranges per sample
# f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=False)

# note:
# assert long_window_size > (f_s/f_rot)

def plotSTFT(xs, ys, f_s, c, short_window_size=16, size=10_000):

    # we want the long_window_size to be longer than the period of the propellors.
    # this is so that we don't see the micro-dopler in the long window
    long_window_size = int((f_s / c['f_rot']) * 1.3)
    assert long_window_size > (f_s/c['f_rot'])

    fig, axs = plt.subplots(3, ncols=1, figsize=(14,10))
    fig.suptitle(f"N: {c['N']}, f_s: {f_s}, L_1: {c['L_1']}, L_2: {c['L_2']}, lambda: {c['lamb']}, f_rot: {c['f_rot']}, SNR: {c['SNR']}")
    # time-domain
    axs[0].set_title("time domain signal")
    axs[0].plot(xs, ys)

    axs[1].set_title("short-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=short_window_size,
                            noverlap=short_window_size//2, return_onesided=False)
    axs[1].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))

    axs[2].set_title("long-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=long_window_size,
                            noverlap=long_window_size//2, return_onesided=False)
    axs[2].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
    axs[2].axis([0,0.1,-size, size])

    plt.xlabel("time (s)")
    plt.show()


def plotManySTFT(xs, multi_ys, f_s, names, c, short_window_size=16, size=10_000):

    # we want the long_window_size to be longer than the period of the propellors.
    # this is so that we don't see the micro-doppler in the long window

    fig, axs = plt.subplots(len(names), ncols=1)
    fig.suptitle(f"STFTs of multiple different commercial drones")
    # time-domain
    for x in range(len(names)):
        axs[x].set_title(names[x])
        f, t, Zxx = signal.stft(multi_ys[x], f_s, window='hamming', nperseg=short_window_size,
                                noverlap=short_window_size//2, return_onesided=False)
        axs[x].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
        axs[x].axes.get_xaxis().set_ticks([])

    plt.tight_layout()
    plt.xlabel("time (s)")
    plt.show()
