import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



## nperseg is the number of frequency ranges per sample
# f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=False)

# note:
# assert long_window_size > (f_s/f_rot)

def plotSTFT(xs, ys, f_s, c, short_window_size=16):

    # we want the long_window_size to be longer than the period of the propellors.
    # this is so that we don't see the micro-dopler in the long window
    long_window_size = int((f_s / c['f_rot']) * 1.3)
    assert long_window_size > (f_s/c['f_rot'])

    fig, axs = plt.subplots(3)
    fig.suptitle(f"N: {c['N']}, f_s: {f_s}, L_1: {c['L_1']}, L_2: {c['L_2']}, lambda: {c['lamb']}, f_rot: {c['f_rot']}, SNR: {c['SNR']}")
    # time-domain
    axs[0].set_title("time domain signal")
    axs[0].plot(xs, ys)

    axs[1].set_title("short-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=short_window_size,
                            noverlap=short_window_size//2, return_onesided=False)
    print(Zxx.shape)
    axs[1].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
    # axs[1].axis([0,0.1,-40000, 40000])

    axs[2].set_title("long-window STFT")
    f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=long_window_size,
                            noverlap=long_window_size//2, return_onesided=False)
    axs[2].pcolormesh(t, np.fft.fftshift(f), 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0))))
    # axs[2].axis([0,0.1,-40000, 40000])

    plt.xlabel("time (s)")
    plt.show()
