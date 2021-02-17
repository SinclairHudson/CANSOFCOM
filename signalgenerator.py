import math
import cmath
import numpy as np
from scipy import signal

j = complex(0, 1)  # sqrt negative 1


def sinc(x):
    if(x == 0):
        return 1  # this makes it continuous
    return math.sin(x) / x


def psigenerator(f_c, lamb, N, L_1, L_2, f_rot, SNR=None, **kwargs):
    """
    This function returns a psi function, which represents the RADAR signal off
    of a drone.
    :param SNR: Signal-to-noise ratio, given in dB. If None, don't add noise
    """

    A_r = np.random.chisquare(4)  # A_r is a random value from X^2 with 4 dof
    R = np.random.uniform(low=1000, high=5000)
    theta = np.random.uniform(low=0, high=np.pi/2)
    V_rad = 0

    def psi(t):
        prefactor = A_r * \
            cmath.exp(j*(2*math.pi*f_c*t - (4*math.pi/lamb)*(R+V_rad*t)))
        # prefactor = 1  # as suggested in https://discord.com/channels/760673053695934495/794667756132761610/799474019308404786
        accum = complex(0, 0)
        for n in range(N):
            exponential = cmath.exp(-j*(4*math.pi/lamb)*(((L_1 + L_2)/2) *
                                                         math.cos(theta)*math.sin(2*math.pi*f_rot*t + (2*math.pi*n)/N)))
            sincterm = sinc(((4*math.pi)/lamb)*((L_2 - L_1)/2) *
                            math.cos(theta)*math.sin(2*math.pi*(f_rot*t+(n/N))))
            accum += exponential*sincterm

        return prefactor * accum

    if not SNR is None:
        # this is a re-arrangement of dB = 10\log_{10}{A_r^2/\sigma^2}
        variance = 10**(2*math.log10(A_r) - (SNR/10))

        def fuzzypsi(t):
            real_noise = np.random.normal(0, math.sqrt(variance))
            imag_noise = np.random.normal(0, math.sqrt(variance))
            return psi(t) + real_noise + imag_noise *j

        return fuzzypsi

    else:  # if we don't want noise, just return psi
        return psi


def generateData(psi, f_s, sample_length, offset=False):
    """
    Create a discrete series of measurements for a given psi function, taking
    just the real component
    :param psi: the function being sampled
    :param f_s: the sampling frequency in Hz
    :param sample_length: the length of the sample in seconds

    :returns: a tuple of time values x, and their corresponding psi(x) values.
    """
    xs = []
    real_ys = []
    imaginary_ys = []
    offset_val = 0
    if offset:
        # offset a sample anywhere from 0 to 1 seconds
        offset_val = np.random.uniform(0, 1)
    for i in range(int(f_s*sample_length)):
        x = i/f_s
        xs.append(x)
        val = psi(offset_val + x)
        real_ys.append(val.real)
        imaginary_ys.append(val.imag)

    return xs, real_ys, imaginary_ys


def generateSTFT(psi, f_s, sample_length, offset=False):

    xs, r_ys, i_ys = generateData(psi, f_s, sample_length, offset=True)

    # perform a fourier transform on both the real part and imaginary part of the 
    # signal, independently
    f, t, Zreal = signal.stft(
        r_ys, f_s, window='hamming', nperseg=16, noverlap=8, return_onesided=False)
    Xreal = 20*np.log10(np.abs(np.fft.fftshift(Zreal, axes=0)))

    f, t, Zimag = signal.stft(
        i_ys, f_s, window='hamming', nperseg=16, noverlap=8, return_onesided=False)
    Ximag = 20*np.log10(np.abs(np.fft.fftshift(Zimag, axes=0)))

    return np.stack((Xreal, Ximag))
