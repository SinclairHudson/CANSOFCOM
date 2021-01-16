import math
import cmath
import numpy as np

j = complex(0, 1) # sqrt negative 1

def sinc(x):
    if(x == 0):
        return 1  # this makes it continuous
    return math.sin(x) / x

def psigenerator(A_r, f_c, lamb, R, V_rad, N, L_1, L_2, f_rot, theta, SNR=None, **kwargs):
    """
    This function returns a psi function, which represents the RADAR signal off
    of a drone.
    :param SNR: Signal-to-noise ratio, given in dB. If None, don't add noise
    """

    def psi(t):
        # data will be more realistic if this prefactor is used, instead of 1
        # prefactor = A_r * cmath.exp(j*(2*math.pi*f_c*t - (4*math.pi/lamb)*(R+V_rad*t)))
        prefactor = 1  # as suggested in https://discord.com/channels/760673053695934495/794667756132761610/799474019308404786
        accum = complex(0, 0)
        for n in range(N):
            exponential = cmath.exp(-j*(4*math.pi/lamb)*(((L_1 + L_2)/2)*math.cos(theta)*math.sin(2*math.pi*f_rot*t + (2*math.pi*n)/N)))
            sincterm = sinc(((4*math.pi)/lamb)*((L_2 - L_1)/2)*math.cos(theta)*math.sin(2*math.pi*(f_rot*t+(n/N))))
            accum += exponential*sincterm

        return prefactor * accum

    if not SNR is None:
        # this is a re-arrangement of dB = 10\log_{10}{A_r^2/\sigma^2}
        variance = 10**(2*math.log10(A_r) - (SNR/10))
        def fuzzypsi(t):
            noise = np.random.normal(0, math.sqrt(variance))
            return psi(t) + noise

        return fuzzypsi

    else:  # if we don't want noise, just return psi
        return psi


def generateData(psi, f_s, sample_length):
    """
    Create a discrete series of measurements for a given psi function, taking
    just the real component
    :param psi: the function being sampled
    :param f_s: the sampling frequency in Hz
    :param sample_length: the length of the sample in seconds

    :returns: a tuple of time values x, and their corresponding psi(x) values.
    """
    xs = []
    ys = []
    for i in range(int(f_s*sample_length)):
        x = i/f_s
        xs.append(x)
        ys.append(psi(x).real)

    return xs, ys

