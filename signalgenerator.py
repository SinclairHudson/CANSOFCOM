import math
import cmath
from scipy import signal
import numpy as np

import matplotlib.pyplot as plt

j = complex(0, 1) # sqrt negative 1

def sinc(x):
    if(x == 0):
        return 0
    return math.sin(x) / x

def psigenerator(A_r, f_c, lamb, R, V_rad, N, L_1, L_2, f_rot, theta, SNR=None):
    """
    :param SNR: Signal-to-noise ratio, given in dB. If None, don't add noise
    """

    def psi(t):
        # prefactor = A_r * cmath.exp(j*(2*math.pi*f_c*t - (4*math.pi/lamb)*(R+V_rad*t)))
        prefactor = 1  # as suggested in https://discord.com/channels/760673053695934495/794667756132761610/799474019308404786
        accum = complex(0, 0)
        for n in range(N):
            exponential = cmath.exp(-j*(4*math.pi/lamb)*(((L_1 + L_2)/2)*math.cos(theta)*math.sin(2*math.pi*f_rot*t + (2*math.pi*n)/N)))
            sincterm = sinc(((4*math.pi)/lamb)*((L_2 - L_1)/2)*math.cos(theta)*math.sin(2*math.pi*(f_rot*t+(n/N))))
            accum += exponential*sincterm

        return prefactor * accum

    if not SNR is None:
        variance = 10**(2*math.log10(A_r) - (SNR/10))
        def fuzzypsi(t):
            noise = np.random.normal(0, math.sqrt(variance))
            return psi(t) + noise

        return fuzzypsi

    else:  # if we don't want noise, just return psi
        return psi


def generateData(psi, f_s, sample_length):
    xs = []
    ys = []
    for i in range(int(f_s*sample_length)):
        x = i/f_s
        xs.append(x)
        ys.append(p(x).real)

    return xs, ys




