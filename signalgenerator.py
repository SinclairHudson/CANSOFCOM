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


giga = 1e10
c = 2.998e8  # speed of light in m/s
f_c = giga

paperconfig1 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 4,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 40,
    "lamb": 0.2,
    "theta": 0,
    "f_c": c/0.2,
}


paperconfig2 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 5,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 40,
    "lamb": 0.2,
    "theta": 0,
    "f_c": c/0.2,
}

config3= {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 4,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 120,
    "lamb": 0.2,
    "theta": 0.1,
    "f_c": c/0.2,
    "SNR": None
}

scenario
djimavicair2
djimavicmini2

dict(dic0, **dic1)

p = psigenerator(**config3)

def generateData(psi, f_s, sample_length):
    xs = []
    ys = []
    for i in range(int(f_s*sample_length)):
        x = i/f_s
        xs.append(x)
        ys.append(p(x).real)

    return xs, ys

sample_length = 1.05 # in seconds
f_s = 100_000  # sample frequency, in hz

xs, ys = generateData(p, f_s, sample_length)
## nperseg is the number of frequency ranges per sample
f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=True)
print(f, t)
print(len(f), len(t))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.plot(xs, ys)
plt.show()


