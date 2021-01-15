import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from signalgenerator import psigenerator, generateData
from configurations import *

sample_length = 0.05 # in seconds
f_s = 100_000  # sample frequency, in hz

config = dict(scenario, **djimavicair2)
# config = paperconfig2
p = psigenerator(**config)
xs, ys = generateData(p, f_s, sample_length)



## nperseg is the number of frequency ranges per sample
f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=True)
print(f, t)
print(len(f), len(t))
print(Zxx.shape)
print(np.abs(Zxx).shape)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.plot(xs, ys)
plt.show()
