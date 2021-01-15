import matplotlib
import scipy

from signalgenerator import psigenerator, generateData
from configurations import *

sample_length = 1.05 # in seconds
f_s = 100_000  # sample frequency, in hz


xs, ys = generateData(p, f_s, sample_length)
p = psigenerator(**config3)

## nperseg is the number of frequency ranges per sample
f, t, Zxx = signal.stft(ys, f_s, nperseg=1024, return_onesided=True)
print(f, t)
print(len(f), len(t))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.plot(xs, ys)
plt.show()
