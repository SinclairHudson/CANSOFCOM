from signalgenerator import psigenerator, generateData
from configurations import *

from fourier import plotSTFT

config = paperconfig1
config = dict(scenarioWband, **djiphantom4)
psi = psigenerator(**config)

f_s = 10_000
sample_length = 0.15
xs, r_ys, i_ys = generateData(psi, f_s, sample_length)

plotSTFT(xs, r_ys, f_s, config)
plotSTFT(xs, i_ys, f_s, config)
