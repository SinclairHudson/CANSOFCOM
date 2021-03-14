from signalgenerator import psigenerator, generateData
from drone_constants import drones, class_map
from configurations import *

from fourier import plotSTFT, plotManySTFT

f_s = 26_000
sample_length = 0.15
multi_ys = []

for drone in drones:
    config = dict(scenarioWband, **drone)
    psi = psigenerator(**config)

    xs, r_ys, i_ys = generateData(psi, f_s, sample_length)
    multi_ys.append(r_ys)

plotManySTFT(xs, multi_ys, f_s, class_map, config)
# plotSTFT(xs, i_ys, f_s, config)
