from signalgenerator import psigenerator, generateData
from drone_constants import drones, class_map
from configurations import *

from fourier import plotSTFT, plotManySTFT, plotcomparisonSTFT

f_s = 26_000
sample_length = 0.15
multi_ys = []

# SNR = [None, 20, 10, 5, 0]

# for S in SNR:
    # config = dict(scenarioWband, **djimavicair2)
    # config["SNR"] = S
    # psi = psigenerator(**config)

    # xs, r_ys, i_ys = generateData(psi, f_s, sample_length)
    # multi_ys.append(r_ys)

# plotManySTFT(xs, multi_ys, f_s, ["No Noise", "20 SNR", "10 SNR", "5 SNR", "0 SNR"], config)

config = paperconfig2

config["SNR"] = None
psi = psigenerator(**config, deterministic=True)
xs, r_ys, i_ys = generateData(psi, f_s, sample_length)
plotSTFT(xs, r_ys, f_s, config, size=10_000)
