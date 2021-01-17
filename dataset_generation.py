"""
This file is responsible for generating the dataset, which is fairly complicated.
It involves classifying 5 drones, with 2 different frequencies, and 3 different noise ratios
"""
import os
from signalgenerator import psigenerator, generateData
from configurations import *
from fourier import plotSTFT
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sample_length = 0.15 # in seconds
Xf_s = 10_000  # sample frequency for X band, in hz
Wf_s = 26_000  # sample frequency for W band, in hz

testset_size = 1_000  # per denomination
trainset_size = 5_000  # per denomination

# config = dict(scenarioWband, **djimatrice300rtk)
frequencies = [Xf_s, Wf_s]
SNRs = [10, 5, 0]
drones = [djimavicair2, djimatrice300rtk, djimavicmini, djiphantom4, parrotdisco]

for f_s in frequencies:
    for SNR in SNRs:
        for drone in drones:
            scenario = {}
            if f_s == 26_000:
                scenario = scenarioWband
            else:
                scenario = scenarioXband

            print(SNR)
            print(drone["name"])
            print(f_s)
            scenario["SNR"] = SNR
            p = psigenerator(**dict(scenario, **drone))
            # make the directory
            os.system(f"mkdir testset/{f_s}fs/{SNR}SNR/{drone['name']} -p")
            for x in range(testset_size):
                xs, ys = generateData(p, f_s, sample_length, offset=False)
                f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=16, noverlap=8, return_onesided=False)
                Z = 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0)))
                np.save(f"testset/{f_s}fs/{SNR}SNR/{drone['name']}/{x:06}.npy", Z)

            os.system(f"mkdir trainset/{f_s}fs/{SNR}SNR/{drone['name']} -p")
            for x in range(testset_size):
                xs, ys = generateData(p, f_s, sample_length, offset=False)
                f, t, Zxx = signal.stft(ys, f_s, window='hamming', nperseg=16, noverlap=8, return_onesided=False)
                Z = 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0)))
                np.save(f"trainset/{f_s}fs/{SNR}SNR/{drone['name']}/{x:06}.npy", Z)
