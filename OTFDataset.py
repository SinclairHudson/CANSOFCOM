import torch
from signalgenerator import psigenerator, generateData, generateSTFT
from configurations import djimavicair2, djimatrice300rtk, djimavicmini, djiphantom4, parrotdisco
import random
import numpy as np
from drone_constants import c, drones


class OTFDataset(torch.utils.data.Dataset):
    """
    This class creates training and testing examples on the fly.
    This means I don't have to save 100 different versions of what's effectively the
    same dataset.
    """

    def __init__(self, lamb, length, SNR, f_s, sample_length):
        super(OTFDataset, self).__init__()
        self.lamb = lamb
        self.length = length
        self.SNR = SNR
        self.f_s = f_s
        self.sample_length = sample_length

    def __getitem__(self, index):

        scenario = {
            "SNR": self.SNR,
            "theta": np.random.uniform(low=0, high=np.pi/2),
            "R": np.random.uniform(low=1000, high=5000),
            "A_r": np.random.chisquare(4),
            "f_c": c/self.lamb,
            "lamb": self.lamb,
        }

        # class of the drone is random
        drone_class = random.randint(0, len(drones)-1)

        p = psigenerator(**dict(scenario, **drones[drone_class]))

        STFT = generateSTFT(p, self.f_s, self.sample_length, offset=True)

        return STFT, drone_class

    def __len__(self):
        return self.length
