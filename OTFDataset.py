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

    def __init__(self, lamb, length, SNR, f_s, sample_length, in_memory=True):
        """
        in_memory is a boolean, specifying if the dataset should be pre-generated
        and just stored in memory for the duration of the object's life.
        This greatly speeds up training, but the dataset is technically smaller.
        However, with large dataset, the variance should be the same.
        if in_memory is false, then the dataset actually changes every epoch.
        Again, this shouldn't have a large impact on large datasets.
        If your dataset is too large, of if f_s or sample_length are very big,
        then it's possible that initializing this object with in_memory
        will take all the RAM in your system, crashing it. In this case,
        you're forced to either reduce the size or set in_memory=False.
        """
        super(OTFDataset, self).__init__()
        self.lamb = lamb
        self.length = length
        self.SNR = SNR
        self.f_s = f_s
        self.sample_length = sample_length

        self.in_memory = in_memory

        if self.in_memory:
            print(f"generating an in-memory on the fly dataset of size {self.length}.")
            self.drone_classes = []
            self.STFTs = []
            for x in range(self.length):
                if x % 500 == 0:
                    print(f"{x}/{self.length}")

                stft, dc = self.generate_item(x)
                self.drone_classes.append(dc)
                self.STFTs.append(stft)


    def __getitem__(self, index):

        if self.in_memory:
            return self.STFTs[index], self.drone_classes[index]
        else:
            return generate_item(index)

    def generate_item(self, index):

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
