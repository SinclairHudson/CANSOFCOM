import torch.nn
import wandb
import torchvision
import torchvision.datasets as ds
import numpy as np

def Xdataloader(file_extension):
    data = np.load(file_extension)
    data = torch.Tensor(data)
    data = torchvision.transforms.Normalize((-24,), (50,))(data)
    return data

def Wdataloader(file_extension):

Wband0SNR = ds.DatasetFolder("")
