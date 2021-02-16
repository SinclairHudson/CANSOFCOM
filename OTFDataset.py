import torch
from signalgenerator import psigenerator, generateData

class OTFDataset(torch.utils.data.Dataset):
    """
    This class creates training and testing examples on the fly.
    """
    def __init__(self, A_r_dist, R_dist, theta_dist, V_rad_dist, SNR, f_s, sample_length_dist):
        super(OTFDataset, self).__init__()

    def __getitem__(self, index):
        psigenerator(A_r, f_c, lamb, R, V_rad, N, L_1, L_2, f_rot, theta, SNR=None, **kwargs):



