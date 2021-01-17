import torch.nn as nn
import time
import wandb
import torch
import torchvision
import torchvision.datasets as ds
import numpy as np
from classifier import RadarDroneClassifierW, RadarDroneClassifierX
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn

c = {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "SNR": 5,
    "f_s": 26000,
}

def dataloader(file_extension):
    data = np.load(file_extension)
    return data

def TestClassifier(f_s, SNR):
    c = {
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 64,
        "SNR": SNR,
        "f_s": f_s,
    }

    softmax = nn.Softmax(dim=1)  # class dimension, not batch
    testds = ds.DatasetFolder(
        f"testset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, c["batch_size"], shuffle=True)

    class_map = ["DJI_Matrice_300_RTK", "DJI_Mavic_Air_2",
                 "DJI_Mavic_Mini", "DJI_Phantom_4", "Parrot_Disco"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if c["f_s"] == 26_000:
        net = RadarDroneClassifierW().to(device)
    else:
        net = RadarDroneClassifierX().to(device)


    net.load_state_dict(torch.load(f"e{c['epochs']}SNR{c['SNR']}f_s{c['f_s']}.pt"))
    net.eval()

    confm = np.zeros((5, 5), dtype=int)
    correct = 0
    total = 0
    testloss = 0
    for i, data in enumerate(testLoader):
        inputs, y = data
        inputs = inputs.to(device)
        y = y.to(device)

        yhat = net(inputs.float())
        loss = criterion(yhat, y)
        testloss += loss.item()

        # search for max along dimension 1, also note that index.
        softout = softmax(yhat)
        confidence, predicted = torch.max(softout.data, 1)

        l, p = torch.Tensor.cpu(y).numpy(), torch.Tensor.cpu(predicted).numpy()

        cm = confuse(l, p, 5)

        confm = np.add(confm, cm)
        correct += (predicted == y).sum().item()
        total += c["batch_size"]

    plt.close()
    fig, ax = plot_confusion_matrix(conf_mat=confm,
                                    show_normed=True,
                                    colorbar=True)
    wandb.log({
        "test_loss": testloss,
        "test_accuracy": correct/total,
        "test_confusion_matrix": plt

    })
