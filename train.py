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


class_map = ["DJI_Matrice_300_RTK", "DJI_Mavic_Air_2",
             "DJI_Mavic_Mini", "DJI_Phantom_4", "Parrot_Disco"]

c = {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "SNR": 0,
    "f_s": 26000,
}

def confuse(l, p, num_classes):
    cm = confusion_matrix(l, p)
    classesrep = cm.shape[0]
    if classesrep < num_classes:
        pad = num_classes - classesrep
        cm = np.pad(cm, ((0, pad), (0, pad)), mode='constant', constant_values=(0,0))  # zero pad

    assert cm.shape == (num_classes, num_classes)
    return cm

wandb.init(project="cansofcom", config=c)

# I have a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataloader(file_extension):
    data = np.load(file_extension)
    return data


if c["f_s"] == 26_000:
    net = RadarDroneClassifierW().to(device)
else:
    net = RadarDroneClassifierX().to(device)


trainds = ds.DatasetFolder(
    f"trainset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))

trainLoader = torch.utils.data.DataLoader(
    trainds, c["batch_size"], shuffle=True)

testds = ds.DatasetFolder(
    f"testset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))
testLoader = torch.utils.data.DataLoader(
    testds, c["batch_size"], shuffle=True)

optim = torch.optim.AdamW(net.parameters(), lr=c["learning_rate"])
criterion = nn.CrossEntropyLoss().to(device)

for x in range(c["epochs"]):

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
        _, predicted = torch.max(yhat.data, 1)
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

    net.train()

    for i, data in enumerate(trainLoader):

        inputs, y = data
        inputs = inputs.to(device)
        y = y.to(device)
        optim.zero_grad()

        start = time.time()
        yhat = net(inputs.float())
        middle = time.time()
        loss = criterion(yhat, y)
        loss.backward()
        end = time.time()
        optim.step()

        wandb.log({
            "train_loss": loss.item(),
            "forward_time": middle - start,
            "backward_time": end - middle,
        })
        print(
            f"epoch: {x}, loss: {loss.item():06}, forward_time: {(middle - start):06}, backward_time: {(end - middle):06}")