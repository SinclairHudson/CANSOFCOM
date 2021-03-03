import torch.nn as nn
import time
import wandb
import torch
import torchvision
import torchvision.datasets as ds
import numpy as np
from classifier import RadarDroneClassifier, SanityNet
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from helpers import confuse
from OTFDataset import OTFDataset
from drone_constants import class_map


c = {
    "epochs": 75,
    "learning_rate": 0.001,
    "batch_size": 128,
    "SNR": 10,
    "f_s": 26000,
}


wandb.init(project="cansofcom", config=c)

# I have a GPU BONUS!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataloader(file_extension):
    data = np.load(file_extension)
    return data


net = RadarDroneClassifier().to(device)
# net = SanityNet().to(device)


trainds = ds.DatasetFolder(
    f"trainset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))


trainLoader = torch.utils.data.DataLoader(
    trainds, c["batch_size"], shuffle=True, num_workers=2)

testds = ds.DatasetFolder(
    f"testset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))
testLoader = torch.utils.data.DataLoader(
    testds, c["batch_size"], shuffle=True, num_workers=2)

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

        if i % 500 == 0:
            wandb.log({
                "train_loss": loss.item(),
                "forward_time": middle - start,
                "backward_time": end - middle,
            })
            print(
                f"epoch: {x}, loss: {loss.item():06}, forward_time: {(middle - start):.6f}, backward_time: {(end - middle):.6f}")

net.eval()
torch.save(net.state_dict(), f"models/e{c['epochs']}SNR{c['SNR']}f_s{c['f_s']}.pt")
