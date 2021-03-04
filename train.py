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
from drone_constants import class_map, c


def OTFTrain(conf):
    """
    This function generates a dataset on the fly, so no need to save anything.
    However, it's slower, since the CPU has to perform a fourier transform
    at every step instead of loading data from a file.
    :param conf: a dictionary containing parameters for the data and hyperparameters
    for training.
    """
    wandb.init(project="cansofcom", config=conf)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = RadarDroneClassifier().to(device)

    trainds = OTFDataset(c/conf["f_c"], conf["train_set_size"], conf["SNR"],
                        conf["f_s"], conf["signal_duration"])

    trainLoader = torch.utils.data.DataLoader(
        trainds, conf["batch_size"], shuffle=True, num_workers=4)

    testds = OTFDataset(c/conf["f_c"], conf["test_set_size"], conf["SNR"],
                        conf["f_s"], conf["signal_duration"])

    testLoader = torch.utils.data.DataLoader(
        testds, conf["batch_size"], shuffle=True, num_workers=4)

    optim = torch.optim.AdamW(net.parameters(), lr=conf["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    for x in range(conf["epochs"]):
        epoch_time_start = time.time()
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
            l, p = torch.Tensor.cpu(y).numpy(
            ), torch.Tensor.cpu(predicted).numpy()

            cm = confuse(l, p, 5)

            confm = np.add(confm, cm)
            correct += (predicted == y).sum().item()
            total += conf["batch_size"]

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
        epoch_time_end = time.time()
        print(f"epoch time: {(epoch_time_end - epoch_time_start):.6f}")

    net.eval()
    torch.save(net.state_dict(),
               f"models/{str(conf)}.pt")


def train(conf):
    wandb.init(project="cansofcom", config=conf)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def dataloader(file_extension):
        data = np.load(file_extension)
        return data

    net = RadarDroneClassifier().to(device)

    trainds = ds.DatasetFolder(
        f"trainset/{conf['f_s']}fs/{conf['SNR']}SNR", dataloader, extensions=("npy",))

    trainLoader = torch.utils.data.DataLoader(
        trainds, conf["batch_size"], shuffle=True, num_workers=2)

    testds = ds.DatasetFolder(
        f"testset/{conf['f_s']}fs/{conf['SNR']}SNR", dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, conf["batch_size"], shuffle=True, num_workers=2)

    optim = torch.optim.AdamW(net.parameters(), lr=conf["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    for x in range(conf["epochs"]):

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
            l, p = torch.Tensor.cpu(y).numpy(
            ), torch.Tensor.cpu(predicted).numpy()

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
    torch.save(net.state_dict(),
               f"models/e{conf['epochs']}SNR{conf['SNR']}f_s{conf['f_s']}.pt")
