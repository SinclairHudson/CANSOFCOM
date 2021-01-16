import torch.nn
import time
import wandb
import torchvision
import torchvision.datasets as ds
import numpy as np
from classifier import RadarDroneClassifier

c = {
    "epochs": 5,
    "learning_rate": 0.001,
}

wandb.init(project="cansofcom", config=c)

def dataloader(file_extension):
    data = np.load(file_extension)
    return data

Wband0SNR = ds.DatasetFolder("dataset/26000fs/0SNR")

net = RadarDroneClassifier()

optim = torch.optim.AdamW(net.parameters(), lr=c["learning_rate"])
criterion = nn.CrossEntropyLoss()

for x in range(c["epochs"]):
    for i, data in enumerate(trainLoader):

        inputs, y = data
        optim.zero_grad()

        start = time.time()
        yhat = net(inputs)
        middle = time.time()
        loss = criterion(yhat, y)
        loss.backward()
        end = time.time()
        optim.step()

        wandb.log({
            "loss": loss.item(),
            "forward_time": middle - start,
            "backward_time": end - middle,
            "accuracy": end - middle,
        })


