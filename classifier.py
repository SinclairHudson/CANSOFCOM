import torch.nn as nn
import torch


class RadarDroneClassifier(nn.Module):
    def __init__(self):
        super(RadarDroneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, (1, 7))  # conv1D, effectively
        self.conv2 = nn.Conv2d(5, 10, (5, 5), padding=2)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(10, 6, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(6, 6, (3, 3), padding=1)

        self.conv5 = nn.Conv2d(16, 4, (3, 3),padding=1)

        # maxpool here again

        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()


    def forward(self, x):
        raise NotImplementedError()


class RadarDroneClassifierX(RadarDroneClassifier):
    def __init__(self):
        super(RadarDroneClassifierX, self).__init__()
        self.lastdim = 756

        self.LN = nn.LayerNorm((10, 8, 94))
        self.linear = nn.Linear(self.lastdim, 5)


    def forward(self, x):
        # input shape is 16x189
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        x = self.maxpool(x)
        x1 = self.relu(self.LN(self.conv2(x)))
        x = self.relu(self.conv3(x1))
        x = self.drop(x)
        x = self.relu(self.conv4(x))
        # the skip connection
        x = self.relu(self.conv5(torch.cat((x1, x), dim=1)))
        x = self.maxpool(x)
        x = x.reshape((-1, self.lastdim))
        x = self.linear(x)
        return x


class RadarDroneClassifierW(RadarDroneClassifier):
    def __init__(self):
        super(RadarDroneClassifierW, self).__init__()
        self.lastdim = 1920

        self.linear = nn.Linear(self.lastdim, 5)

        self.LN = nn.LayerNorm((10, 8, 241))

    def forward(self, x):
        # input shape is 16x189
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        x = self.maxpool(x)
        x1 = self.relu(self.LN(self.conv2(x)))
        x = self.relu(self.conv3(x1))
        x = self.drop(x)
        x = self.relu(self.conv4(x))
        # the skip connection
        x = self.relu(self.conv5(torch.cat((x1, x), dim=1)))
        x = self.maxpool(x)
        x = x.reshape((-1, self.lastdim))
        x = self.linear(x)
        return x
