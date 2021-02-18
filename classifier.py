import torch.nn as nn
import torch


class RadarDroneClassifier(nn.Module):
    def __init__(self):
        super(RadarDroneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 5, (1, 9))
        self.IN1 = nn.InstanceNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, (5, 5), padding=2)
        self.IN2 = nn.InstanceNorm2d(10)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(10, 6, (3, 3), padding=1)
        self.IN3 = nn.InstanceNorm2d(6)
        self.conv4 = nn.Conv2d(6, 6, (3, 3), padding=1)

        self.IN4 = nn.InstanceNorm2d(6)

        self.conv5 = nn.Conv2d(16, 10, (3, 3), padding=1)
        self.IN5 = nn.InstanceNorm2d(10)
        self.conv6 = nn.Conv2d(10, 5, (3, 3), padding=1)

        # self.IN6 = nn.InstanceNorm2d(8)
        # maxpool here again

        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # input shape is Bx2x?x?
        x = self.relu(self.IN1(self.conv1(x)))
        x = self.drop(x)
        x = self.maxpool(x)
        x1 = self.relu(self.IN2(self.conv2(x)))
        x = self.relu(self.IN3(self.conv3(x1)))
        x = self.drop(x)
        x = self.relu(self.IN4(self.conv4(x)))
        # the skip connection
        x = self.relu(self.IN5(self.conv5(torch.cat((x1, x), dim=1))))
        x = self.drop(self.maxpool(x))
        x = self.conv6(x)
        x = torch.mean(x, dim=(2,3))  # reduce over the spatial dimensions
        # output shape is Bx5
        return x


class SanityNet(nn.Module):
    """
    This is a super simple neural network, that's easy to run.
    It's designed to be a sort of testing network. It's not
    expected to do well.
    """
    def __init__(self):
        super(SanityNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 5, (1, 9))
        self.IN1 = nn.InstanceNorm2d(5)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # input shape is Bx2x?x?
        x = self.relu(self.IN1(self.conv1(x)))
        x = torch.mean(x, dim=(2,3))  # reduce over the spatial dimensions

        # output shape is Bx5
        return x
