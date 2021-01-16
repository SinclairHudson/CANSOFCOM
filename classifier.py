import torch.nn as nn


class RadarDroneClassifier(nn.Module):
    def __init__(self):
        super(RadarDroneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1,5, (1, 7))  # conv1D, effectively
        self.conv2 = nn.Conv2d(5,10, (5, 5))
        self.maxpool = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(10, 6, (3, 3))
        self.conv4 = nn.Conv2d(10, 6, (3, 3))

        self.conv5 = nn.Conv2d(16, 6, (3, 3))

        self.linear = nn.Linear(something, 5)

        self.drop = nn.Dropout2D(p=0.5)
        self.relu = nn.LeakyReLU()

        self.LN = nn.LayerNorm(size[1:])

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        x1 = self.relu(self.LN(self.conv2(x)))
        x = self.relu(self.conv3(x1))
        x = self.drop(x)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(torch.cat((x1, x), dim=1)))  # the skip connection
        x = x.reshape((-1, ))
        x = self.linear(x)
        return x
