import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class FCNAttacker(nn.Module):
    """Classifier used "Efficient passive membership inference attack in federated learning".
    See https://arxiv.org/abs/2111.00430.
    Parameters (kernel size and other) obtained after discussion with the author.
    NB: the authors lost access to the code.
    """

    def __init__(self):
        # 1,316,866 parameters.
        super(FCNAttacker, self).__init__()
        self.block1 = ConvBlock(1, 128)
        self.block2 = ConvBlock(128, 256)
        self.block3 = ConvBlock(256, 128)
        self.bn_final = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn_final(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = FCNAttacker()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_params])
    print(f"{nb_params:,d} trainable parameters.")
