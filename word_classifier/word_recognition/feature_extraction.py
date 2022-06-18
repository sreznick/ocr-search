import torch
from torch import nn


class OriginalCNN(nn.Module):
    """
    CNN architecture from the CRNN paper by Shi, et. al. (arXiv:1507.05717v1).
    """
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.stack(x)


class Features2Seq(nn.Module):
    """
    Transforms CNN output to a sequence.
    """
    def __init__(self, in_channels: int, img_height: int, out_features: int):
        super().__init__()
        in_features = in_channels * img_height
        self.in_channels = in_channels
        self.img_height = img_height
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        assert channel == self.in_channels and height == self.img_height
        x = x.view(batch, channel * height, width).permute(2, 0, 1)
        return self.fc(x)
