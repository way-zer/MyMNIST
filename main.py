import numpy as np
import torch
from torchvision.utils import make_grid

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib as plt

from oracle import MNIST, OracleMNIST

"""
模型代码参考: https://github.com/Tanuj-tj/Flowers_Recognition_Project/blob/main/FlowerReco_ResNet9.ipynb
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

dataSource = lambda train: MNIST(root="datasets/mnist", train=train, download=True)
# dataSource = lambda train: OracleMNIST(root="datasets/mnist", train=train, download=True)
batch_size = 64

trainData = dataSource(True)
testData = dataSource(False)

trainDL = DataLoader(trainData, batch_size, shuffle=True, num_workers=4, pin_memory=True)
testDL = DataLoader(testData, batch_size, num_workers=4, pin_memory=True)

for images, labels in trainDL:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    break

from torch import nn


class DropConnect(nn.Module):
    def __init__(self, f: nn.Module):
        super().__init__()
        self.w = nn.Parameter(Tensor([1]))
        self.f = f

    def forward(self, x: Tensor):
        return self.f(x) + self.w * x
