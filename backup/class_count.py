import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import os
import time

transform_cifar = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(), ])
cifar_path = '/home/ncl/data/'
train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform_cifar)
test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform_cifar)


unq, unq_cnt = np.unique(train_dataset.targets, return_counts=True)
cls_cnt = {i: n for i, n in zip(unq, unq_cnt)}
# https://discuss.pytorch.org/t/filtering-a-class-form-subset-of-fashonmnist/65162/6
train_dataset2 = Subset(train_dataset, indices=[range(30)])
# unq, unq_cnt = np.unique(train_dataset.targets, return_counts=True)
# cls_cnt = {i: n for i, n in zip(unq, unq_cnt)}
a = train_dataset2.indices
b = np.array(train_dataset.targets)[tuple(a)]
print(b)


