import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import os
import time
import logging
from utils.util_log import MyLogger
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# import matplotlib.image as image
from models import resnet, vgg, lenet

########
##logger
########
logger_name = 'aerial_baseline'
log_writer = MyLogger(logger_name, set_level=logging.DEBUG)
tb_writer = SummaryWriter('log/'+logger_name)

#######
##path
#######
base_path = '/home/ncl/data/aerial_data/datasets/'
cifar_path = '/home/ncl/data/'
dataset = ['AID','NWPU-RESISC45', 'PatternNet','UCMerced_LandUse']

#########
##dataset
#########
img_size=32 #224
transform_base = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(), ])

transform_cifar = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(), ])


train_dataset = []
test_dataset= []
batch_size = 100
custom_dataset = True
if custom_dataset:
    for dataset_ in dataset:
        dataset_path = base_path + dataset_
        train_d = torchvision.datasets.ImageFolder(root=dataset_path+'/train', transform=transform_base)
        val_d = torchvision.datasets.ImageFolder(root=dataset_path+'/val', transform=transform_base)
        train_dataset.append(train_d)
        test_dataset.append(val_d)
        log_writer.info(f'@ Training : {dataset_} has {len(train_d)} images and each has {train_d.classes}')
        log_writer.info(f'@ Val      : {dataset_} has {len(val_d)} images and each has {val_d.classes}')

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    num_cls = 12
else:
    train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform_cifar)
    test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform_cifar)
    num_cls = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# print(len(train_dataset))
# print(train_dataset[0][0].shape)

########
##random
########
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

########
##args
########
num_epochs = 50
lr_sgd = 0.01
lr_adam = 0.001
momentum_sgd = 0.9
# model_name = 'resnet18'
model_name = 'lenet'
opt_name='adam'

#########
##model
#########
if model_name is 'resnet18':
    model = resnet.resnet18(num_classes=num_cls,pretrained=False)
elif model_name is 'lenet':
    model = lenet.LeNet(num_classes=num_cls)
else:
    raise NotImplementedError
model.to(device)
loss_fn = nn.CrossEntropyLoss()
if opt_name is 'sgd':
    opt = torch.optim.SGD(model.parameters(),lr=lr_sgd, momentum=momentum_sgd)
elif opt_name is 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)

log_writer.info(f'img_size@{img_size} :: batch_size@{batch_size}')
log_writer.info(f'model@{model_name} :: loss_fn@crossentropy :: optimizer@{opt_name} :: lr@{lr_adam}')

for epoch in range(num_epochs):
    running_loss = []
    running_acc = []
    model.train()
    for input_, target_ in train_loader:
        input_ , target_ = input_.to(device), target_.to(device)
        out_ = model(input_)
        loss = loss_fn(out_, target_)

        opt.zero_grad()
        loss.backward()
        opt.step()
        pred = torch.argmax(out_, dim=1)
        acc = (pred==target_).float().sum()
        running_loss.append(loss.detach())
        running_acc.append(acc)
    total_acc = np.sum(running_acc)/ len(running_acc)
    total_loss = np.sum(running_loss)/len(running_loss)
    log_writer.info(f'TRAIINING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')

    with torch.no_grad():
        model.eval()
        test_running_loss = []
        test_running_acc = []
        for input_, target_ in test_loader:
            input_, target_ = input_.to(device), target_.to(device)
            out_ = model(input_)
            loss = loss_fn(out_, target_)
            pred = torch.argmax(out_, dim=1)
            acc = (pred == target_).float().sum()
            test_running_loss.append(loss.detach())
            test_running_acc.append(acc)
        total_acc = np.sum(running_acc) / len(running_acc)
        total_loss = np.sum(running_loss) / len(running_loss)
        log_writer.info(f'TESTING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')







# class CustomDataset(Dataset):
#     def __init__(self):

