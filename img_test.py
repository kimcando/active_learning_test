import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import os, time , logging, yaml, sys, random

from torch.utils.tensorboard import SummaryWriter
from data_handler.get_data import get_dataset, get_handler

from utils.util_log import MyLogger, MyWorkBook
from utils.util_args import arg_parser, test_arg_parser
from utils.util_transform import base_transform
from select_sampler import get_strategy
# import matplotlib.pyplot as plt
# import matplotlib.image as image
from models import base_lenet, resnet, vgg

########
##load
########
args = test_arg_parser()
config = yaml.load(open(args.config), Loader=yaml.FullLoader)
wb = MyWorkBook(args, config)

########
##random
########
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.random_seed)
random.seed(args.random_seed)
# device_id= 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########
##logger
########
logger_name = f'test_{args.data_name}_{args.model_name}_{args.opt_name}'
log_writer = MyLogger(args,logger_name, set_level=logging.DEBUG)
tb_writer = SummaryWriter(args.tb_path+logger_name)

img_size=32 #224
transform_base = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(), ])
cifar_path = '/home/ncl/ADD/data/cifar10'
train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform_base)
test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform_base)
num_cls = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=1)

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

#########
##model
#########
if args.model_name is 'resnet18':
    model = resnet.resnet18(num_classes=num_cls,pretrained=False)
elif args.model_name is 'lenet':
    model = base_lenet.LeNet(num_classes=num_cls)
else:
    raise NotImplementedError

model.to(device)
loss_fn = nn.CrossEntropyLoss()
if args.opt_name is 'SGD':
    opt = torch.optim.SGD(model.parameters(),lr=lr_sgd, momentum=momentum_sgd)
elif args.opt_name is 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
else:
    raise NotImplementedError

log_writer.info(f'img_size@{img_size} :: batch_size@{args.tr_batch_size}')
log_writer.info(f'model@{args.model_name} :: loss_fn@crossentropy :: optimizer@{args.opt_name} :: lr@{lr_adam}')
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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

    img_grid = torchvision.utils.make_grid(input_)
    tb_writer.add_image('img', img_grid, epoch)
    tb_writer.add_text('label', ' '.join([str(i) for i in list(target_.detach().cpu().numpy())]), epoch)
    tb_writer.add_text('label_name', ' '.join([classes[i] for i in list(target_.detach().cpu().numpy())]), epoch)
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
        total_acc = np.sum(test_running_acc) / len(test_running_acc)
        total_loss = np.sum(test_running_loss) / len(test_running_loss)
        log_writer.info(f'TESTING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')