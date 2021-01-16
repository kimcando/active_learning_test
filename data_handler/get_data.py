import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import (DataLoader,
                              Dataset,
                              ConcatDataset,
                              RandomSampler,
                              SubsetRandomSampler)
from abc import ABC, abstractmethod
from PIL import Image

def select_data(args, config):
    args_pool = {'mnist':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                     'loader_tr_args':{'batch_size': args.tr_batch_size, 'num_workers': 1},
                     'loader_te_args':{'batch_size': args.te_batch_size, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'fashionmnist':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                     'loader_tr_args':{'batch_size': args.tr_batch_size, 'num_workers': 1},
                     'loader_te_args':{'batch_size': args.te_batch_size, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'svhn':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                     'loader_tr_args':{'batch_size': args.tr_batch_size, 'num_workers': 1},
                     'loader_te_args':{'batch_size': args.te_batch_size, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'cifar10':
                    {'n_epoch': 3, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                     'loader_tr_args':{'batch_size': args.tr_batch_size, 'num_workers': 1},
                     'loader_te_args':{'batch_size': args.te_batch_size, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                     'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}
                    }

    return args_pool[args.data_name]



def get_dataset(args, config):
    if args.data_name == 'mnist':
        return get_MNIST(args)
    elif args.data_name == 'fashionmnist':
        return get_FashionMNIST(args)
    elif args.data_name == 'svhn':
        return get_SVHN(args)
    elif args.data_name == 'cifar10':
        return get_CIFAR10(args)

def get_MNIST(args):
    raw_tr = datasets.MNIST(args.data_path + '/mnist', train=True, download=True)
    raw_te = datasets.MNIST(args.data_path + '/mnist', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(args):
    raw_tr = datasets.FashionMNIST(args.data_path + '/fashionmnist', train=True, download=True)
    raw_te = datasets.FashionMNIST(args.data_path + '/fashionmnist', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(args):
    data_tr = datasets.SVHN(args.data_path + 'cifar', split='train', download=True)
    data_te = datasets.SVHN(args.data_path +'/svhn', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(args):
    data_tr = datasets.CIFAR10(args.data_path + '/cifar10', train=True, download=True)
    data_te = datasets.CIFAR10(args.data_path + '/cifar10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_handler(args):
    if args.data_name == 'mnist':
        return DataHandler3
    elif args.data_name == 'fashionmnist':
        return DataHandler1
    elif args.data_name == 'svhn':
        return DataHandler2
    elif args.data_name == 'cifar10':
        return DataHandler3
    else:
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)



#######
##custom path
#######
# base_path = '/home/ncl/data/aerial_data/datasets/'
# dataset = ['AID','NWPU-RESISC45', 'PatternNet','UCMerced_LandUse']

# if args.data_name:
#     for dataset_ in dataset:
#         dataset_path = base_path + dataset_
#         train_d = torchvision.datasets.ImageFolder(root=dataset_path+'/train', transform=transform_base)
#         val_d = torchvision.datasets.ImageFolder(root=dataset_path+'/val', transform=transform_base)
#         train_dataset.append(train_d)
#         test_dataset.append(val_d)
#         log_writer.info(f'@ Training : {dataset_} has {len(train_d)} images and each has {train_d.classes}')
#         log_writer.info(f'@ Val      : {dataset_} has {len(val_d)} images and each has {val_d.classes}')
#
#     train_dataset = torch.utils.data.ConcatDataset(train_dataset)
#     test_dataset = torch.utils.data.ConcatDataset(test_dataset)
#     num_cls = 12
# else:
#     train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform_cifar)
#     test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform_cifar)
#     num_cls = 10