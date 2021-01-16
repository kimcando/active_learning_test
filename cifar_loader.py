# https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/data/ImbalanceCIFAR.py

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import random
from abc import ABC, abstractmethod
import argparse
import random

class ImbalanceBaseDataset(Dataset, ABC):
    """
    this is constructing for training dataset
    """
    def __init__(self, root, args, imbalance_ratio, log_writer, train, download, transform=None, target_transform=None):
        super(ImbalanceBaseDataset, self).__init__()
        self.args = args
        self.log_writer = log_writer
        self.imbalance_ratio = imbalance_ratio
        self.train = train
        self.shuffle = args.shuffle
        self.target_transform = target_transform
        # self.images, self.targets = self.get_dataset()
        # self.transform = transform
        # self.init_targets = self.targets

    def get_dataset(self):
        """
        return data, label -> list, np.array
        """
        if self.args.data_name =="cifar10":
            data_tr = datasets.CIFAR10(self.args.data_path + '/cifar10', train=True, download=True)
            self._init_labels_class_counts(data_tr.targets)
            return data_tr.data, np.array(data_tr.targets)
        else:
            raise NotImplementedError

    def _init_labels_class_counts(self, targets):
        self._init_cls, self._init_counts = np.unique(targets, return_counts = True)
        # print(f'{self.args.data_name}: {self._init_cls} class, each has {self._init_counts}')
        self.log_writer.info(f'initial count: {self.args.data_name}: {self._init_cls} class, each has {self._init_counts}')
    def __getitem__(self, index):
        img, label = self.images[index], self.targets[index]
        # returning
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, index

    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def get_imbalanced_data(self):
        pass

    def update_labels(self, new_labels):
        self.targets = np.array(new_labels)

class ImbalanceCIFAR10(ImbalanceBaseDataset):
    """
    this is constructing for training dataset.
    initiation is happend in the parent class
    """

    def __init__(self, root, args, imbalance_ratio, log_writer,train, download, transform=None):
        super(ImbalanceCIFAR10, self).__init__(root, args, imbalance_ratio, log_writer,train, download, transform=None)
        self.args = args
        self.log_writer = log_writer
        # TODO
        self.imbalance_ratio = imbalance_ratio # list
        self.train = train
        self.shuffle = args.shuffle
        self.images, self.targets = self.get_dataset()
        self.transform = transform

        self.get_imbalanced_data()
        self.labels_class_counts(self.targets, self.new_idxs)

        # if not np.all(np.equal(self.imbalance_ratio,np.hstack([1.0]*10))):
        #     self.get_imbalanced_data()
        #     self.labels_class_counts(self.targets)

    def labels_class_counts(self, targets, new_idxs):
        cls,counts  = np.unique(targets[new_idxs], return_counts = True)
        # print(f'{self.args.data_name}: {cls} class, each has {counts}')
        self.log_writer.info(f'{self.args.data_name}: {cls} class, each has {counts}')

    def get_imbalanced_data(self):
        """
        no randomness yet
        """
        trgs, cls_counts = np.unique(self.targets, return_counts = True)
        cls_indices ={i: np.where(self.targets == i)[0] for i in trgs}
        cls_idx_counts = {k:len(v) for k, v in cls_indices.items()}
        self.imbal_cls_counts = {cls:int(count*prop) for (cls, count), prop in zip(cls_idx_counts.items(), self.imbalance_ratio)}
        new_idxs = []
        for c in trgs:
            new_idxs.append(cls_indices[c][:self.imbal_cls_counts[c]])
        self.new_idxs = np.hstack(new_idxs)
        # update --> this doesn't hold
        # self.targets = self.targets[new_idxs]
        # random.shuffle(self.targets)

    def __getitem__(self, index):
        img, label = self.images[self.new_idxs[index]], self.targets[self.new_idxs[index]]
            # returning
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, index
        # if not np.all(np.equal(self.imbalance_ratio, np.hstack([1.0] * 10))):
        #     img, label = self.images[self.new_idxs[index]], self.targets[self.new_idxs[index]]
        # else:
        #     img, label = self.images[index], self.targets[index]
        #     # returning
        # img = Image.fromarray(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        # return img, label, index

    def __len__(self):
        return len(self.new_idxs)
        # if not np.all(np.equal(self.imbalance_ratio, np.hstack([1.0] * 10))):
        #     return len(self.new_idxs)
        # else:
        #     return len(self.targets)

    def cls_info(self):
        self.labels_class_counts(self.targets)

    def update_labels(self, new_labels):
        self.targets = np.array(new_labels)

class LabelController(object):
    def __init__(self, args, targets):
        self.total_pool = len(targets)
        self.init_pool = args.nStart
        self.add_num = args.nQuery
        self.idxs_lb = np.zeros(len(targets, dtype=bool))
        self.idxs_tmp = np.arange(len(targets))
        self.args = args
        if args.shuffle:
            np.random.shuffle(self.idxs_tmp)
        self._init_training_pool()

    def _init_training_pool(self):
        self.idxs_lb[self.idxs_tmp[:self.add_num]] = True

    def update_training_pool(self, new):
        self.idxs_lb[new] = True
        return self.idxs_lb

    @property
    def num_idx(self):
        return len(np.where(self.idxs_lb == True))

# class SplitBaseDataset(Dataset, ABC):
if __name__=="__main__":
    # data_tr = datasets.CIFAR10("/home/ncl/ADD/data" + '/cifar10', train=True, download=True)
    # imbalance_ratio = np.hstack(([0.1] *2, [1.0] * 2))
    # print(imbalance_ratio)
    # cls_counts= {1:np.arange(12), 2:np.arange(3), 3:np.arange(4), 4:np.arange(6)}
    # cls_counts_num = {k: len(v) for k, v in cls_counts.items()}
    # imbal_class_counts = [int(count*prop) for count, prop in zip(cls_counts_num.values(), imbalance_ratio)]
    # print(imbal_class_counts)
    imbalance_ratio = np.hstack(([0.1] * 5, [1.0] * 5))
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', help='acquisition algorithm', type=str, default='False')
    parser.add_argument('--data_name', help='acquisition algorithm', type=str, default='cifar10')
    parser.add_argument('--data_path', help='/home/ncl/ADD/data', type=str, default='cifar10')
    args = parser.parse_args()
    dataset = ImbalanceCIFAR10("/home/ncl/ADD/data" + '/cifar10', args, imbalance_ratio, train=True, download=True)

    dataset.get_imbalanced_data()
    dataset.cls_info()



