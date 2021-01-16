# https://deeplizard.com/learn/video/0LhiS6yu2qQ
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt


def save_cm(args, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    if args.data_name == 'cifar10':
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(cm)
    plt.switch_backend('agg')
    fig = plt.figure()
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.matshow(cm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig('test.png')

class ConfusionMatrixController(object):
    def __init__(self, args):
        self.args = args
        self.set_new()
        self.set_data_type()
        self.class_wise_acc = {i: [] for i in range(len(self.classes))}
        self.tr_class_wise_acc = {i: [] for i in range(len(self.classes))}
    def set_new(self):
        self.tr_true_labels_list = np.array([])
        self.tr_pred_labels_list = np.array([])
        self.te_true_labels_list = np.array([])
        self.te_pred_labels_list = np.array([])

    def set_data_type(self):
        if self.args.data_name =='cifar10':
            self.classes =  ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def update_training(self, true_label, pred_label):
        """
        update training true label, pred label
        """
        self.tr_true_labels_list = np.append(self.tr_true_labels_list, true_label)
        self.tr_pred_labels_list = np.append(self.tr_pred_labels_list, pred_label)

    def update_testing(self, true_label, pred_label):
        self.te_true_labels_list = np.append(self.te_true_labels_list, true_label)
        self.te_pred_labels_list = np.append(self.te_pred_labels_list, pred_label)

    def normalize(self, cm):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    def class_acc_update(self):
        cls_cm = self.normalize(self.get_test_cm)
        for k, cls_acc in zip(self.class_wise_acc.keys(), cls_cm.diagonal()):
            self.class_wise_acc[k].append(cls_acc)
        return self.class_wise_acc

    def class_acc_training_update(self):
        cls_cm = self.normalize(self.get_training_cm)
        for k, cls_acc in zip(self.tr_class_wise_acc.keys(), cls_cm.diagonal()):
            self.tr_class_wise_acc[k].append(cls_acc)
        return self.tr_class_wise_acc

    def get_cls_wise_acc(self, cls_cm_te):
        if self.class_wise_acc != cls_cm_te:
            raise ValueError
        cls_acc = {i:[] for i in range(len(self.classes))}
        for k, v in self.class_wise_acc.items():
            cls_acc[k].append(v[-1])
        return cls_acc

    def get_cls_training_wise_acc(self, cls_cm_te):
        if self.tr_class_wise_acc != cls_cm_te:
            raise ValueError
        cls_acc = {i:[] for i in range(len(self.classes))}
        for k, v in self.tr_class_wise_acc.items():
            cls_acc[k].append(v[-1])
        return cls_acc

    @property
    def get_training_cm(self):
        return confusion_matrix(self.tr_true_labels_list, self.tr_pred_labels_list)

    @property
    def get_test_cm(self):
        return confusion_matrix(self.te_true_labels_list, self.te_pred_labels_list)
