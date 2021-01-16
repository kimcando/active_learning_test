import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import os, time , logging, yaml, sys, random

from torch.utils.tensorboard import SummaryWriter
from data_handler.get_data import get_dataset, get_handler

from utils.util_log import MyLogger, MyWorkBook
from utils.util_args import arg_parser
from utils.util_transform import base_transform, normalize_transform
from select_sampler import get_strategy
from cifar_loader import ImbalanceCIFAR10
# import matplotlib.pyplot as plt
# import matplotlib.image as image
from models import resnet, vgg
import pdb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from confusion_save import ConfusionMatrixController
########
##load
########
args = arg_parser()
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device_id)
# print(torch.cuda.current_device())

########
##logger
########
if args.imbal_mode:
    mode = 'imbal_test'
else:
    mode = 'basic_test'
imbalance_ratio = np.hstack([1.0] * 10)
# if config['ratio']:
#     imbalance_ratio = np.hstack(config['ratio'])

for i in args.ratio:
    imbalance_ratio = np.array([float(i) for i in args.ratio[0].split(',')])

logger_name = f'{mode}_{args.split_order}_{args.alg}_{args.data_name}_{args.model_name}_{args.opt_name}'
log_writer = MyLogger(args,logger_name, set_level=logging.DEBUG)
tb_writer = SummaryWriter(args.tb_path+logger_name)
log_writer.info(f'{mode} with imbalance_ratio: {imbalance_ratio}')

#########
##model
#########
if args.model_name is 'resnet18':
    # model = resnet.resnet18(num_classes=config['num_classes'],pretrained=False)
    model = resnet.ResNet18(num_cls=config['num_classes'])
    print(model)
elif args.model_name is 'lenet':
    model = lenet.LeNet(num_classes=config['num_classes'])
elif args.model_name is 'vgg':
    model = net = vgg.VGG('VGG16')
else:
    raise NotImplementedError

#########
##parallel
#########
if args.parallel:
    if device == 'cuda':
        log_writer.info(f'parallel mode with using {torch.cuda.device_count() } gpus')
        model = torch.nn.DataParallel(model)
    else:
        log_writer.info(f'no cuda found')
        model.to(device)
else:
    log_writer.info(f'parallel mode is {args.parallel}')
    model.to(device)

#########
##loss, optimizer
#########
loss_fn = nn.CrossEntropyLoss()
if args.opt_name is 'SGD':
    opt = torch.optim.SGD(model.parameters(),lr=args.lr_sgd, momentum=args.momentum_sgd)
elif args.opt_name is 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_adam)

#########
##dataset : Training
#########
transform_base = normalize_transform(args)
train_dataset = ImbalanceCIFAR10('./data', args, imbalance_ratio, log_writer, train=True, download=False, transform=transform_base)
train_loader = DataLoader(train_dataset,
                          shuffle=args.shuffle, #shuffle effect?
                          batch_size=args.tr_batch_size,
                          num_workers=args.num_workers)
#########
##dataset : Testing
#########
cifar_path = '/home/ncl/data/'
test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform_base)
test_loader = DataLoader(test_dataset,
                         shuffle=args.shuffle,
                         batch_size=args.te_batch_size,
                         num_workers=args.num_workers)
# pdb.set_trace()

#########
##base learning mode
#########
cm_obj = ConfusionMatrixController(args)
for epoch in range(args.num_epochs):
    log_writer.info(f'Epoch@@{epoch}')
    running_loss = []
    running_acc = []
    cm_obj.set_new() # new accumulate

    model.train()
    t0 = time.time()

    for batch_idx, (input_, target_, idxs) in enumerate(train_loader):
        input_ , target_ = input_.cuda(), target_.cuda()
        out_, _ = model(input_)
        loss = loss_fn(out_, target_)
        # img check
        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = torch.argmax(out_, dim=1)
        acc = (pred==target_).float().sum()
        cm_obj.update_training(target_.detach().cpu().numpy(),pred.detach().cpu().numpy()) #

        running_loss.append(loss.detach())
        running_acc.append(acc)
    img_grid = torchvision.utils.make_grid(input_)
    tb_writer.add_image('img', img_grid, epoch)
    tb_writer.add_text('label', str(target_), epoch)
    t1 = time.time()-t0
    # pdb.set_trace()
    total_acc = np.sum(running_acc)/ len(train_dataset.new_idxs)
    total_loss = np.sum(running_loss)/len(running_loss)
    if args.needs_confusion_matrix:
        cm_tr = cm_obj.get_training_cm
        print(cm_tr)
        cls_cm_tr = cm_obj.class_acc_training_update()
        cls_acc_tr = cm_obj.get_cls_training_wise_acc(cls_cm_tr)
        log_writer.info(f'TRAINING class wise acc@{epoch} :: {cls_acc_tr}')

    log_writer.info(f'TRAIINING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')
    tb_writer.add_scalar('training/acc', total_acc, epoch)
    tb_writer.add_scalar('training/loss', total_loss, epoch)
    tb_writer.add_scalar('training/time',t1 , epoch)
    for i in range(len(cls_acc_tr)):
        tb_writer.add_scalar(f'training_cls_acc/{i}_{cm_obj.classes[i]}', cls_acc_tr[i][-1], epoch)

    with torch.no_grad():
        model.eval()
        test_running_loss = []
        test_running_acc = []
        # test_true_labels_list, test_pred_labels_list = np.array([]), np.array([])
        # pdb.set_trace()
        for batch_idx, (input_, target_) in enumerate(test_loader):
            input_, target_ = input_.cuda(), target_.cuda()

            out_, _ = model(input_)
            loss = loss_fn(out_, target_)
            pred = torch.argmax(out_, dim=1)
            acc = (pred == target_).float().sum()
            cm_obj.update_testing(target_.detach().cpu().numpy(),pred.detach().cpu().numpy()) #

            test_running_loss.append(loss.detach())
            test_running_acc.append(acc)
        # pdb.set_trace()
        img_grid = torchvision.utils.make_grid(input_)
        tb_writer.add_image('test_img', img_grid, epoch)
        tb_writer.add_text('test_label', str(target_), epoch)
        total_acc = np.sum(test_running_acc) / len(test_dataset.targets)
        total_loss = np.sum(test_running_loss) / len(test_running_loss)
        if args.needs_confusion_matrix:
            cm_te = cm_obj.get_test_cm
            print(cm_te)
            cls_cm_te = cm_obj.class_acc_update()
            cls_acc = cm_obj.get_cls_wise_acc(cls_cm_te)
            log_writer.info(f'TESTING class wise acc@{epoch} :: {cls_acc}')
        log_writer.info(f'TESTING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')
        tb_writer.add_scalar('testing/acc', total_acc, epoch)
        tb_writer.add_scalar('testing/loss', total_loss, epoch)
        for i in range(len(cls_acc)):
            tb_writer.add_scalar(f'testing_cls_acc/{i}_{cm_obj.classes[i]}', cls_acc[i][-1], epoch)


# #########
# ##active learning mode
# #########
# handler = get_handler(args)
# n_pool = len(Y_tr)
# n_test = len(Y_te)
# idxs_lb = np.zeros(n_pool, dtype=bool)
# idxs_tmp = np.arange(n_pool)
#
# strategy = get_strategy(args, X_tr, Y_tr, idxs_lb, model, handler, transform_base, transform_base)
# np.random.shuffle(idxs_tmp)
# idxs_lb[idxs_tmp[:args.nStart]] = True
# # train_dataset, test_dataset = get_CIFAR10(args)
# # train_dataset = []
# # test_dataset= []
# batch_size = args.tr_batch_size
#
# NUM_ROUND = int((args.nEnd -args.nStart)/ args.nQuery)
# DATA_NAME = args.data_name
#
# log_writer.info(f'>> Mode @ {args.strategy}')
# log_writer.info(f' >> img_size@{args.img_c} :: training_bs @{args.tr_batch_size} :: testing_bs @{args.te_batch_size}')
# log_writer.info(f' >> model@{args.model_name} :: loss_fn@crossentropy :: optimizer@{args.opt_name} :: lr@{args.lr_adam}')
# log_writer.info(f' >> # of initial labeled pool :: @{args.nStart} & # of querying exmaples :: @{args.nQuery}')
# log_writer.info(f' >> # of unlabeled pool pool :: @{len(Y_tr) - args.nStart}')
# log_writer.info(f' >> # of testing pool :: @{len(Y_te)}')
#
# #########
# ##initial training
# #########
#
# print('initial training')
# strategy.train()
#
# P = strategy.predict(X_te, Y_te)
# acc = np.zeros(NUM_ROUND+1)
# acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
# log_writer.info(f' >> # of data @{args.nStart} || total train time @{strategy.get_time:.3f}s || test acc@{acc[0]:.3f}')
#
# tb_writer.add_scalar('training_epoch', strategy.get_last_epoch, sum(idxs_lb))
# tb_writer.add_scalar('training_time', strategy.get_time, sum(idxs_lb))
# tb_writer.add_scalar('test_acc', acc[0], sum(idxs_lb))
#
# for round in range(1,NUM_ROUND+1):
#     print(f'Round@{round}')
#     output = strategy.query(args.nQuery)
#     q_idxs = output
#     idxs_lb[q_idxs] = True
#     # report weighted accuracy
#     # corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]
#
#     # update
#     strategy.update(idxs_lb)
#     strategy.train()
#
#     # round accuracy
#     P = strategy.predict(X_te, Y_te)
#     acc[round] = 1.0*(Y_te ==P).sum().item() / len(Y_te)
#     log_writer.info(f' >> # of data @{sum(idxs_lb)} || total train time @{strategy.get_time:.3f}s || test acc@{acc[round]:.3f}')
#
#     tb_writer.add_scalar('training_epoch', strategy.get_last_epoch, sum(idxs_lb))
#     tb_writer.add_scalar('training_time', strategy.get_time, sum(idxs_lb))
#     tb_writer.add_scalar('test_acc', acc[round], sum(idxs_lb))
#     # to do
#     if args.alg == 'badge':
#         tb_writer.add_scalar('choosing_time', strategy.get_sampling_time, sum(idxs_lb))
#     if sum(~strategy.idxs_lb) < args.nQuery:
#         sys.exit('too few remaining points to query')


# for epoch in range(args.num_epochs):
#     running_loss = []
#     running_acc = []
#     model.train()
#     for input_, target_ in train_loader:
#         input_ , target_ = input_.to(device), target_.to(device)
#         out_ = model(input_)
#         loss = loss_fn(out_, target_)
#
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         pred = torch.argmax(out_, dim=1)
#         acc = (pred==target_).float().sum()
#         running_loss.append(loss.detach())
#         running_acc.append(acc)
#     total_acc = np.sum(running_acc)/ len(running_acc)
#     total_loss = np.sum(running_loss)/len(running_loss)
#     log_writer.info(f'TRAIINING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')
#
#     with torch.no_grad():
#         model.eval()
#         test_running_loss = []
#         test_running_acc = []
#         for input_, target_ in test_loader:
#             input_, target_ = input_.to(device), target_.to(device)
#             out_ = model(input_)
#             loss = loss_fn(out_, target_)
#             pred = torch.argmax(out_, dim=1)
#             acc = (pred == target_).float().sum()
#             test_running_loss.append(loss.detach())
#             test_running_acc.append(acc)
#         total_acc = np.sum(running_acc) / len(running_acc)
#         total_loss = np.sum(running_loss) / len(running_loss)
#         log_writer.info(f'TESTING epoch@{epoch} ::  acc@{total_acc:.3f}  loss@{total_loss:.3f}')
#


tb_writer.close()
wb.save()