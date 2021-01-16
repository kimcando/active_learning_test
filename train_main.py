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
from utils.util_args import arg_parser
from utils.util_transform import base_transform, normalize_transform
from select_sampler import get_strategy
# import matplotlib.pyplot as plt
# import matplotlib.image as image
from models import resnet, vgg
import pdb
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
logger_name = f'{args.alg}_{args.data_name}_{args.model_name}_{args.opt_name}'
log_writer = MyLogger(args,logger_name, set_level=logging.DEBUG)
tb_writer = SummaryWriter(args.tb_path+logger_name)

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
########
if device == 'cuda':
    log_writer.info(f'using multiple gpus')
    model = torch.nn.DataParallel(model)
else:

    log_writer.info(f'single gpu')
    model.to(device)
loss_fn = nn.CrossEntropyLoss()

# if args.opt_name is 'SGD':
#     opt = torch.optim.SGD(model.parameters(),lr=args.lr_sgd, momentum=args.momentum_sgd)
# elif args.opt_name is 'Adam':
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr_adam)

#########
##dataset
#########
transform_base = normalize_transform(args)
X_tr, Y_tr, X_te, Y_te = get_dataset(args,config) #(50000, 32, 32, 3) 50000
# print(np.shape(X_tr)[1:]) # print(torch.tensor(X_tr)[1].shape)
handler = get_handler(args)
n_pool = len(Y_tr)
n_test = len(Y_te)
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)

strategy = get_strategy(args, X_tr, Y_tr, idxs_lb, model, handler, transform_base, transform_base)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:args.nStart]] = True
# train_dataset, test_dataset = get_CIFAR10(args)
# train_dataset = []
# test_dataset= []
batch_size = args.tr_batch_size

NUM_ROUND = int((args.nEnd -args.nStart)/ args.nQuery)
DATA_NAME = args.data_name

log_writer.info(f'>> Mode @ {args.strategy}')
log_writer.info(f' >> img_size@{args.img_c} :: training_bs @{args.tr_batch_size} :: testing_bs @{args.te_batch_size}')
log_writer.info(f' >> model@{args.model_name} :: loss_fn@crossentropy :: optimizer@{args.opt_name} :: lr@{args.lr_adam}')
log_writer.info(f' >> # of initial labeled pool :: @{args.nStart} & # of querying exmaples :: @{args.nQuery}')
log_writer.info(f' >> # of unlabeled pool pool :: @{len(Y_tr) - args.nStart}')
log_writer.info(f' >> # of testing pool :: @{len(Y_te)}')

#########
##initial training
#########

print('initial training')
pdb.set_trace()
strategy.train()

P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
log_writer.info(f' >> # of data @{args.nStart} || total train time @{strategy.get_time:.3f}s || test acc@{acc[0]:.3f}')

tb_writer.add_scalar('training_epoch', strategy.get_last_epoch, sum(idxs_lb))
tb_writer.add_scalar('training_time', strategy.get_time, sum(idxs_lb))
tb_writer.add_scalar('test_acc', acc[0], sum(idxs_lb))

for round in range(1,NUM_ROUND+1):
    print(f'Round@{round}')
    output = strategy.query(args.nQuery)
    q_idxs = output
    idxs_lb[q_idxs] = True
    # report weighted accuracy
    # corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[round] = 1.0*(Y_te ==P).sum().item() / len(Y_te)
    log_writer.info(f' >> # of data @{sum(idxs_lb)} || total train time @{strategy.get_time:.3f}s || test acc@{acc[round]:.3f}')

    tb_writer.add_scalar('training_epoch', strategy.get_last_epoch, sum(idxs_lb))
    tb_writer.add_scalar('training_time', strategy.get_time, sum(idxs_lb))
    tb_writer.add_scalar('test_acc', acc[round], sum(idxs_lb))
    # to do
    if args.alg == 'badge':
        tb_writer.add_scalar('choosing_time', strategy.get_sampling_time, sum(idxs_lb))
    if sum(~strategy.idxs_lb) < args.nQuery:
        sys.exit('too few remaining points to query')

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




