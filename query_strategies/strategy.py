import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import sys, time
from copy import deepcopy
from abc import ABC, abstractmethod

class Strategy:
    def __init__(self, args, X, Y, idxs_lb, model, handler, tr_transform, te_transform=None):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.model = model
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.tr_transform = tr_transform
        if te_transform is None:
            self.te_transform = tr_transform
        else:
            self.te_transform = te_transform
        use_cuda = torch.cuda.is_available()

    def query(self, n):
        pass
    
    def update(self, idxs_lb):
        """
        update labelling
        :param idxs_lb: 
        :return: 
        """
        self.idxs_lb = idxs_lb
    
    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        
        # n_epoch = self.args.num_epochs
        self.new_model = self.model.apply(weight_reset).cuda()
        opt = self.get_optimizer()
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        # Y type numpy 아님??
        train_loader = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                               transform = self.tr_transform),
                                               shuffle=True,
                                               batch_size = self.args.tr_batch_size,
                                               num_workers= self.args.num_workers
                                               )
        epoch = 1
        accCurrent = 0.
        t0 = time.time()
        while accCurrent < 0.98:
            accCurrent = self._train(epoch, train_loader, opt)
            epoch += 1
            print(f'Epoch @{epoch} :: training accuracy@{accCurrent:.3f}')

            if (epoch%50 ==0) and (accCurrent <0.2): # reset if not converging
                self.new_model = self.model.apply(weight_reset)
                opt = self.get_optimizer()
                sys.exit('did not converge')
        self.train_time = time.time()-t0 #converging not considered yet
        self.last_epoch = epoch

    def predict(self, test_X, test_Y):
        if type(test_X) is np.ndarray:
            test_loader = DataLoader(self.handler(test_X, test_Y, transform = self.te_transform),
                                      shuffle=False,
                                      batch_size=self.args.te_batch_size,
                                      num_workers=self.args.num_workers)
        else:
            test_loader = DataLoader(self.handler(test_X.numpy(), test_Y, transform=self.te_transform),
                                     shuffle=False,
                                     batch_size=self.args.te_batch_size,
                                     num_workers=self.args.num_workers)
        self.new_model.eval()
        P = torch.zeros(len(test_Y)).long()
        with torch.no_grad():
            # one shot으로 testing 해버리네
            for in_, target_, idxs in test_loader:
                in_, target_ = in_.cuda(), target_.cuda()
                out, e1 = self.new_model(in_)
                pred = torch.argmax(out, dim=1)#out.max(out,1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def get_optimizer(self):
        return getattr(torch.optim, self.args.opt_name)(self.new_model.parameters(),
                                                          lr=self.args.lr)
        # return getattr(torch.optim, self.args.opt_name)(self.model.parameters(),
        #                                                   **self.config['opt_options'])
    @property
    def get_time(self):
        return self.train_time

    @property
    def get_last_epoch(self):
        return self.last_epoch

    def _train(self, epoch, train_loader, opt):
        self.new_model.train()
        accFinal = 0.
        for batch_idx, (in_, target_, idxs) in enumerate(train_loader):
            in_, target_ = in_.cuda(), target_.cuda()
            opt.zero_grad()
            out, e1 = self.new_model(in_)
            loss = F.cross_entropy(out, target_)
            loss.backward()
            accFinal += torch.sum(torch.max(out, 1)[1] == target_).float().detach().item()

            # clamp gradients,
            for p in filter(lambda p:p.grad is not None, self.new_model.parameters()): p.grad.data.clamp_(min=-.1,max=.1)
            opt.step()
        return accFinal/ len(train_loader.dataset.X)

    def predict_prob(self, test_X, test_Y):
        test_loader = DataLoader(self.handler(test_X, test_Y, transform=self.te_transform),
                                 shuffle=False,
                                 batch_size=self.args.te_batch_size,
                                 num_workers=self.args.num_workers)
        self.new_model.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.test_Y))])
        with torch.no_grad():
            for in_, target_, idxs in test_loader:
                in_, target_ = in_.cuda(), target_.cuda()
                out, e1 = self.new_model(in_)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().item()

        return probs

    def predict_prob_dropout(self, test_X, test_Y, n_drop):
        test_loader = DataLoader(self.handler(test_X, test_Y, transform=self.te_transform),
                                 shuffle=False,
                                 batch_size=self.args.te_batch_size,
                                 num_workers=self.args.num_workers)
        self.new_model.eval()
        probs = torch.zeros([len(test_Y), len(np.unique(test_Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print(f'n_drop {i+1}/{n_drop}')
                for in_, target_, idxs in test_loader:
                    in_, target_ = in_.cuda(), target_.cuda()
                    out, e1 = self.new_model(in_)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu().item()
        probs /= n_drop
        return probs
    
    def predict_prob_dropoutsplit(self, test_X, test_Y, n_drop):
        test_loader = DataLoader(self.handler(test_X, test_Y, transform=self.te_transform),
                                 shuffle=False,
                                 batch_size=self.args.te_batch_size,
                                 num_workers=self.args.num_workers)
        self.new_model.eval()
        probs = torch.zeros([n_drop, len(test_Y), len(np.unique(test_Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print(f'n_drop {i+1}/{n_drop}')
                for in_, target_, idxs in test_loader:
                    in_, target_ = in_.cuda(), target_.cuda()
                    out, e1 = self.new_model(in_)
                    probs[i][idxs] += F.softmax(out, dim_1).cpu().item()
        return probs
    
    def get_embedding(self, test_X, test_Y):
        test_loader = DataLoader(self.handler(test_X, test_Y, transform=self.te_transform),
                                 shuffle=False,
                                 batch_size=self.args.te_batch_size,
                                 num_workers=self.args.num_workers)
        self.new_model.eval()
        embedding = torch.zeros([len(test_Y), self.new_model.get_embedding_dim()])
        with torch.no_grad():
            for in_, target_, idxs in test_loader:
                in_, target_ = in_.cuda(), target_.cuda()
                out, e1 = self.new_model(in_)
                embedding[idxs] = e1.cpu().item()
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    # actually not test datapoints!
    def get_grad_embedding(self, test_X, test_Y):
        embDim = self.new_model.get_embedding_dim()
        nLab = len(np.unique(test_Y))
        embedding = np.zeros([len(test_Y), embDim * nLab])
        test_loader = DataLoader(self.handler(test_X, test_Y, transform=self.te_transform),
                                 shuffle=False,
                                 batch_size=self.args.te_batch_size,
                                 num_workers=self.args.num_workers)
        self.new_model.eval()
        with torch.no_grad():
            for in_, target_, idxs in test_loader:
                in_, target_ = in_.cuda(), target_.cuda()
                cout, out = self.new_model(in_)
                out = out.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).detach().cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)

                for j in range(len(target_)):

                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)