import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform):
        super(RandomSampling, self).__init__(args, X, Y, idxs_lb, net, handler,
                                             tr_transform, te_transform)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
