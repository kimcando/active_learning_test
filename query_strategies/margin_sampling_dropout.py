import numpy as np
import torch
from .strategy import Strategy

class MarginSamplingDropout(Strategy):
    def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform, n_drop=10):
        super(MarginSamplingDropout, self).__init__(args, X, Y, idxs_lb, net, handler, tr_transform, te_transform)
        self.n_drop = n_drop

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pdb.set_trace()
        probs = self.predict_prob_dropout(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1][:n]]
