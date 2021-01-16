import numpy as np
from .strategy import Strategy
import pdb

class MarginSampling(Strategy):
    def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform):
        super(MarginSampling, self).__init__(args, X, Y, idxs_lb, net, handler, tr_transform, te_transform)

    def query(self, n):
        t0 = time.time()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        self.sampling_time = time.time() - t0
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]

    @property
    def get_sampling_time(self):
        return self.sampling_time
