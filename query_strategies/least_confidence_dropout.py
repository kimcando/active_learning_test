import numpy as np
import torch
from .strategy import Strategy

class LeastConfidenceDropout(Strategy):
	def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform, n_drop=10):
		super(LeastConfidenceDropout, self).__init__(args, X, Y, idxs_lb, net, handler, tr_transform, te_transform)
		self.n_drop = n_drop

	def query(self, n):
		t0 = time.time()
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
		U = probs.max(1)[0]
		self.sampling_time = time.time() - t0
		return idxs_unlabeled[U.sort()[1][:n]]

	@property
	def get_sampling_time(self):
		return self.sampling_time
