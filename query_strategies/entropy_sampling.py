import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform):
		super(EntropySampling, self).__init__(args, X, Y, idxs_lb, net, handler, tr_transform, te_transform)

	def query(self, n):
		t0 = time.time()
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		self.sampling_time = time.time() - t0
		return idxs_unlabeled[U.sort()[1][:n]]

	@property
	def get_sampling_time(self):
		return self.sampling_time