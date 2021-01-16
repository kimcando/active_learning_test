import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
	def __init__(self, args, X, Y, idxs_lb, net, handler, tr_transform, te_transform):
		super(KMeansSampling, self).__init__(args, X, Y, idxs_lb, net, handler, tr_transform, te_transform)

	def query(self, n):
		t0 = time.time()
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		embedding = embedding.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers)**2
		dis = dis.sum(axis=1)
		q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
		self.sampling_time = time.time() - t0

		return idxs_unlabeled[q_idxs]

	@property
	def get_sampling_time(self):
		return self.sampling_time
