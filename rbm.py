import numpy as np

class RBM:
	def __init__(self, numV, numH, L = 0.1, sigma = 0.1):
		self.numV = numV
		self.numH = numH
		self.L = L
		self.sigma = sigma
		self.W = self.sigma * np.random.randn(self.numV, self.numH)
		self.Bh = np.zeros(self.numH)
		self.Bv = np.zeros(self.numV)

	def _HFromV(self, V):
		H = np.zeros(np.shape(V)[0], self.numH)
		activation = self._logistic(self.Bh + np.dot(V, self.W))
		for i in xrange(np.shape(V)[0]):
			for j in xrange(self.numH):
				H[i, j] = (activation(i, j) <= np.random.random())

		return H

	def _VFromH(self, H):
		V = np.zeros(np.shape(H)[0], self.numV)
		activation = self._logistic(self.Bv + np.dot(self.W, H.T)).T
		for i in xrange(np.shape(H)[0]):
			for j in xrange(self.numV):
				V[i, j] = (activation(i, j) <= np.random.random())

		return V

	def _logistic(self, x):
		return 1. / (1. + np.exp(-x))

	def _gradient(self, V, H):
		return np.dot(V.T, H)