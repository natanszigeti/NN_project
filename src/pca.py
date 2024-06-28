import numpy as np
from numpy.linalg import eig


class PCA(object):

    def __init__(self, k: int = None):
        self._k = k
        self._scaler = None
        self._transform_matrix = None

    def fit(self, data):
        """
        Implementation of principal component analysis
        :param data: Data represented as M x N matrix, M = number of samples, N = number of features
        :return: Data reduced to matrix of size M x k
        """
        if self._k is None:
            self._k = np.shape(data)[1]
        if self._k > np.shape(data)[1]:
            raise Exception("New dimensions too large!")

        # Scale each column so the mean is 0
        self._scaler = np.mean(data, axis=0)
        scaled_data = data - self._scaler

        # Calculate covariance matrix
        cov_matrix = np.cov(scaled_data, rowvar=False)

        # Calculate eigenvalues and eigenvectors of covariance matrix
        eig_values, eig_vectors = eig(cov_matrix)
        eigs_paired = [[eig_values[i], eig_vectors[:, i]] for i in range(len(eig_values))]

        # Sort eigenvectors according to corresponding eigenvalue, largest to smallest
        eigs_paired = sorted(eigs_paired, reverse=True, key=lambda x: x[0])

        # Choose the first k eigenvectors and produce matrix transform
        principal_vectors = [pair[1] for pair in eigs_paired[:self._k]]
        self._transform_matrix = np.c_[principal_vectors].T
        return self

    def transform(self, data):
        return np.matmul(data - self._scaler, self._transform_matrix)
