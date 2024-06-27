import numpy as np
from numpy.linalg import eig


def calculate_pca(data, k: int = None):
    """
    Implementation of principal component analysis
    :param data: Data represented as M x N matrix, M = number of samples, N = number of features
    :param k: Number of dimensions to reduce data to; if not specified, k = N
    :return: Data reduced to matrix of size M x k
    """
    if k is None:
        k = np.shape(data)[1]
    if k > np.shape(data)[1]:
        raise Exception("New dimensions too large!")

    # Scale each column so the mean is 0
    scaled_data = data - np.mean(data, axis=0)

    # Calculate covariance matrix
    cov_matrix = np.cov(scaled_data, rowvar=False)

    # Calculate eigenvalues and eigenvectors of covariance matrix
    eig_values, eig_vectors = eig(cov_matrix)
    eigs_paired = [[eig_values[i], eig_vectors[:, i]] for i in range(len(eig_values))]

    # Sort eigenvectors according to corresponding eigenvalue, largest to smallest
    eigs_paired = sorted(eigs_paired, reverse=True, key=lambda x: x[0])

    # Choose the first k eigenvectors and produce matrix transform
    principal_vectors = [pair[1] for pair in eigs_paired[:k]]
    transform_matrix = np.c_[principal_vectors]
    return np.matmul(scaled_data, transform_matrix.T)
