"""
function K = computeK(kernel_type, X, Z)
computes a matrix K such that Kij=g(x,z);
for three different function linear, rbf or polynomial.

Input:
kernel_type: either 'linear','poly','rbf'
X: n input vectors of dimension d (dxn);
Z: m input vectors of dimension d (dxn);
kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)

OUTPUT:
K : nxm kernel matrix
"""

import numpy as np
from l2distance import l2distance

def computeK(kernel_type, X, Z, kpar):
    assert kernel_type in ['linear', 'poly', 'rbf'], kernel_type + ' is an unrecognized kernel type in computeK'
    
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to computeK'
    
    K = np.zeros((n,m))
    
    # JUDGE WHICH TYPE['linear', 'poly', 'rbf'] OF KERNEL
    if kernel_type == 'linear':
        K = np.dot(X.T, Z)
    
    if kernel_type == 'poly':
        # Ktmp = np.add(np.dot(X.T, Z), 1)
        Ktmp = np.dot(X.T, Z) + np.ones((n, m))
        K = Ktmp ** kpar

    if kernel_type == 'rbf':
        dist_square_matrix = l2distance(X, Z) ** 2
        K = np.exp(-kpar * dist_square_matrix)
    
    return K
