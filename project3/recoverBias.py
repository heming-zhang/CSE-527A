"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K, yTr, alphas, C):

    # Final Optimal Distance alphas_i Range (0, C)
    n, _ = yTr.shape
    distance = np.abs(alphas - C/2)
    s = np.argmin(distance)
    # Fetch the Corresponding Kernel(x_s, x_[1,2,...n]) and y_s
    # Since Kernel is symetric Kernel(x_[1,2,...n], x_s) Equal
    k_si = K[s, :].reshape(-1, 1)
    y_s = yTr[s]
    # Compute Bias
    # bias_1 = y_s - np.sum(alphas * yTr * k_si)
    bias = y_s - np.dot(alphas.T, (yTr * k_si))

    return bias 
    
