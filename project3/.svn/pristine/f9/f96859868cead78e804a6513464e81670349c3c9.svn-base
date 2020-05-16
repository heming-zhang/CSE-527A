import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X, Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    # COMPUTE L2 DISTANCE FOR SET OF VECTORS X AND Z Efficiently
    # （X-Z)^2 = X^2 + Z^2 - 2XZ
    # get X[:,i]^2, sum along d  (n,1)->(n,m)
    X2 = np.sum(np.square(X),axis=0).reshape(-1,1)
    X2 = np.tile(X2,(1,m))
    # get Z[:,j]^2, sum along d  (1,m)->(n,m)
    Z2 = np.sum(np.square(Z),axis=0).reshape(1,-1)
    Z2 = np.tile(Z2,(n,1))
    # get 2XZ  (n,m)
    XZ2 = 2*np.dot(X.T,Z)
    # calculate the result  (n,m)
    result = np.subtract(np.add(X2,Z2),XZ2)
    # deal with numerical instability before doing square root
    result = np.where(result<0,0,result)  # turn <0 value to 0
    D = np.sqrt(result)

    # D = np.zeros((n, m))
    # for i in range(n):
    #     for j in range(m):
    #         D[i, j] = np.linalg.norm(X[:, i] - Z[:, j])

    return D