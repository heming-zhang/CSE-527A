"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""
import numpy as np
from cvxopt import matrix

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]

    # Objective Function
    Q = np.dot(yTr, yTr.T) * K
    p = np.ones((n, 1)) * (-1)
    # Restriction [0 < G.alpha < C] nx1
    gc = np.eye(n)
    hc = np.ones((n, 1)) * C
    g0 = np.eye(n) * (-1)
    h0 = np.zeros((n, 1))
    G = np.vstack((gc, g0))
    h = np.vstack((hc, h0))
    # Equation  [A.alpha = 0] scalar
    A = np.array(yTr.T)
    b = np.zeros((1, 1))

    return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)


if __name__ == "__main__":
    n = 20
    K = np.ones((n, n))
    yTr = np.ones((n, 1))
    C = 0.5
    generateQP(K, yTr, C)