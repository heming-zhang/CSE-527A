import math
import numpy as np

# def logistic(w, xTr, yTr, lambdaa):
def logistic(w, xTr, yTr):
    # INPUT:
    # xTr dxn matrix (each column is an input vector)
    # yTr 1xn matrix (each entry is a label)
    # w weight vector (default w=0)
    # 
    # OUTPUTS:
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w
    # [d, n] = size(xTr);

    # xTr dxn matrix  &  w dx1 matrix
    Xw = np.dot(xTr.T, w)
    yXw = (yTr.T) * Xw
    # (1xn) (dxn) => (dxn)
    yX = (yTr * xTr).T 
    # calculate the loss and gradient for n points
    loss = 0    
    points = np.shape(xTr)[1]
    dims = np.shape(xTr)[0]
    gradient = np.zeros((dims, 1))
    for i in range(0, points):
        loss = loss + math.log(1 + math.exp(-yXw[i]))
        gradient = gradient + ((-yX[i]) / (1 + math.exp(yXw[i]))).reshape(dims, 1)
    # gradient = gradient + 2 * lambdaa * w
    return loss, gradient