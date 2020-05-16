import numpy as np


def ridge(w, xTr, yTr, lambdaa):
    # INPUT:
    # w weight vector (default w=0)
    # xTr:dxn matrix (each column is an input vector)
    # yTr:1xn matrix (each entry is a label)
    # lambdaa: regression constant
    #
    # OUTPUTS:
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w
    #
    # [d,n]=size(xTr);

    # first transpose yTr to nx1 matrix
    yTr = yTr.T
    # xTr dxn matrix  &  w dx1 matrix
    Xw = np.dot(xTr.T, w)
    ww = np.dot(w.T, w)
    XXw = np.dot(xTr, Xw)
    Xy = np.dot(xTr, yTr)
    # calculate loss and gradient 
    dims = np.shape(xTr)[0]
    gradient = np.zeros((dims, 1))
    loss = np.dot((Xw - yTr).T, Xw - yTr) + lambdaa * ww
    gradient = (2 * (XXw - Xy + lambdaa * w))
    return loss, gradient
