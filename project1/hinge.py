from numpy import maximum
import numpy as np


def hinge(w, xTr, yTr, lambdaa):
    # INPUT:
    # xTr dxn matrix (each column is an input vector)
    # yTr 1xn matrix (each entry is a label)
    # lambda: regularization constant
    # w weight vector (default w=0)
    #
    # OUTPUTS:
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w

    # YOUR CODE HERE
    # # first transpose yTr to nx1 matrix
    # yTr = yTr.T
    # # xTr dxn matrix  &  w dx1 matrix
    # Xw = np.dot(xTr.T, w)
    # yXw = yTr * Xw
    # ww = np.dot(w.T, w)
    # # calculate loss and gradient
    # loss = 0
    # points = np.shape(xTr)[1]
    # dims = np.shape(xTr)[0]
    # gradient = np.zeros((dims, 1))
    # for i in range(points):
    #     if 1 - yXw[i] > 0: #yxw[i] <= 1
    #         loss = loss + 1 - yXw[i]
    #         gradient = gradient + (-yTr[i] * (xTr.T)[i]).reshape((dims, 1))
    # gradient = gradient + 2 * lambdaa * w
    # loss = loss + 2 * lambdaa * ww

    yTr = yTr.T # nx1
    Xw = np.dot(xTr.T, w)
    yXw = yTr * Xw
    ww = np.dot(w.T, w)
    t = 1 - yXw
    t2 = t > 0
    loss = np.sum(t * t2) + lambdaa * ww
    # print(loss)
    gradient = 2 * lambdaa * w + np.sum(-yTr * xTr.T * t2, axis = 0).reshape(w.shape)
    return loss, gradient