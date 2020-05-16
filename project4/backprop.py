# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:53 2019

@author: Jerry Xing
"""
import numpy as np
def backprop(W, aas,zzs, yTr,  trans_func_der):
#% function [gradient] = backprop(W, aas, zzs, yTr,  der_trans_func)
#%
#% INPUT:
#% W weights (list of ndarray)
#% aas output of forward pass (list of ndarray)
#% zzs output of forward pass (list of ndarray)
#% yTr 1xn ndarray (each entry is a label)
#% der_trans_func derivative of transition function to apply for inner layers
#%
#% OUTPUTS:
#% 
#% gradient = the gradient at w as a list of ndarries
#%

    n = np.shape(yTr)[1]
    delta = zzs[0] - yTr
    
    # compute gradient with back-prop
    gradient = [None] * len(W)
    for i in range(len(W)):
        # INSERT CODE HERE:
        # gradient[i] = delta.dot(zzs[i + 1].T) / n
        # n2 = W[i].shape[1]
        # a = W[i][:, [0, n2 - 1]]
        # delta = np.multiply(trans_func_der(aas[i + 1]), (W[i][:, 0:n2 - 1].T.dot(delta)))


        gradient[i] = delta @ zzs[i + 1].T/n   # (1,21),(20,21)
        n2 = W[i].shape[1]   # 20
        # print("delta shape",delta.shape)  #(1,305),(20,305)
        # print("zzs[i + 1] shape", zzs[i + 1].shape)  #(21,305)
        # print("W shape",W[i].shape)  # (1,21),(20,21)
        # print("trans_func_der(aas[i + 1])",trans_func_der(aas[i + 1]).shape)  #(20,305)
        # print("W[i][:,:n2-1]).T shape",W[i][:,:n2-1].T.shape)  #(20,1),(20,20)
        delta = trans_func_der(aas[i + 1]) * ((W[i][:,:n2-1]).T @ delta)  # *-element-wise, @-matrix

    return gradient 


