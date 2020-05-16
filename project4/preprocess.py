# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray


    d,nTr=np.shape(xTr)  # (13,305)
    d,nTe=np.shape(xTe)  # (13,101)
    m = np.mean(xTr,1).reshape(d,1)  # (13,1)
    std = np.std(xTr,1)  # (13,)
    u = np.diag(1./std)  # (13,13)
    # xTr = u @ (xTr-np.tile(m,(1,nTr)))  # tile:(13,305)
    # xTe = u @ (xTe-np.tile(m,(1,nTe)))
    xTr = u @ (xTr-m)  # tile:(13,305)
    xTe = u @ (xTe-m)
    ## >>
    return xTr, xTe, u, m