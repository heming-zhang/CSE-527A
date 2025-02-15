#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Heming
"""

import numpy as np

def classifyLinear(x, w, b):
# =============================================================================
#function preds=classifyLinear(x,w,b);
#
#Make predictions with a linear classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#w : weight vector
#b : bias
#
#Output:
#preds: predictions
# =============================================================================

    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    W = np.matrix(w)
    
# =============================================================================
# fill in code here

    # construct b as bias vector with n x 1 dimensions
    d, n = X.shape
    b = b * np.ones([1, n])
    # classification results
    result = np.dot(W.T, X) + b
    preds = np.sign(result)

    return preds
# =============================================================================
