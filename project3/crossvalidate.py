"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
from sklearn.model_selection import KFold
import math
from trainsvm import trainsvm

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    k = 10
    kf = KFold(n_splits = k)
    for i in range(len(Cs)):
        for j in range(len(paras)):
            val_error = 0
            for train_index, val_index in kf.split(yTr):
                X_train, X_val = (xTr.T[train_index]).T, (xTr.T[val_index]).T
                y_train, y_val = yTr[train_index], yTr[val_index]
                svmclassify = trainsvm(X_train, y_train, Cs[i], 'rbf', paras[j])
                val_preds = svmclassify(X_val)
                val_error += np.mean(val_preds != y_val)
            val_error /= k
            errors[j][i] = val_error
    lowest_error = np.amin(errors)
    P_idxs, C_idxs = np.where(errors==lowest_error)
    bestP,bestC = P_idxs[0], C_idxs[0]

    
    return bestC, bestP, lowest_error, errors


    