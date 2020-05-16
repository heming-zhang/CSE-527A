import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV


def linear_regression(xTr, yTr):
    # model = LinearRegression().fit(xTr, yTr)
    # model = Ridge(alpha = 0.3).fit(xTr, yTr)
    model = RidgeCV(alphas = np.linspace(0.01,1,100)).fit(xTr, yTr)
    print(model.coef_)
    print(model.coef_.shape)
    return model

def linear_predict(model, xTe, xId):
    y_pred = model.predict(xTe)
    # output = np.hstack((xId, y_pred))
    # savetxt('output.csv', output, delimiter=',')
    y_pred = y_pred.tolist()
    xId = xId.tolist()
    final_pred = []
    XID = []
    for list1 in y_pred:
        final_pred.append(list1[0])
    for list2 in xId:
        XID.append(int(list2[0]))
    my_submission = pd.DataFrame({'ID': XID, 'Horizontal_Distance_To_Fire_Points': final_pred})
    my_submission.to_csv('submission.csv', index = False)

if __name__ == "__main__":
    xTr = np.load("xTr.npy").astype(float)
    yTr = np.load("yTr.npy").astype(float)
    xId = np.load("xId.npy").astype(float)
    xTe = np.load("xTe.npy").astype(float)
    model = linear_regression(xTr, yTr)
    linear_predict(model, xTe, xId)