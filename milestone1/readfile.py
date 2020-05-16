import csv
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# def read_test_file(path, col_mean, col_var):
def read_test_file(path):
    testfile = open(path)
    csv_reader = csv.reader(testfile, delimiter = ",")
    xId = np.zeros([1, 1]).reshape(1, 1)
    xTe = np.zeros([1, 11]).reshape(1, 11)
    count = 0
    for row in csv_reader:
        x_id = np.array(row[0])
        x_data = np.array(row[1:])
        if(count != 0):
            # decomposition soil into climate, geo and delete soil
            soil = int(row[-1])
            climate = np.array([int(soil / 1000)])
            geo = np.array([int((soil % 1000) / 100)])
        else:
            soil = row[-1]
            climate = np.array([0])
            geo = np.array([0])
        x_data = np.delete(x_data, -1)
        x_data = np.append(x_data, climate)
        x_data = np.append(x_data, geo)
        xTe = np.vstack((xTe, x_data))
        xId = np.vstack((xId, x_id))
        count = count + 1
    xId = np.delete(xId, [0, 1], 0)
    xTe = np.delete(xTe, [0, 1], 0).astype(int)
    # encode climate and geo data
    climate_array = xTe[:, -2]
    geo_array = xTe[:, -1]
    encode_climate = num_encode(climate_array)
    encode_geo = num_encode(geo_array)
    xTe = np.hstack((xTe, encode_climate))
    xTe = np.hstack((xTe, encode_geo))
    xTe = np.delete(xTe, [9, 10], 1)
    # # use mean, sigma of each column from xTr
    # xNorm = (xTe - col_mean) / col_var
    # xTe = xNorm
    np.save("xId.npy", xId)
    np.save("xTe.npy", xTe)
    return xTe


def read_train_file(path):
    testfile = open(path)
    csv_reader = csv.reader(testfile, delimiter = ",")
    xTr = np.zeros([1, 11]).reshape(1, 11)
    yTr = np.zeros([1, 1]).reshape(1, 1)
    count = 0
    for row in csv_reader:
        x_data = np.array(row[1:11])
        y_data = np.array(row[11])
        if(count != 0):
            # decomposition soil into climate, geo and delete soil
            soil = int(row[-2])
            climate = np.array([int(soil / 1000)])
            geo = np.array([int((soil % 1000) / 100)])
        else:
            soil = row[-2]
            climate = np.array([0])
            geo = np.array([0])
        count = count + 1
        x_data = np.delete(x_data, -1)
        x_data = np.append(x_data, climate)
        x_data = np.append(x_data, geo)
        xTr = np.vstack((xTr, x_data))
        yTr = np.vstack((yTr, y_data))
    xTr = np.delete(xTr, [0, 1], 0).astype(int)
    yTr = np.delete(yTr, [0, 1], 0).astype(int)
    # # compute mean, sigma of each column
    # col_var = np.var(xTr, axis = 0)
    # col_mean = np.mean(xTr, axis = 0)
    # xNorm = (xTr - col_mean) / col_var
    # xTr = xNorm
    # encode climate and geo data
    climate_array = xTr[:, -2]
    geo_array = xTr[:, -1]
    encode_climate = num_encode(climate_array)
    encode_geo = num_encode(geo_array)
    xTr = np.hstack((xTr, encode_climate))
    xTr = np.hstack((xTr, encode_geo))
    xTr = np.delete(xTr, [9, 10], 1)
    np.save("xTr.npy", xTr)
    np.save("yTr.npy", yTr)
    return xTr, yTr
    # return xTr, yTr, col_mean, col_var


def num_encode(column):
    # binary encode
    # enc = OneHotEncoder(sparse=False)
    enc = OneHotEncoder()
    column = column.reshape(-1, 1)
    enc.fit(column)
    encode_col = enc.transform(column).toarray()
    return encode_col


if __name__ == "__main__":
    # path = "train.csv"
    # xTr, yTr, col_mean, col_var = read_train_file(path)
    # path = "test.csv"
    # read_test_file(path, col_mean, col_var)
    train_path = "train.csv"
    xTr, yTr = read_train_file(train_path)
    test_path = "test.csv"
    read_test_file(test_path)