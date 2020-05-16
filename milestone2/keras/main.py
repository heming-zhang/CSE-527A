import numpy as np
import pandas as pd
from keras_nn import KerasNN, RunNN
from keras.models import load_model

def build_rand_nn(layer0, dropout0, layer1, dropout1):
    model = KerasNN().keras_nn(layer0, dropout0, layer1, dropout1)
    return model

def run_nn(model, epoch, batch_size, verbose):
    # AUTO UPDATE WEIGHT
    xTr = np.load('./xTr.npy')
    yTr = np.load('./yTr.npy')
    model, history = RunNN(model).train(xTr, yTr, epoch, batch_size, verbose)
    return model, history

def test_nn(model, verbose):
    xTe = np.load('./xTe.npy')
    y_pred = RunNN(model).predict(xTe, verbose)
    yTe_pred_label = np.argmax(y_pred, axis = -1)
    print(yTe_pred_label)
    return y_pred

if __name__ == "__main__":
    # BUILD NEURAL NETWORK
    print('BUILDING CUSTOMED NERUAL NETWORK...')
    layer0 = 256
    dropout0 = 0.2
    layer1 = 128
    dropout1 = 0.2
    model = build_rand_nn(layer0, dropout0, layer1, dropout1)
    
    # RUN NEURAL NETWORK
    print('RUNNING NERUAL NETWORK...')
    epoch = 20
    batch_size = 128
    verbose = 1
    model, history = run_nn(model, epoch, batch_size, verbose)

    # # RUN SAVED TRAINED MODEL
    # print('CONTINUING TO RUN NERUAL NETWORK...')
    # epoch = 20
    # batch_size = 128
    # verbose = 1
    # model = load_model('model.h2')
    # model, history = run_nn(model, epoch, batch_size, verbose)

    # # PREDICT IN TEST DATASET
    # print('TEST NERUAL NETWORK...')
    # verbose = 1
    # model = load_model('model.h2')
    # y_pred = test_nn(model, verbose)