import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

class KerasNN():
    def __init__(self):
        pass

    # NEW RAND NETWORK
    def keras_nn(self, layer0, dropout0, layer1, dropout1):
        # INITIALIZE THE CONSTRUCTOR
        model = Sequential()
        # ADD AN INPUT LAYER
        model.add(Dense(layer0, activation='relu'))
        model.add(Dropout(dropout0))
        # ADD AN HIDDEN LAYER
        model.add(Dense(layer1, activation='relu'))
        model.add(Dropout(dropout1))
        # ADD FINAL OUTPUT LAYER
        model.add(Dense(7, activation='softmax'))
        return model

class RunNN():
    def __init__(self, model):
        self.model = model

    def train(self, xTr, yTr, epoch, batch_size, verbose):
        model = self.model
        model.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        history = model.fit(xTr, yTr, epochs = epoch, validation_split = 1, batch_size = batch_size)
        model.save('model.h2')
        return model, history

    def predict(self, xTe, verbose):
        model = self.model
        y_pred = model.predict(xTe)
        return y_pred