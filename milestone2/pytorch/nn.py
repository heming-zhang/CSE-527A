import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nn_dataset import NNTrain

class NNLoad():
    def __init__(self, batchsize, worker):
        self.batchsize = batchsize
        self.worker = worker

    # Design a mini-batch gradient descent
    def load_data(self):
        batchsize = self.batchsize
        worker = self.worker
        train_dataset = NNTrain()
        train_loader = DataLoader(dataset = train_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        return train_loader

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.l1 = torch.nn.Linear(25, 1024)
        self.l2 = torch.nn.Linear(1024, 1024)
        self.l3 = torch.nn.Linear(1024, 1024)
        self.l4 = torch.nn.Linear(1024, 7)
    
    # Use ReLU as activation function
    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        out3 = F.relu(self.l3(out2))
        y_pred = self.l4(out3)
        return y_pred

class NNParameter():
    def __init__(self, 
                learning_rate, 
                momentum,
                cuda):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cuda = cuda
    
    def nn_function(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        cuda = self.cuda
        if cuda:
            model = NN().cuda()
        else:
            model = NN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                    lr = learning_rate, 
                    momentum = momentum)
        return model, criterion, optimizer

class RunNN():
    def __init__(self, model, criterion, optimizer, 
                train_loader, cuda):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.cuda = cuda

    def train_nn(self):
        model = self.model
        model.train()
        criterion = self.criterion
        optimizer = self.optimizer
        train_loader = self.train_loader
        cuda = self.cuda
        # Train data in certain epoch
        train_correct = 0
        for i, data in enumerate(train_loader):
            train_input, train_label = data
            train_input = np.array(train_input)
            train_label = np.array(train_label)
            # Wrap them in Variable
            train_input = Variable(torch.Tensor(train_input), requires_grad = False)
            train_label = Variable(torch.LongTensor(train_label), requires_grad = False)
            if cuda:
                train_input = train_input.cuda()
                train_label = train_label.cuda()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(train_input)
            train_label = train_label.squeeze()
            # Compute the train_loss with Cross-Entropy loss
            train_loss = criterion(y_pred, train_label)
            # Clear gradients of all optimized class
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # Compute accuracy rate
            _, train_pred = torch.max(y_pred.data, 1)
            train_correct += (train_pred == train_label).sum()
        train_accuracy = float(train_correct) / len(train_loader.dataset)
        return float(train_loss), train_accuracy
    