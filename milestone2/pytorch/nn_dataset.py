import numpy as np
from torch.utils.data import Dataset

# CUSTOM YOURSELF CLASS DATASET FOR DATALOADER
class NNTrain(Dataset):
    def __init__(self):
        xTr = np.load("./xTr.npy")
        yTr = np.load("./yTr.npy")
        self.x = np.array(xTr)
        self.y = np.array(yTr)
        self.len = np.shape(self.x)[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len