from code.utils import colorify
from sklearn.metrics import classification_report, mean_squared_error
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from imblearn.over_sampling import SMOTE

import torch
import pandas as pd
import numpy as np
from code.parameters import PARAMS

from random import choices

def classifier_report(orig, pred, task):
    metrics = classification_report(orig,
                                    pred,
                                    target_names=[f'No {task}', task],
                                    digits=4,
                                    zero_division=1)
    print("# Metrics")
    print(metrics)

def regresor_report(orig, pred, task):
    metrics = mean_squared_error(orig, pred)
    print ('# Metrics in', task)
    print ('  MSE:', metrics, '\n')

class basicModel:
    def __init__(self, name:str):
        self.classifier = None 
        self.vectorizer = None 
        self.model_name = name
        self.report_function = 0
    
    def getBalanceForArrays(self):
        if self.report_function == 0:
            return SMOTE()
        else:
            raise BaseException("currently there is not a package to balance continous data")
    
    def report(self, orig, pred, task):
        if self.report_function == 0:
            classifier_report(orig, pred, task)
        else:
            regresor_report(orig, pred, task)

    def fit(self):
        print ('# Start training phase')
    
    def predict(self):
        print ('# Start prediction phase')
    
    def save(self):
        print('# Saved model:', colorify(self.model_name))
    
    def load(self):
        print('# Loading model:', colorify(self.model_name))

class basicDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.x_name = 'tweet'
        self.id_name = 'index'
        self.y_name = 'humor'

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ids = int(self.data_frame.loc[idx, self.id_name])
        sent = self.data_frame.loc[idx, self.x_name]

        try:
            target = int(self.data_frame.loc[idx, self.y_name])
        except:
            target = 0

        sample = {'x': sent, 'y': target, 'id':ids}
        return sample

def getWeights(csv_path:str):
    data = pd.read_csv(csv_path)
    target_name = PARAMS["DATA_TARGET_COLUMN_NAME"]

    positive = len(data.query(target_name + "==1"))
    negative = len(data.query(target_name + "==0"))

    wc = [negative, positive]

    weights = [ 1/wc[ int( data.loc[i, target_name] ) ] for i in range(len(data)) ]

    return weights

def getWeightsNP(y:np.ndarray):
    pw = 1. / (y == 1).sum()
    pn = 1. / (y == 0).sum()
    
    weights = y * pw + (1 - y) * pn 
    
    return weights.tolist()


def makeDataSet(csv_path:str, shuffle=False, balance=False):
    data = basicDataset(csv_path)
    batch = PARAMS['batch']

    sampler = None
    if balance:
        sample_weight = getWeights(csv_path)
        sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(data), replacement=True)
        shuffle = None

    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=PARAMS['workers'], drop_last=False, sampler=sampler)
    return data, loader

class npsDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray|None):
        assert X.shape[0] == y.shape[0]

        self.X = X
        self.y = y
        
        self.parity = np.array(choices([0,1], k=X.shape[0]), dtype=np.int8)
        self.pairs  = np.array([0] * X.shape[0], dtype=np.int32)

        self.auxiliarInx = np.array(range(X.shape[0]), dtype=np.int32)
        
        self.calculatePairs()
    
    def _chose(self, idx, pm, nm):
        if self.parity[idx] == 0:
            return np.random.choice(self.auxiliarInx[nm])
        else:
            return np.random.choice(self.auxiliarInx[pm])
        
    def calculatePairs(self):
        self.parity = self.parity ^ 1

        positive_mask = (self.y == 1)
        negative_mask = (self.y == 0)

        pairs_list = list(map(lambda x: self._chose(x, positive_mask, negative_mask), range(self.X.shape[0])))

        self.pairs = np.array(pairs_list, dtype=np.int32)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vect = torch.from_numpy(np.concatenate([self.X[idx], self.X[ self.pairs[idx] ] ]))

        # print (vect.shape)

        target = 0
        if self.y is not None:
            target = 0 if self.y[idx] == self.y[ self.pairs[idx] ] else 1

        sample = {'x': vect, 'y': target}
        return sample

def makeNPSDataset(X:np.ndarray, y:np.ndarray|None=None, shuffle=False, balance=False):
    data = npsDataset(X, y)
    batch = PARAMS['batch']

    sampler = None
    if balance:
        sample_weight = getWeightsNP(y)
        
        sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(data), replacement=True)
        shuffle = None

    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=PARAMS['workers'], drop_last=False, sampler=sampler)
    return data, loader

def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")