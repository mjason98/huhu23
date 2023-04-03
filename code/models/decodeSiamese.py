from code.models.basicModel import basicModel, getDevice, makeNPSDataset
from code.parameters import PARAMS
from code.models.transformerOnlyEncoder import makeEncoding, encodePredictData

import numpy as np
import pandas as pd
import os

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive


class seameseModel(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential( nn.Linear(input_size, hidden_size), nn.Tanh(), 
                                  nn.Linear(hidden_size, hidden_size//2), nn.Tanh() )
        # self.distanse = nn.CosineSimilarity()
        self.criterion = ContrastiveLoss()

    def forward(self, X):
        """ X shape (bach, input_size *2) """
        
        x1,x2 = X[:,:self.input_size], X[:,self.input_size:]

        x1, x2 = self.net(x1), self.net(x2)

        return x1, x2

    def makeOptimizer(self, lr=5e-5, decay=2e-5, algorithm='adam'):
        pars = [{'params':self.net.parameters()}]

        if algorithm == 'adam':
            return torch.optim.Adam(pars, lr=lr, weight_decay=decay)
        elif algorithm == 'rms':
            return torch.optim.RMSprop(pars, lr=lr, weight_decay=decay)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)



class siameseOnEncoder(basicModel):

    def __init__(self, name: str):
        super().__init__(name)

    def _loadData(self, task):
        train_name, test_name = makeEncoding()
        train = np.load(train_name)
        test = np.load(test_name)

        data_train = pd.read_csv(PARAMS['data_train'])[task].to_numpy()
        data_test = pd.read_csv(PARAMS['data_test'])[task].to_numpy()

        return {
            "x": train,
            "y": data_train
        }, {
            "x": test,
            "y": data_test
        }

    def fit(self):
        super().fit()
        task = PARAMS["DATA_TARGET_COLUMN_NAME"]
        epochs = PARAMS["epochs"]
        model_path = os.path.join(PARAMS['MODEL_FOLDER'],  f'{self.model_name}.pt')

        train, test = self._loadData(task)
        vec_size = train['x'].shape[0]

        # if PARAMS["balance"]:
        #     print ('  Balance SMOTE applied')
        #     oversample = self.getBalanceForArrays()
        #     vecs_train, y_train = oversample.fit_resample(vecs_train, y_train)

        device = getDevice()
        self.classifier = seameseModel(vec_size, vec_size//2)
        self.classifier.eval()
        self.classifier.to(device=device)

        optim = self.classifier.makeOptimizer(lr=PARAMS["lr"], algorithm=PARAMS["optim"])

        for e in range(1, epochs+1):
            print (f"# Start epoch {e}/{epochs}")

            for dname in ['train', 'val']:
                
                dataref = None
                if dname == 'train':
                    self.classifier.train()
                    dataref = train
                else:
                    self.classifier.eval()
                    dataref = test

                _, data = makeNPSDataset(dataref['x'], dataref['y'], dname == 'train', False)
                iter = tqdm(data, f'{dname}')

                total_loss, dl, best_score = 0., 0, 0.
            
                for batch in iter:
                    optim.zero_grad()

                    with torch.set_grad_enabled(dname == 'train'):
                        X = batch['x'].to(device=device)
                        y = batch['y'].to(device=device)

                        x1, x2 = self.classifier(X)

                        loss = self.classifier.criterion(x1, x2, y)
                        
                        if dname == 'train':
                            loss.backward()
                            optim.step()

                        total_loss += loss.item() * y.shape[0]
                        # total_acc += (y1 == y_hat.argmax(dim=-1).flatten()).sum().item()
                        dl += y.shape[0]
                
                if best_score > total_loss and dname == 'val':
                    best_score = total_loss
                    self.classifier.save(model_path)

                print('# {} epoch {} Loss {:.3} Acc {:.3}{}'.format(dname, e, total_loss/dl, '*' if total_loss == best_score else ' '))

    def predict(self):
        super().predict()

        # pred_vecs = encodePredictData()
        # pred_vecs = np.load(pred_vecs)

        # pred = self.classifier.predict(pred_vecs)

        # return pred.tolist()

    def save(self):
        super().save()
        model_path = os.path.join(PARAMS['MODEL_FOLDER'],  f'{self.model_name}.pt')
        self.classifier.save(model_path)

    def load(self):
        super().load()
        model_path = os.path.join(PARAMS['MODEL_FOLDER'],  f'{self.model_name}.pt')
        self.classifier.load(model_path)

def createSEModel(name: str) -> siameseOnEncoder:
    model = siameseOnEncoder(name)
    return model


# def createRFERModel(name: str) -> rfBasedOnEncoder:
#     model = rfBasedOnEncoder(name)
#     model.classifier = RandomForestRegressor()
#     model.report_function = 1

#     return model