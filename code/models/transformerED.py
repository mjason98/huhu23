from code.models.basicModel import basicModel, makeDataSet, getDevice
from code.parameters import PARAMS

from transformers import AutoTokenizer, AutoModel
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from code.models.basicModel import basicModel
from code.parameters import PARAMS
import numpy as np

import os

class trcModel(nn.Module):
    def __init__(self, transformer_name, hidden_size=768) -> None:
        super().__init__()
        self.net1 = AutoModel.from_pretrained(transformer_name)
        self.net2 = nn.Linear(hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, encoded_input):
        model_output = self.net1(**encoded_input)
        sentence_embeddings = model_output[0][:,0]
        return self.net2(sentence_embeddings)

    def makeOptimizer(self, lr=5e-5, lr_factor=9/10, decay=2e-5, algorithm='adam'):
        pars = [{'params':self.net2.parameters()}]

        for l in self.net1.encoder.layer:
            lr *= lr_factor
            D = {'params':l.parameters(), 'lr':lr}
            pars.append(D)
        try:
            lr *= lr_factor
            D = {'params':self.net1.pooler.parameters(), 'lr':lr}
            pars.append(D)
        except:
            print('#Warning: Pooler layer not found')

        if algorithm == 'adam':
            return torch.optim.Adam(pars, lr=lr, weight_decay=decay)
        elif algorithm == 'rms':
            return torch.optim.RMSprop(pars, lr=lr, weight_decay=decay)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

class tedModel(basicModel):

    def __init__(self, name) -> None:
        super().__init__(name)
        
        transformer_name = PARAMS["transformer_name"]

        self.mxlenght = PARAMS['max_length']
        self.classifier = trcModel(transformer_name)
        self.vectorizer = AutoTokenizer.from_pretrained(transformer_name)
        self.report_function = 0

        self.device = getDevice()
        self.classifier.to(device=self.device)

    def fit(self):
        super().fit()

        epochs = PARAMS["ted_epochs"]
        optim = self.classifier.makeOptimizer(lr=PARAMS["lr"], algorithm=PARAMS["optim"])

        iter_labels_and_data = [
            ('train', makeDataSet(PARAMS[f"data_train"], True, PARAMS["balance"])[1]),
            ('test', makeDataSet(PARAMS[f"data_test"], False, False)[1])
        ]

        best_acc = 0.

        for e in range(epochs):
            for dname, data in iter_labels_and_data:
                print (f'# {dname} epoch {e}')
                
                total_loss, total_acc, dl = 0., 0., 0        
                
                iterd = tqdm(data, f'{dname}')

                if dname == 'train':
                    self.classifier.train()
                    optim.zero_grad()
                else:
                    self.classifier.eval()
                
                with torch.set_grad_enabled(dname == 'train'):
                    for batch in iterd:
                        encoded_input = self.vectorizer(batch['x'], padding=True, truncation=True, return_tensors='pt', max_length=self.mxlenght).to(device=self.device)
                        y_hat = self.classifier(encoded_input)

                        y = batch['y'].to(device=self.device)

                        loss = self.classifier.criterion(y_hat, y)

                        if dname == 'train':
                            loss.backward()
                            optim.step()
                        
                        total_loss += loss.item() * y.shape[0]
                        total_acc += (y == y_hat.argmax(dim=-1).flatten()).sum().item()
                        dl += y.shape[0]
                
                if dname == 'test' and total_acc > best_acc:
                    best_acc = total_acc
                    self.save()
        
        self.load()

        self.custom_report(iter_labels_and_data[1][1])
    
    def custom_report(self, data):
        task = PARAMS["DATA_TARGET_COLUMN_NAME"]

        self.classifier.eval()
        iterd = tqdm(data, f'predictions')

        preds, ori = [], []

        with torch.no_grad():
            for batch in iterd:
                encoded_input = self.vectorizer(batch['x'], padding=True, truncation=True, return_tensors='pt', max_length=self.mxlenght).to(device=self.device)
                y_hat = self.classifier(encoded_input)
                y_hat = y_hat.argmax(dim=-1).squeeze()
                
                preds.append(y_hat.cpu().numpy())
                ori.append(batch['y'].squeeze().numpy())
        
        preds = np.concatenate(preds, axis=0)
        ori = np.concatenate(ori, axis=0)

        self.report(ori, preds, task)

    def predict(self) -> list:
        super().predict()

        dataPred = PARAMS["DATA_PREDICTION_PATH"]
        self.classifier.eval()

        _, data = makeDataSet(dataPred, False, False)
        iterd = tqdm(data, f'predictions')

        preds = []

        with torch.no_grad():
            for batch in iterd:
                encoded_input = self.vectorizer(batch['x'], padding=True, truncation=True, return_tensors='pt', max_length=self.mxlenght).to(device=self.device)
                y_hat = self.classifier(encoded_input)
                y_hat = y_hat.argmax(dim=-1).squeeze()
                
                preds.append(y_hat.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        return preds.tolist()

    def save(self):
        super().save()
        save_path = os.path.join(PARAMS['MODEL_FOLDER'], self.model_name + '.pt')
        self.classifier.save(save_path)        

    def load(self):
        super().load()
        save_path = os.path.join(PARAMS['MODEL_FOLDER'], self.model_name + '.pt')
        self.classifier.load(save_path)

def createTEDModel(name: str) -> tedModel:
    model = tedModel(name)
    return model

