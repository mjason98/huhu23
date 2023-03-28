from code.models.basicModel import basicModel, makeDataSet, getDevice
from code.parameters import PARAMS

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd, os
from tqdm import tqdm
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encodeDF(df, name=""):
    batch = PARAMS['batch']
    
    ctn = int(len(df)/batch)

def makeEncoding():
    tnames = []
    mxlenght = PARAMS['max_length']

    device = getDevice()
    tokenizer = AutoTokenizer.from_pretrained('hackathon-pln-es/paraphrase-spanish-distilroberta')
    model = AutoModel.from_pretrained('hackathon-pln-es/paraphrase-spanish-distilroberta')
    model.eval()
    model.to(device=device)

    for dname in ['train', 'test']:
        print ('# Encoding', dname)
        original_name = PARAMS[f"data_{dname}"]
        temporal_name = os.path.join(PARAMS['DATA_FOLDER'], f'.{dname}_enc_hkpln.npy')
        tnames.append(temporal_name)

        if os.path.isfile(temporal_name):
            continue
        
        encods = []

        _, data = makeDataSet(original_name, False, False)
        iter = tqdm(data, f'encoding {dname}')

        with torch.no_grad():
            for batch in iter:
                # Tokenize sentences
                encoded_input = tokenizer(batch['x'], padding=True, truncation=True, return_tensors='pt', max_length=mxlenght).to(device=device)
                model_output = model(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()

                encods.append(sentence_embeddings)
                
        encods = np.concatenate(encods, axis=0)
        
        np.save(temporal_name, encods)

    return tnames[0], tnames[1]


makeEncoding()