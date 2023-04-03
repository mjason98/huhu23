from code.parameters import PARAMS
from code.models.randomForest import createRFModel, createRFRModel
from code.models.decodeRandomForest import createRFEModel, createRFERModel
from code.models.decodeSiamese import createSEModel
from code.models.basicModel import basicModel

import numpy as np
import random, os
import pandas as pd
import torch


def setSeed():
    my_seed = PARAMS['seed']
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    random.seed(my_seed)


def _typeAndName():
    model_type = PARAMS['model_type']
    model_name = f"{ model_type }_{ PARAMS['DATA_TARGET_COLUMN_NAME'] }"

    return model_type, model_name


def _createModel():
    model_type, model_name = _typeAndName()

    model: basicModel = None
    if model_type == 'rf':
        model = createRFModel(model_name)
    elif model_type == 'rfr':
        model = createRFRModel(model_name)
    elif model_type == 'rfe':
        model = createRFEModel(model_name)
    elif model_type == 'rfer':
        model = createRFERModel(model_name)
    elif model_type == 'se':
        model = createSEModel(model_name)

    return model


def makeModel():
    model = _createModel()

    # training
    model.fit()
    model.save()


def predict():
    model = _createModel()
    model.load()
    preds = model.predict()

    tname = PARAMS['DATA_TARGET_COLUMN_NAME']
    mname = PARAMS['model_type']

    # save predictions
    data = pd.read_csv(PARAMS['DATA_PREDICTION_PATH'])
    cols = list(data.columns) + [tname]

    save_name = f"pred_{ tname }_{ mname }.csv"
    save_name = os.path.join(PARAMS["DATA_FOLDER"], save_name)

    data = pd.concat([data, pd.Series(preds)], axis=1)

    data.to_csv(save_name, header=cols, index=False)
