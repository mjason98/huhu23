from code.parameters import PARAMS
from code.models.randomForest import createRFModel
from code.models.basicModel import basicModel

import numpy as np
import random

def setSeed():
    my_seed = PARAMS['seed']
    np.random.seed(my_seed)
    random.seed(my_seed)

def makeModel():
    model_type = PARAMS['model_type']
    model_name = f"{ model_type }_{ PARAMS['DATA_TARGET_COLUMN_NAME'] }"

    model:basicModel = None
    if model_type == 'rf':
        model = createRFModel(model_name)

    model.fit()
    model.save()
