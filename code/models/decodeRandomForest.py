from code.models.basicModel import basicModel
from code.parameters import PARAMS
from code.models.transformerOnlyEncoder import makeEncoding, encodePredictData

import numpy as np
import pandas as pd
import joblib, os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class rfBasedOnEncoder(basicModel):

    def __init__(self, name: str):
        super().__init__(name)
        self.classifier = RandomForestClassifier()

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

        train, test = self._loadData(task)

        vecs_train =  train["x"]
        y_train = train["y"]

        if PARAMS["balance"]:
            print ('  Balance SMOTE applied')
            oversample = self.getBalanceForArrays()
            vecs_train, y_train = oversample.fit_resample(vecs_train, y_train)

        self.classifier.fit(vecs_train, y_train)
        pred = self.classifier.predict(test["x"])

        self.report(test["y"], pred, task)

    def predict(self):
        super().predict()

        pred_vecs = encodePredictData()
        pred_vecs = np.load(pred_vecs)

        pred = self.classifier.predict(pred_vecs)

        return pred.tolist()

    def save(self):
        super().save()
        joblib.dump(
            self.classifier,
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_rf.joblib'))

    def load(self):
        super().load()
        self.classifier = joblib.load(
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_rf.joblib'))

def createRFEModel(name: str) -> rfBasedOnEncoder:
    model = rfBasedOnEncoder(name)
    return model


def createRFERModel(name: str) -> rfBasedOnEncoder:
    model = rfBasedOnEncoder(name)
    model.classifier = RandomForestRegressor()
    model.report_function = 1

    return model