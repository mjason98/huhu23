from code.models.basicModel import basicModel
from code.parameters import PARAMS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
import pandas as pd

import joblib, os


# from textatistic import Textatistic
class rfModel(basicModel):

    def __init__(self, name) -> None:
        super().__init__(name)
        self.classifier = RandomForestClassifier()
        self.vectorizer = TfidfVectorizer(min_df=0,
                                          max_df=0.8,
                                          sublinear_tf=True,
                                          analyzer='char',
                                          ngram_range=(3, 3),
                                          use_idf=True)
        self.report_function = 0

    def _loadData(self):
        data_train = pd.read_csv(PARAMS['data_train'])
        data_test = pd.read_csv(PARAMS['data_test'])

        return data_train, data_test

    def _loadPredictData(self):
        data_val = pd.read_csv(PARAMS['DATA_PREDICTION_PATH'])

        return data_val

    def fit(self):
        super().fit()

        textc = PARAMS["DATA_TEXT_COLUMN_NAME"]
        task = PARAMS["DATA_TARGET_COLUMN_NAME"]

        train, test = self._loadData()

        vecs_train = self.vectorizer.fit_transform(train[textc]).toarray()
        vecs_test = self.vectorizer.transform(test[textc]).toarray()

        y_train = train[task]

        if PARAMS["balance"]:
            print ('  Balance SMOTE applied')
            oversample = SMOTE()
            vecs_train, y_train = oversample.fit_resample(vecs_train, y_train)

        self.classifier.fit(vecs_train, y_train)
        pred = self.classifier.predict(vecs_test)

        self.report(test[task], pred, task)

    def predict(self) -> list:
        super().predict()

        textc = PARAMS["DATA_TEXT_COLUMN_NAME"]

        data = self._loadPredictData()
        vecs = self.vectorizer.transform(data[textc]).toarray()
        pred = self.classifier.predict(vecs)

        return pred.tolist()

    def save(self):
        super().save()
        joblib.dump(
            self.vectorizer,
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_t2v.joblib'))
        joblib.dump(
            self.classifier,
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_rf.joblib'))

    def load(self):
        super().load()
        self.vectorizer = joblib.load(
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_t2v.joblib'))
        self.classifier = joblib.load(
            os.path.join(PARAMS['MODEL_FOLDER'],
                         self.model_name + '_rf.joblib'))


def createRFModel(name: str) -> rfModel:
    model = rfModel(name)
    return model


def createRFRModel(name: str) -> rfModel:
    model = rfModel(name)
    model.classifier = RandomForestRegressor()
    model.report_function = 1

    return model
