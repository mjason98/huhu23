from code.models.basicModel import basicModel
from code.parameters import PARAMS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

import joblib, os
# from textatistic import Textatistic

class rfModel(basicModel):
    def __init__(self, name) -> None:
        super().__init__(name) 
        self.classifier = RandomForestClassifier()
        self.vectorizer = TfidfVectorizer(min_df = 0,  max_df = 0.8, 
                                sublinear_tf = True, analyzer = 'char',
                                ngram_range=(3, 3),  use_idf = True)
    
    def _loadData(self):
        data_train = pd.read_csv(PARAMS['data_train'])
        data_test = pd.read_csv(PARAMS['data_test'])

        return data_train, data_test

    def fit(self):
        super().fit()

        textc = PARAMS["DATA_TEXT_COLUMN_NAME"]
        task = PARAMS["DATA_TARGET_COLUMN_NAME"]

        train, test = self._loadData()

        vecs_train = self.vectorizer.fit_transform(train).toarray()
        vecs_test  = self.vectorizer.transform(test[textc]).toarray()

        self.classifier.fit(vecs_train, train[task])
        pred = self.classifier.predict(vecs_test)

        metrics = classification_report(test[task], pred, target_names=[f'No {task}', task],  digits=4, zero_division=1)        
        
        print("# Metrics")
        print(metrics)
    
    def predict(self, X):
        super().predict()
    
    def save(self):
        super().save()
        joblib.dump(self.vectorizer, os.path.join(PARAMS['MODEL_FOLDER'], self.model_name+'_t2v.joblib'))
        joblib.dump(self.classifier, os.path.join(PARAMS['MODEL_FOLDER'], self.model_name+'_rf.joblib'))
    
    def load(self):
        super().load()
        self.vectorizer = joblib.load(os.path.join(PARAMS['MODEL_FOLDER'], self.model_name+'_t2v.joblib'))
        self.classifier = joblib.load(os.path.join(PARAMS['MODEL_FOLDER'], self.model_name+'_rf.joblib'))

def createRFModel(name:str) -> rfModel:
    model = rfModel(name)
    return model