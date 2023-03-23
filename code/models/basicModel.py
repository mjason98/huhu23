from code.utils import colorify
from sklearn.metrics import classification_report, mean_squared_error

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