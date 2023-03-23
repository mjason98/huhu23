from code.utils import colorify

class basicModel:
    def __init__(self, name:str):
        self.classifier = None 
        self.vectorizer = None 
        self.model_name = name
    
    def fit(self):
        print ('# Start training phase')
    
    def predict(self):
        print ('# Start prediction phase')
    
    def save(self):
        print('# Saved model:', colorify(self.model_name))
    
    def load(self):
        print('# Loading model:', colorify(self.model_name))