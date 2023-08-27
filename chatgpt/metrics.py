import pandas as pd
from sklearn.metrics import classification_report

# pred = pd.read_csv('../huhu23/data/pred_humor_ensb6.csv')
pred = pd.read_csv("pred_pi.csv")

# pred = pred.rename(columns={
#    'tweet_id': 'index',
#    'humour': 'humor'
#    })


orig = pd.read_csv('test_gold.csv')

orig = orig.merge(pred, on='index', how='left')

task = 'humor'

metrics = classification_report(
        orig['humor_x'],
        orig['humor_y'],
        target_names=[f'No {task}', task],
        digits=4,
        zero_division=1)

print('# Metrics')
print(metrics)
