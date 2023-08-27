import pandas as pd

DEFAULT_VALUE = 0


def str_to_num(val):
    if len(val) < 1:
        return DEFAULT_VALUE
    if val[0] == 'S' or val[0] == 's':
        return 1
    else:
        return 0


data = pd.read_csv('prediction_gpt_pi_test.csv')

data['prediction'] = data['prediction'].apply(str_to_num)

data.to_csv('pred_pi.csv', index=None, header=['index', 'humor'])
