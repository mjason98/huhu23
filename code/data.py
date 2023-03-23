# data preprocesing funtions here

from random import sample
import pandas as pd, os
from code.parameters import PARAMS
from code.utils import colorify

def load_data(data_path: str):
    data = pd.read_csv(data_path)
    return data


def save_data(data: pd.Series):
    file_path_train = PARAMS["data_train"]
    file_path_test = PARAMS["data_test"]

    if os.path.isfile(file_path_train) and os.path.isfile(file_path_test):
        return file_path_train, file_path_test

    target_name = PARAMS["DATA_TARGET_COLUMN_NAME"]
    percent = 1.0 - PARAMS["data_percent"]

    data_true = data.query(target_name + "==1")
    data_false = data.query(target_name + "==0")

    train_true_ids = sample(data_true.index.tolist(), int(len(data_true) * percent))
    train_false_ids = sample(data_false.index.tolist(), int(len(data_false) * percent))

    data_train = pd.concat(
        [data_true.loc[train_true_ids], data_false.loc[train_false_ids]], axis=0
    )
    data_train.to_csv(file_path_train, index=False)

    del data_train

    data_true = data_true.drop(train_true_ids)
    data_false = data_false.drop(train_false_ids)

    data_test = pd.concat([data_true, data_false], axis=0)
    data_test.to_csv(file_path_test, index=False)

    del data_test

    print ('# Created datafiles:', colorify(file_path_test), colorify(file_path_train))

    return file_path_train, file_path_test

# ----------------------------------------------------------------

DATA_PIPELINE = [load_data, save_data]

def processData():
    curr_value = PARAMS["DATA_PATH"]

    for f in DATA_PIPELINE:
        curr_value = f(curr_value)