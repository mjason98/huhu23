import argparse, os
from code.utils import colorify

PARAMS = {
    # general parameters
    "seed": 1234567,
    # train parameters
    "lr": 5e-5,
    "optim": "rms",
    "workers": 2,
    "batch": 12,
    "epochs": 100,
    "ted_epochs": 12,
    "random_pair_selection_rate": 3,
    # model values
    "TRANS_NAME": "bert-base-uncased",
    "MODEL_FOLDER": "pts",
    "model_type": "ted",
    "max_length": 300,
    "transformer_name": "Manauu17/enhanced_roberta_sentiments_es",
    "balance": False,
    "n_references": 100,
    # dataset values
    "DATA_FOLDER": "data",
    "DATA_PATH": "data/train.csv",
    "DATA_PREDICTION_PATH": "data/test.csv",
    "DATA_TEXT_COLUMN_NAME": "tweet",
    "DATA_TARGET_COLUMN_NAME": "humor", # in ['humor', 'mean_prejudice']
    "data_train": "data/train_tmp.csv",
    "data_test": "data/test_tmp.csv",
    "cat_vector": "data/cat_vector.bin",
    "data_percent": 0.05,
    # ...
}


def check_params(arg=None):
    global PARAMS

    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    parse = argparse.ArgumentParser(description="HUHU model")

    parse.add_argument(
        "--datafolder",
        dest="datafolder",
        help="Data folder path",
        required=False,
        default="data",
    )

    parse.add_argument(
        "--lr", dest="lr", help="Learning rate value", required=False, default=0.001
    )

    parse.add_argument(
        "--modeltype", dest="modeltype", help="Model type", required=False, default="rf", choices=['rf', 'rfr', 'rfe', 'rfer', 'se', 'ted']
    )

    parse.add_argument(
        "--target", dest="target", help="Column target", required=False, default="humor", choices=['humor', 'mean_prejudice']
    )

    parse.add_argument('--balance', help='Apply balance tecnique to the unbalance data',
					   required=False, action='store_true', default=False)

    returns = parse.parse_args(arg)
    new_params = {
        "DATA_FOLDER": returns.datafolder,
        "lr": returns.lr,
        "DATA_TARGET_COLUMN_NAME": returns.target,
        "model_type": returns.modeltype,
        "balance": returns.balance,
    }

    PARAMS.update(new_params)

    # forder preparation
    if not os.path.isdir(PARAMS["DATA_FOLDER"]):
        os.mkdir(PARAMS["DATA_FOLDER"])
        print(
            "# Created folder",
            colorify(PARAMS["DATA_FOLDER"]),
            "please copy the data files there",
        )

    if not os.path.isdir(PARAMS["MODEL_FOLDER"]):
        os.mkdir(PARAMS["MODEL_FOLDER"])
        print(
            "# Created folder",
            colorify(PARAMS["MODEL_FOLDER"]),
            "to save the models weights",
        )

    return 1
