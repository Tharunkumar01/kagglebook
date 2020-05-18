import numpy as np
import pandas as pd

from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350

    params_nn = {
        'layers': 3,
        # Setting so this sample code executes quickly
        'nb_epoch': 5,  # 1000
        'patience': 10,
        'dropout': 0.5,
        'units': 512,
    }

    # Specify features
    features = [f'feat_{i}' for i in range(1, 94)]

    # Train and predict using xgboost
    runner = Runner('xgb1', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb1')

    # Train and predict using neural network
    runner = Runner('nn1', ModelNN, features, params_nn)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('nn1')

    '''
    # (For reference) Train and predict using xgboost using all training data
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''
