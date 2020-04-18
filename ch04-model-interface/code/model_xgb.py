import os

import numpy as np
import pandas as pd
import xgboost as xgb

from model import Model
from util import Util


class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # Set the data
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # Set the hyperparameters
        params = dict(self.params)
        num_round = params.pop('num_round')

        # Train
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # To prevent loss of best_ntree_limit model, save model using pickle
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
