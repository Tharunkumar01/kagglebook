import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union

from util import Logger, Util

logger = Logger()


class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
        """Constructor

        :param run_name: Run name
        :param model_cls: Model class
        :param features: List of features
        :param params: Hyperparameters
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """Specify cross validation, train then calculate score

        In addition to calling from other methods, this is also used itself for checks and to adjust parameters

        :param i_fold: fold number (when everything use 'all')
        :return: Tuple containing (model instance, record index, predictions, validation score)
        """
        # Load training data
        validation = i_fold != 'all'
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # Set training and validation data
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # Train model
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # Make predictions using validation data and calculate score
            va_pred = model.predict(va_x)
            score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)

            # Return model, index, predictions and score
            return model, va_idx, va_pred, score
        else:
            # Train using all training data
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # Return model
            return model, None, None, None

    def run_train_cv(self) -> None:
        """Training and evaluation using cross validation

        Train, score, save each fold model, output score to log
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # Train on each fold
        for i_fold in range(self.n_fold):
            # Train
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # Save model
            model.save_model()

            # Retain results
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # Gather results for all folds
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # Save predictions
        Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        # Save scores
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """Take average of results from models trained on each fold and make predictions for test data

        Necessary to run_train_cv beforehand
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # Train on each fold
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # Output mean value of predictions
        pred_avg = np.mean(preds, axis=0)

        # Save predictions
        Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """Train using all training data and save model"""
        logger.info(f'{self.run_name} - start training all')

        # Train on all training data
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """Make predictions using model trained with all training data

        Necessary to run_train_all beforehand
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # Make predictions using model trained on all training data
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # Save predictions
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """Specify cross validation fold and create model

        :param i_fold: fold number
        :return: model instance
        """
        # Create model from run name, fold and model class
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """Load features of training data

        :return: Training data features
        """
        # Load training data
        # Note you must modify this method if you want to do anything more than just extraction by column name
        # As it is inefficient to load train.csv every time, use this method appropriately for the data (same applies for other methods also) 
        return pd.read_csv('../input/train.csv')[self.features]

    def load_y_train(self) -> pd.Series:
        """Load target values of training data

        :return: Training data target values
        """
        # Load target values
        train_y = pd.read_csv('../input/train.csv')['target']
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """Load features of test data

        :return: Test data features
        """
        return pd.read_csv('../input/test.csv')[self.features]

    def load_index_fold(self, i_fold: int) -> np.array:
        """Specify cross validation fold and return corresponding record index

        :param i_fold: Fold number
        :return: Record index of corresponding fold
        """
        # Return index that separates training and validation data
        # Here a random number is created every time, so there is also a method to save it to a file
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]
