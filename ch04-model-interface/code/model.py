import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """Constructor

        :param run_fold_name: concatenation of run name and fold number
        :param params: hyperparameters
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None) -> None:
        """Perform model training and save trained model

        :param tr_x: Training data features
        :param tr_y: Training data target values
        :param va_x: Validation data features
        :param va_y: Validation data target values
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """Return predictions from trained model

        :param te_x: Validation data or test data features
        :return: Predictions
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """Save the model """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model """
        pass
