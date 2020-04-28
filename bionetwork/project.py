from networkdata import DataManager, WeightedNetwork
from algorithm import Predictor
from evaluation import Evaluator, DREAM4S100Evaluator
import numpy as np


class Manager:
    def __init__(self, laboratory_path: str, network: str, save_predictions: bool = True):
        self._save_predictions = save_predictions
        self._data_manager = DataManager(laboratory_path, network)
        self._evaluator = DREAM4S100Evaluator(network)

    # @property
    # def save_predictions(self):
    #     return self._save_predictions
    #
    # @save_predictions.setter
    # def save_predictions(self, value: bool):
    #     self._save_predictions = value

    def generate_predictions(self, param):
        # It must be 'time_lagged' for most usecases, alternative is 'dynamic' nd is used for dynGENIE3 algorithm
        predictor = Predictor(timeseries_method='time_lagged')
        # List of data sets
        data_list = [self._data_manager.get_multifactorials()[0], self._data_manager.get_timeseries_sets()[0]]
        predictor.fit(data_list, param=param)
        predictor.network.distribute_evenly()
        self._data_manager.prediction_levels.append(predictor.network)

        goldstandard = self._data_manager.goldstandard.data
        prediction = self._data_manager.prediction.data
        self._evaluator.fit(goldstandard, prediction)

    def generate_prediction(self):
        predictor = Predictor(timeseries_method='time_lagged')


    def flush_predictions(self):
        self._data_manager.flush()

    @property
    def score(self):
        return self._evaluator.auroc, \
               self._evaluator.aupr, \
               self._evaluator.auroc_p_value, \
               self._evaluator.aupr_p_value, \
               self._evaluator.score
        # TODO: This is not the correct score, please replace the valid score with this one

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._save_predictions:
            self.flush_predictions()
