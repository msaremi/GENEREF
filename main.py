## The main algorithm

import os
import numpy as np
from datetime import datetime as dt
from algorithm import Predictor
from itertools import permutations
from networkdata import DataManager, WeightedNetwork
from config import model, alpha_values, beta_values, max_level, skip_existing_preds
from config import datasets_path, predictions_path
import sys


def modulate(data: np.ndarray):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min + sys.float_info.min)


def get_regularization_matrix(w: WeightedNetwork):
    w = w.get_redistributed_network()
    w = w.data
    reg = 1 - (1 - w ** (2 ** alpha_value)) ** (2 ** beta_value)
    return reg


for network in model.networks:
    print(dt.now(), "Network", network)

    for dataset in model.datasets:
        print(dt.now(), "\t", "Dataset", dataset)
        dataset_path = os.path.join(datasets_path, dataset)
        prediction_path = os.path.join(predictions_path, dataset)
        data_manager = DataManager(dataset_path, network)

        with data_manager:
            num_experiments = len(data_manager.experiments)

            for i in range(1, min(max_level, num_experiments + 1)):
                first_level = i == 1
                second_level = i == 2
                keys = list(permutations(range(num_experiments), i))

                for key in keys:
                    print(dt.now(), "\t\t", "Key", key)
                    current_experiment_id = key[-1]
                    current_experiment = data_manager.experiments[current_experiment_id]
                    alphas = [0] if first_level else alpha_values
                    betas = [0] if first_level else beta_values

                    for alpha_value in alphas:
                        for beta_value in betas:
                            print(dt.now(), "\t\t\t", "Params", (alpha_value, beta_value))

                            if skip_existing_preds and (alpha_value, beta_value) + key in data_manager.predictions:
                                break

                            if first_level:
                                regularization = None
                            else:
                                # key_alpha_beta = (0, 0) if second_level else (alpha_value, beta_value)
                                # d = np.mean(
                                #     [data_manager.predictions[(key_alpha_beta if x > 1 else (0, 0)) + key[:x]].data for
                                #      x in range(1, len(key))], axis=0)
                                # regularization_network = WeightedNetwork(d)
                                regularization_network = data_manager.predictions[key_alpha_beta + key[:-1]]
                                regularization = get_regularization_matrix(regularization_network)

                            predictor = Predictor(
                                trunk_size=100,
                                n_trees=model.learner_params.n_trees,
                                max_features=model.learner_params.max_features,
                                callback=lambda j, n:
                                print('\r%s' % dt.now(), "\t\t\t\t", "Subproblem %d out of %d" % (j + 1, n), flush=True,
                                      end='')
                            )
                            predictor.fit(current_experiment, regularization)
                            prediction = predictor.network
                            prediction_data = prediction.data
                            prediction_data = modulate(prediction_data)
                            prediction = WeightedNetwork(prediction_data)
                            data_manager.predictions[(alpha_value, beta_value) + key] = prediction
                            print('\r', end='')

                    data_manager.predictions.flush()
