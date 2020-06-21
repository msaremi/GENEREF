# The main algorithm

import os
import numpy as np
from datetime import datetime as dt
from algorithm import Predictor
from itertools import permutations
from networkdata import DataManager, WeightedNetwork
from config import load as load_config
import sys


def modulate(data: np.ndarray):
    """The modulation function"""
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min + sys.float_info.min)


def get_regularization_matrix(w: WeightedNetwork):
    """Compute the regularization matrix based on the confidence matrix"""
    w = w.get_redistributed_network()
    w = w.data
    reg = 1 - (1 - w ** (2 ** alpha_value)) ** (2 ** beta_value)
    return reg


config = load_config(None if len(sys.argv) == 1 else sys.argv[1])

for network in config['model']['networks']:  # For DREAM4 there are 5 networks each with 100 genes
    print(dt.now(), "Network", network)

    for dataset in config['model']['datasets']:  # There are 10 dataset folders. In the most cases, 1 folder is enough
        print(dt.now(), "\t", "Dataset", dataset)
        dataset_path = os.path.join(config['datasets_path'], dataset)
        prediction_path = os.path.join(config['predictions_path'], dataset)
        data_manager = DataManager(dataset_path, network, preds_path=prediction_path)  # data_manager load the current dataset folder

        with data_manager:
            num_experiments = len(data_manager.experiments)

            # This algorithm sweeps all configurations of dataset in a breadth first manner:
            # First it runs the first iteration of GENEREF for all existing datasets
            # the it runs the second iteration of the algorithm for each of the confidence
            # matrices generated in the first iteration.

            # Make sure that number of iterations doesn't exceed max_level
            for i in range(1, min(config['max_level'], num_experiments + 1)):
                first_level = i == 1
                second_level = i == 2
                keys = list(permutations(range(num_experiments), i))

                #  keys keep track of the current path that GENEREF has followed in the breadth-first navigation
                for key in keys:
                    print(dt.now(), "\t\t", "Key", key)
                    current_experiment_id = key[-1]
                    current_experiment = data_manager.experiments[current_experiment_id]
                    alphas = [0] if first_level else config['alpha_log2_values']
                    betas = [0] if first_level else config['beta_log2_values']

                    #  The algorithm is run for all combinations of alphas and betas
                    for alpha_value in alphas:
                        for beta_value in betas:
                            print(dt.now(), "\t\t\t", "Params", (alpha_value, beta_value))

                            if config['skip_existing_preds'] \
                                    and (alpha_value, beta_value) + key in data_manager.predictions:
                                continue

                            if first_level:
                                regularization = None
                            else:
                                key_alpha_beta = (0, 0) if second_level else (alpha_value, beta_value)
                                # Load the regularization matrix computed in the previous level (2 lines)
                                regularization_network = data_manager.predictions[key_alpha_beta + key[:-1]]
                                regularization = get_regularization_matrix(regularization_network)

                            # Compute the next level confidence matrix
                            predictor = Predictor(
                                num_of_jobs=config['learner_params']['parallel_jobs'],
                                trunk_size=config['learner_params']['trunk_size'],
                                n_trees=config['learner_params']['n_trees'],
                                max_features=config['learner_params']['max_features'],
                                callback=lambda j, n:
                                print('\r%s' % dt.now(), "\t\t\t\t", "Subproblem %d out of %d" % (j + 1, n), flush=True,
                                      end='')
                            )
                            predictor.fit(current_experiment, regularization)
                            prediction = predictor.network
                            prediction_data = prediction.data
                            prediction_data = modulate(prediction_data)
                            prediction = WeightedNetwork(prediction_data)
                            # Save the predicted confidence matrix in the corresponding file
                            data_manager.predictions[(alpha_value, beta_value) + key] = prediction
                            print('\r', end='')

                            # Store the predictions in the file and remove it from the memory
                            data_manager.predictions.free_memory()


if __name__ != '__main__':
    raise PermissionError("You cannot import this file. Run it only from the terminal.")
