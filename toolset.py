"""
Use this file to generate the results based on the predictions
"""

import os
import numpy as np
from datetime import datetime as dt
from algorithm import Predictor
from itertools import permutations
from networkdata import DataManager, WeightedNetwork
from config import model, alpha_log2_values, beta_log2_values, max_level, \
    score_name, datasets_path, predictions_path, results_path
import pandas as pd
from glob import iglob


def _str_to_tuple(string):
    def _to_num(num_str: str):
        try:
            return int(num_str)
        except ValueError:
            return float(num_str)

    return tuple(map(_to_num, filter(None, string[1:-1].split(','))))


def generate_results():
    """produces results based on the predictions"""
    all_scores = {}

    for network in model.networks:
        print(dt.now(), "Network", network)
        evaluator = model.evaluator(network)
        all_scores[network] = {}

        for dataset in model.datasets:
            print(dt.now(), "\t", "Dataset", dataset)
            dataset_path = os.path.join(datasets_path, dataset)
            prediction_path = os.path.join(predictions_path, dataset)
            data_manager = DataManager(dataset_path, network, prediction_path)

            with data_manager:
                num_experiments = len(data_manager.experiments)

                for i in range(1, min(max_level, num_experiments + 1)):
                    first_level = i == 1
                    keys = list(permutations(range(num_experiments), i))

                    for key in keys:
                        print(dt.now(), "\t\t", "Key", key)
                        alphas = [0] if first_level else alpha_log2_values
                        betas = [0] if first_level else beta_log2_values

                        if key not in all_scores[network]:
                            columns = pd.MultiIndex.from_product((alphas, betas), names=['alpha', 'beta'])
                            all_scores[network][key] = pd.DataFrame(columns=columns)

                        all_scores[network][key].loc[dataset] = np.zeros((len(all_scores[network][key].columns),))

                        for alpha_value in alphas:
                            for beta_value in betas:
                                prediction = data_manager.predictions[(alpha_value, beta_value) + key]
                                evaluator.fit(data_manager.goldstandard.data, prediction.data)
                                all_scores[network][key][alpha_value, beta_value][dataset] = \
                                    getattr(evaluator, score_name)

    os.makedirs(results_path, exist_ok=True)

    for network in all_scores:
        for key in all_scores[network]:
            fname = os.path.join(results_path, "%s_%s_%s.csv" % (network, str(key), score_name))
            all_scores[network][key].to_csv(fname)


def load_results():
    """load saved results"""
    all_scores = {}

    for network in model.networks:
        all_scores[network] = {}
        fpattern = os.path.join(results_path, "%s_(*)_%s.csv" % (network, score_name))

        for fname in iglob(fpattern):
            basename = os.path.basename(fname)
            strkey = basename[len(network) + 1:-len(score_name) - 5]
            key = _str_to_tuple(strkey)
            all_scores[network][key] = pd.read_csv(fname, index_col=0, header=[0, 1])

    return all_scores


def cross_validate(all_scores: dict):
    """gets the scores on all of the datasets and returns the cross-validation scores"""
    top_scores = {}
    mean_scores_columns = []

    for network in all_scores:
        top_scores[network] = {}

        for key in all_scores[network]:
            top_scores[network][key] = pd.DataFrame()
            mean_scores_columns.append((key, network))

    for network in all_scores:
        for key in all_scores[network]:
            score = all_scores[network][key]
            argmax = np.nanargmax(score.to_numpy(), axis=1)
            max_alpha_betas = score.columns[argmax]

            for network2 in all_scores:
                if network != network2:
                    prediction_vals = all_scores[network2][key][max_alpha_betas].to_numpy().diagonal()
                    prediction = pd.DataFrame(data=prediction_vals, index=all_scores[network2][key].index)
                    top_scores[network2][key][network] = prediction[0]

    columns = pd.MultiIndex.from_tuples(mean_scores_columns, names=['key', 'network'])
    mean_scores = pd.DataFrame(columns=columns)

    for network in all_scores:
        for key in all_scores[network]:
            mean_score_vals = np.mean(top_scores[network][key].to_numpy(), axis=1)
            index = top_scores[network][key].index
            mean_scores[key, network] = pd.DataFrame(data=mean_score_vals, index=index)[0]

    return mean_scores


def average_datasets(scores: pd.DataFrame, stack: bool = False):
    """averages the dataframe over the indexes"""
    mean_score_vals = np.mean(scores.to_numpy(), axis=0).reshape((1, -1))
    mean_scores = pd.DataFrame(data=mean_score_vals, columns=scores.columns, index=['avg'])

    if stack:
        mean_scores = mean_scores.stack(level=-1).droplevel(level=0)

    return mean_scores


# generate_results()
# results = load_results()
# cross_vals = cross_validate(results)
# mean_vals = cv_mean_datasets(cross_vals)
# print(mean_vals)
