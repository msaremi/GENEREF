"""
Use this file to generate the results based on the predictions
"""

import os
import numpy as np
from datetime import datetime as dt
from itertools import permutations
from networkdata import DataManager
from config import load as load_config
import evaluation
from sklearn.utils.extmath import cartesian
import pandas as pd
from glob import iglob
import argparse
from inspect import signature
import sys


def _str_to_tuple(string):
    def _to_num(num_str: str):
        try:
            return int(num_str)
        except ValueError:
            return float(num_str)

    return tuple(map(_to_num, filter(None, string[1:-1].split(','))))


def generate_results(data_path: str = None, score_name: str = None):
    """produces results based on the predictions"""
    config = load_config(data_path)
    all_scores = {}

    if score_name is None:
        score_name = config['score_name']

    evaluator_class = getattr(evaluation, config['model']['evaluator']) \
        if 'evaluator' in config['model'] \
        else evaluation.Evaluator

    p_values_path = config['p_values_path'] if config['model']['has_p_values'] else None
    self_loops = config['model']['has_self_loops']

    for network in config['model']['networks']:
        print(dt.now(), "Network", network)

        evaluator = evaluator_class(network=network, p_values_path=p_values_path, self_loops=self_loops)
        all_scores[network] = {}

        for dataset in config['model']['datasets']:
            print(dt.now(), "\t", "Dataset", dataset)
            dataset_path = os.path.join(config['datasets_path'], dataset)
            prediction_path = os.path.join(config['predictions_path'], dataset)
            goldstandard_path = config['goldstandards_path']
            data_manager = DataManager(dataset_path, network, preds_path=prediction_path, gold_path=goldstandard_path)

            with data_manager:
                num_experiments = len(data_manager.experiments)

                for i in range(1, min(config['max_level'], num_experiments + 1)):
                    first_level = i == 1
                    keys = list(permutations(range(num_experiments), i))

                    for key in keys:
                        print(dt.now(), "\t\t", "Key", key)
                        alphas = [0] if first_level else config['alpha_log2_values']
                        betas = [0] if first_level else config['beta_log2_values']

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

    os.makedirs(config['results_path'], exist_ok=True)

    for network in all_scores:
        for key in all_scores[network]:
            fname = os.path.join(config['results_path'], "%s_%s_%s.csv" % (network, str(key), score_name))
            all_scores[network][key].to_csv(fname)


def _load_results(data_path: str = None, score_name: str = None):
    """load saved results"""
    config = load_config(data_path)
    all_scores = {}

    for network in config['model']['networks']:
        all_scores[network] = {}
        fpattern = os.path.join(config['results_path'], "%s_(*)_%s.csv" % (network, score_name))

        for fname in iglob(fpattern):
            basename = os.path.basename(fname)
            strkey = basename[len(network) + 1:-len(score_name) - 5]
            key = _str_to_tuple(strkey)
            all_scores[network][key] = pd.read_csv(fname, index_col=0, header=[0, 1])

    return all_scores


def _cross_validate(all_scores: dict):
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


def _average_datasets(scores: pd.DataFrame, stack: bool = False):
    """averages the dataframe over the indexes"""
    mean_score_vals = np.mean(scores.to_numpy(), axis=0).reshape((1, -1))
    mean_scores = pd.DataFrame(data=mean_score_vals, columns=scores.columns, index=['avg'])

    if stack:
        mean_scores = mean_scores.stack(level=-1).droplevel(level=0)

    return mean_scores


def report_results(data_path: str = None, score_name: str = None,
                   cross_validate: bool = True, average_datasets: bool = True):
    results = _load_results(data_path, score_name)

    if cross_validate:
        results = _cross_validate(results)

    if average_datasets:
        for network in results:
            for key in results[network]:
                results[network][key] = _average_datasets(results[network][key], stack=True)

    print(results)


def report_grid_values(data_path: str = None):
    config = load_config(data_path)
    alpha, beta = config['alpha_log2_values'], config['beta_log2_values']
    grid_values = 2 ** cartesian((alpha, beta))
    grid = pd.DataFrame(grid_values, columns=['alpha', 'beta'])
    print(grid)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        function_name = sys.argv[1]

        if function_name.startswith('_'):
            raise PermissionError("You are not allowed to run the protected function '%s'" % function_name)

        parser = argparse.ArgumentParser()
        parser.add_argument('function_name')
        function = globals()[function_name]
        sig = signature(function)

        for name, value in sig.parameters.items():
            if value.annotation == str:
                parser.add_argument('-%s' % name, dest=name, default=None)
            elif value.annotation == bool:
                parser.add_argument('--%s' % name, dest=name, default=False, action='store_true')

        arguments = vars(parser.parse_args())
        del arguments['function_name']
        function(**arguments)
    else:
        raise PermissionError("Cannot run this module directly without a function name")
