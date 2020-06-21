import yaml
import os
import numpy as np
import sys


def load(data_path: str = None):
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    if data_path is None:
        config['data_root'] = os.path.abspath(config['data_root'])
        model_path = os.path.join(config['data_root'], config['model_name'])
    else:
        config['data_root'] = ''
        model_path = os.path.abspath(data_path)

    model_config_path = os.path.join(model_path, "config.yaml")

    with open(model_config_path, 'r') as model_file:
        model = yaml.safe_load(model_file)

    config = update(config, model)

    if isinstance(config['alpha_log2_values'], dict):
        values = config['alpha_log2_values']
        config['alpha_log2_values'] = np.linspace(values['start'], values['stop'], values['num'])

    if isinstance(config['beta_log2_values'], dict):
        values = config['beta_log2_values']
        config['beta_log2_values'] = np.linspace(values['start'], values['stop'], values['num'])

    if isinstance(config['model']['datasets'], dict):
        values = config['model']['datasets']
        datasets = np.arange(values['start'], values['stop'], values['step'])
        config['model']['datasets'] = [str(i) for i in datasets]

    if 'datasets_path' not in config:
        config['datasets_path'] = os.path.join(model_path, config['model']['datasets_folder'])

    if 'goldstandards_path' not in config:
        config['goldstandards_path'] = os.path.join(model_path, config['model']['goldstandards_folder'])

    if 'predictions_path' not in config:
        config['predictions_path'] = os.path.join(model_path, config['model']['predictions_folder'])

    if 'results_path' not in config:
        config['results_path'] = os.path.join(model_path, config['model']['results_folder'])

    if 'p_values_path' not in config:
        config['p_values_path'] = os.path.join(model_path, config['model']['p_values_folder'])

    if config['max_level'] == -1:
        config['max_level'] = sys.maxsize

    return config
