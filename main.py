import os
import numpy as np
from project import Manager
from utils import print_timed
import pandas as pd

# Name of the algorithm and the results folder
# See Manager.generate_predictions for the details of the configuration
algorithm = "GENEREF (MF+TS)"
datasets_path = os.path.join(os.getcwd(), "data", "datasets")
result_path = os.path.join(os.getcwd(), "data", "results", algorithm)
os.makedirs(result_path, exist_ok=True)

networks = ["insilico_size100_%d" % (i + 1) for i in range(5)]
datasets = range(10)
alpha_values = np.linspace(-25, 10, 60)

num_networks = len(networks)
num_datasets = len(datasets)
num_values = alpha_values.shape[0]

print_timed("Algorithm: %s" % algorithm)

results = dict([(network, {
    "auroc":            np.zeros((num_values, num_datasets)),
    "auroc_p_value":    np.zeros((num_values, num_datasets)),
    "aupr":             np.zeros((num_values, num_datasets)),
    "aupr_p_value":     np.zeros((num_values, num_datasets)),
    "score":            np.zeros((num_values, num_datasets)),
}) for network in networks])

for dataset in datasets:
    print_timed("Dataset %d" % dataset, depth=1)

    dataset_path = os.path.join(datasets_path, str(dataset))

    for network in networks:
        print_timed(network, depth=2)

        for i, alpha_value in enumerate(alpha_values):
            print_timed("alpha = %g (%g%%)" % (alpha_value, ((i + 0.5) / num_values * 100)), depth=3,
                        start='\r', end='')

            with Manager(dataset_path, network) as manager:
                manager.generate_predictions(alpha_value)
                score = manager.score

                for k, metric in enumerate(results[network]):
                    results[network][metric][i, dataset] = score[k]

        print('\r', end='')

for network in networks:
    for metric in results[network]:
        data = pd.DataFrame(results[network][metric].T, columns=alpha_values)
        file_name = os.path.join(result_path, "%s_%s.tsv" % (network, metric))
        data.to_csv(file_name, sep="\t")

# for network in networks:
#     print_timed("Network %d" % network, depth=1)
#
#     for i, val in enumerate(alpha_values):
#         print_timed("Iteration %d out of %d: param = %g" % (i + 1, num_values, val), depth=2)
#         result = 0.0
#
#         for dataset in datasets:
#             dataset_path = os.path.join(datasets_path, str(dataset), str(network + 1))
#             results_path = os.path.join(dataset_path, "results", algorithm)
#             os.makedirs(results_path, exist_ok=True)
#             results = ResultPack(num_values, num_datasets)
#
#             print_timed("Round %d out of %d (%g%%)" % (j + 1, num_datasets, ((j + 1) / num_datasets * 100)), depth=3, start='\r', end='')
#
#             with Manager(dataset_path, network) as manager:
#                 manager.generate_predictions(val)
#                 results[i, j] = manager.score
#
#         print('\r', end='')
#
#     results.save(os.path.join(results_path, "%s.csv"))
# # plt.show()
# # for i in range(5):
# #     mean, std = np.mean(results[:, :, i], axis=1), np.std(results[:, :, i], axis=1)
# # # plt.plot(vals, mean, label="GENEREF (MF+TS)")
# # # plt.fill_between(vals, mean - std / 2, mean + std / 2, color='orange', alpha=0.4)
# #     print(mean)
#     # print(std)
# # plt.legend()
# # winsound.MessageBeep()
# # plt.show()
#
# # plt.plot([vals[0], vals[-1]], [base.genie3multifactorial.auroc.expectations[network]] * 2, label=r"GENIE3")
# # plt.plot([vals[0], vals[-1]], [base.genie3timeseries.auroc.expectations[network]] * 2, label=r"GENIE3-like timeseries")
# # plt.grid(True)
# # plt.title(r'Network %d' % (network + 1), fontsize=14)
# # plt.xlabel(r'log₂ρ', fontsize=14)
# # plt.ylabel(r'AUROC', fontsize=14)