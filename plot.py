import os
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
from config import model, score_name, data_path, max_level

datasets_path = os.path.join(data_path, "datasets")

best_values = {}
# 0: Multifactorial
# 1: Knockout
# 2: TimeSeries
# plot_keys = [(2,), (1,), (0,)]
# plot_keys = [(0,), (1,), (0, 1), (1, 0)]
plot_keys = [(2,), (1,), (0,), (1, 2), (2, 0), (1, 0), (1, 2, 0)]

num_networks = len(model.networks)
num_boxes = len(plot_keys) #* 2
j0 = -(num_boxes // 2)
j = j0#+1
boxes = {}

for key in plot_keys:
	print("Plotting", key)
	fname = os.path.join(data_path, "results", "result_%s_%s.csv" % (score_name, str(key)))
	y = np.genfromtxt(fname).T
	y = np.hstack((y, np.mean(y, axis=1).reshape(-1, 1)))
	print(y.shape)
	x = np.arange(num_networks + 1) * (num_boxes + 2) + j
	bp = plt.boxplot(y, positions=x, showfliers=False)

	medians = np.reshape([m.get_ydata()[0] for m in bp['medians']], [-1, 1])
	whiskers = np.reshape([m.get_ydata()[0] for m in bp['whiskers']], [-1, 2])
	caps = np.reshape([m.get_ydata()[0] for m in bp['caps']], [-1, 2])
	boxes[key] = np.hstack((x.reshape(-1, 1), caps[:, :1], whiskers[:, :1], medians, whiskers[:, 1:], caps[:, 1:]))
	plot_fname = os.path.join(data_path, "results", "GENEREF_%s_%s.csv" % (str(key)[1:-1].replace(' ', ''), score_name))
	np.savetxt(plot_fname, boxes[key], delimiter=',')

	j += 1

plt.xlim(j0 - 1, num_networks * (num_boxes + 2) + j0)


# scores = []
#
# for i in range(1, max_level):
# 	keys = list(permutations(range(3), i))
# 	scores.append(np.zeros((5, len(keys))))
#
# 	for j, key in enumerate(keys):
# 		print("Plotting", key)
# 		fname = os.path.join(data_path, "results", "result_%s_%s.csv" % (score_name, str(key)))
# 		print(np.mean(np.genfromtxt(fname), axis=1).shape)
# 		scores[-1][:, j] = np.mean(np.genfromtxt(fname), axis=1)
#
# 	scores[-1] = np.mean(scores[-1], axis=1)
#
# scores = np.vstack(scores)
# print(scores)
# x = np.tile(np.arange(1, 4), (5, 1)).T
#
# for i in range(x.shape[1]):
# 	plt.plot(x[:, i], scores[:, i], label="Network %d" % (i + 1))
#
# plt.grid(True)
# plt.legend()
# plt.show()
		# x = np.arange(num_networks) * (num_boxes + 2) + j
		# bp = plt.boxplot(y, positions=x, showfliers=False)
		#
		# medians = np.reshape([m.get_ydata()[0] for m in bp['medians']], [-1, 1])
		# whiskers = np.reshape([m.get_ydata()[0] for m in bp['whiskers']], [-1, 2])
		# caps = np.reshape([m.get_ydata()[0] for m in bp['caps']], [-1, 2])
		# boxes[key] = np.hstack((x.reshape(-1, 1), caps[:, :1], whiskers[:, :1], medians, whiskers[:, 1:], caps[:, 1:]))
		# plot_fname = os.path.join(data_path, "results", "plot_%s_%s.csv" % (score_name, str(key)[1:-1].replace(' ', '')))
		# np.savetxt(plot_fname, boxes[key], delimiter=',')
#

# import os
# import numpy as np
# from collections import deque
# from algorithm import Predictor
# from itertools import permutations
# from evaluation import DREAM4Evaluator, DREAM5Evaluator
# from networkdata import DataManager, WeightedNetwork
# from types import SimpleNamespace as __
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# models = {
# 	"dream4": __(
# 		networks=["insilico_size100_%d" % (i + 1) for i in range(0, 1)],
# 		datasets=['%d' % i for i in range(1)],
# 		evaluator=DREAM4Evaluator
# 	),
# 	"dream5": __(
# 		networks=["dream5_%d" % (i + 1) for i in [2]],
# 		datasets=['%d' % i for i in range(1)],
# 		evaluator=DREAM5Evaluator
# 	)
# }
#
# model_name = "dream4"
# datasets_path = os.path.join(os.getcwd(), "data", model_name, "datasets")
# alpha_values = np.linspace(-4, 6, 13)
# beta_values = np.linspace(-6, 4, 13)
# max_level = 4
#
# for network in models[model_name].networks:
# 	evaluator = models[model_name].evaluator(network)
#
# 	for dataset in models[model_name].datasets:
# 		dataset_path = os.path.join(datasets_path, dataset)
# 		data_manager = DataManager(dataset_path, network)
#
# 		with data_manager:
# 			num_experiments = len(data_manager.experiments)
#
# 			for i in range(1, max_level):
# 				ax = plt.figure().add_subplot(111, projection='3d')
# 				first_level = i == 1
# 				second_level = i == 2
# 				keys = list(permutations(range(num_experiments), i))
#
# 				for key in keys:
# 					current_experiment_id = key[-1]
# 					current_experiment = data_manager.experiments[current_experiment_id]
# 					alphas = [0] if first_level else alpha_values
# 					betas = [0] if first_level else beta_values
# 					scores = np.zeros((len(alphas), len(betas)))
#
# 					for x, alpha_value in enumerate(alphas):
# 						for y, beta_value in enumerate(betas):
# 							prediction = data_manager.predictions[(alpha_value, beta_value) + key]
# 							evaluator.fit(data_manager.goldstandard.data, prediction.data)
# 							scores[x, y] = evaluator.auroc
#
# 					# scores = np.max(scores, axis=0)
#
# 					if first_level:
# 						scores = np.tile(scores, (len(alpha_values), len(beta_values)))
#
# 					X, Y = np.meshgrid(alpha_values, beta_values)
# 					ax.plot_wireframe(X, Y, scores, label=(network,) + key)
#
# 				plt.legend()
# 				plt.grid = True
#
#
#
# # for network in models[model_name].networks:
# # 	print("Network %s" % network)
# # 	keys = deque([])
# # 	key = ()
# # 	level = 0
# # 	evaluator = models[model_name].evaluator(network)
# #
# # 	while level <= max_level:
# # 		for dataset in models[model_name].datasets:
# # 			dataset_path = os.path.join(datasets_path, dataset)
# # 			data_manager = DataManager(dataset_path, network)
# #
# # 			with data_manager:
# # 				for experiment_id, experiment in enumerate(data_manager.experiments):
# # 					# alphas = alpha_values if level != 0 else [float('NaN')]
# #
# # 					def level_key():
# # 						return key + (experiment_id,)
# #
# # 					if level > 0:
# # 						if level > 0:
# # 							scores = []
# #
# # 							for alpha_value in alpha_values:
# # 								prediction = data_manager.predictions[(alpha_value,) + level_key()]
# # 								evaluator.fit(data_manager.goldstandard.data, prediction.data)
# # 								scores.append(evaluator.auroc)
# #
# # 							plt.plot(alpha_values, scores, label=(network,) + level_key())
# #
# # 					keys.append(level_key())
# #
# # 		key = keys.popleft()
# #
# # 		if len(key) > level:
# # 			plt.legend()
# # 			plt.figure()
# #
# # 		level = len(key)
# #
# # plt.legend()
# # plt.show()
