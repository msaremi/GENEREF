import os
import numpy as np
from evaluation import DREAM4S10Evaluator, DREAM4S100Evaluator, DREAM5Evaluator, GenericSimpleEvaluator
from types import SimpleNamespace as __
import sys

# Two models are present in this work, the DREAM4 models and the DREAM5 models
models = {
	"dream4_size10": __(
		name="dream4_size10",
		networks=["insilico_size10_%d" % (i + 1) for i in range(5)],
		datasets=['%d' % i for i in range(2)],
		evaluator=DREAM4S10Evaluator,
		learner_params=__(
			n_trees=100,
			max_features=1/7
		)
	),
	"dream4_size100": __(
		name="dream4_size100",
		networks=["insilico_size100_%d" % (i + 1) for i in range(5)],
		datasets=['%d' % i for i in range(10)],
		evaluator=DREAM4S100Evaluator,
		learner_params=__(
			n_trees=100,
			max_features=1/7
		)
	),
	"ecoli_subnetworks": __(
		name="ecoli_subnetworks",
		networks=["ecoli-%d" % i for i in [40, 80, 160, 320, 640]],
		datasets=['%d' % i for i in range(1)],
		evaluator=GenericSimpleEvaluator,
		learner_params=__(
			n_trees=100,
			max_features="sqrt"
		)
	),
	"dream5": __(
		name="dream5",
		# We only used newtork3 (E. Coli) to evaluate our algorithm
		networks=["dream5_%d" % (i + 1) for i in [2]],
		datasets=['%d' % i for i in range(1)],
		evaluator=DREAM5Evaluator,
		learner_params=__(
			n_trees=10,
			max_features=1/200
		)
	),
}

# any of the following criteria can be used
score_names = {
	"score": "score",
	"score_aupr": "score_aupr",
	"auroc": "auroc",
	"auroc_p_value": "auroc_p_value",
	"aupr": "aupr",
	"aupr_p_value": "aupr_p_value"
}

#
# Set the following hyper-parameters for your own use
#

alpha_log2_values = np.linspace(-2, 7, 13)
beta_log2_values = np.linspace(-7, 2, 13)

max_level = -1

model_name = "ecoli_subnetworks"
model = models[model_name]

score_name = score_names["auroc"]
# Do you want not to re-generate already made predictions? Set this flag to True
skip_existing_preds = False
# Dataset root
data_path = os.path.join(os.getcwd(), "data", model.name)

# Where the datasets are stored
datasets_path = os.path.join(os.getcwd(), "data", model.name, "datasets")
# Where the predictions will be saved
predictions_path = os.path.join(os.getcwd(), "data", model.name, "predictions")
# Where the results will be saved
results_path = os.path.join(os.getcwd(), "data", model.name, "results")

if max_level == -1:
	max_level = sys.maxsize
