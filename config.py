import os
import numpy as np
from evaluation import DREAM4Evaluator, DREAM5Evaluator
from types import SimpleNamespace as __

# Two models are present in this work, the DREAM4 models and the DREAM5 models
models = {
	"dream4": __(
		name="dream4",
		networks=["insilico_size100_%d" % (i + 1) for i in range(5)],
		datasets=['%d' % i for i in range(10)],
		evaluator=DREAM4Evaluator,
		learner_params=__(
			n_trees=100,
			max_features=1/7
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
	)
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

alpha_values = np.linspace(-2, 7, 13)
beta_values = np.linspace(-7, 2, 13)
# alpha_values = np.linspace(-2, 7, 5)
# beta_values = np.linspace(-7, 2, 5)

# (0, 2)
# alpha_values = [0.475]
# beta_values = [-0.49]
# (2, 0)
# alpha_values = [-0.845]
# beta_values = [1.055]

max_level = 4

model = models["dream4"]
score_name = score_names["score"]
# Don't you want to re-generate already made predictions? Set this flag to True
skip_existing_preds = False
# Dataset root
data_path = os.path.join(os.getcwd(), "data", model.name)


datasets_path = os.path.join(os.getcwd(), "data", model.name, "datasets")
predictions_path = os.path.join(os.getcwd(), "data", model.name, "predictions")