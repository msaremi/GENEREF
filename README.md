# GENEREF
GEne NEtwork inference with REgularized Forests

## Algorithm Setup
You can run the algorithm by running "[main.py](main.py)". 
By running this file, the GENEREF algorithm is run for every network in the current model for every pack of datasets of the model
for every permutations of the datasets in each dataset pack for every 𝛼 and 𝛽 hyper-parameters. 

Suppose that: You have 𝐶 networks in your model;
There is 𝑃 dataset packs you want to reconstruct the model 
(𝑃=1 is normally what you need. If you want to calculate the robustness of the algorithm, then you can have more than 1 packs);
In the dataset pack there are 𝑀 datasets;
And, there are 𝑚×𝑛 combinations of 𝛼 and 𝛽. 
Then the complexity of the algorithm that is run in [main.py](main.py) will be 𝑂(𝐶 × 𝑃 × 𝑀 × 𝑚 × 𝑛).

The basic configurations that you need for running GENEREF are in "[config.py](config.py)".
These include the model you use (e.g. DREAM4 or DREAM5), the range of hyper-parameters 𝛼 and 𝛽 for grid search, etc.

The preferred configuration of the algorithm for each model is sotred in the variable `model`.
For example, for the DREAM4 networks, we defined the following in [config.py](config.py):

    from evaluation import DREAM4Evaluator
    from types import SimpleNamespace as __
    
    models["dream4"] = __(
		name="dream4",
		networks=["insilico_size100_%d" % (i + 1) for i in range(5)],
		datasets=['%d' % i for i in range(10)],
		evaluator=DREAM4S100Evaluator,
		learner_params=__(
			n_trees=100,
			max_features=1/7
		)
	)
	
Every model needs a name. In this case the name is "dream4". Hence, we set the value `name="dream4"`.
On the subsequent line, we defined the name of the networks to be used.
There are five networks whose names starts with "insilico_size100_" and is followed by an index.
To define the names of these networks, we wrote `networks=["insilico_size100_%d" % (i + 1) for i in range(5)]`.
For this model, we used ten dataset packs that are stored in ten folders ['0'](data/dream4/datasets/0), ..., ['9'](data/dream4/datasets/9).
So we defined the names of these packs using `datasets=['%d' % i for i in range(10)],`.
The `evaluator` of the results is not used by the main algorithm. Yet, if one wants to measure, AUROC, AUPR, etc.
they will need an evaluator class. 
`evaluator=DREAM4S100Evaluator` defines the class `DREAM4S100Evaluator` to be the class that can evaluate the results
constructed for the DREAM4 networks.
The `learner_params` parameter sets the parameters of the random forests that are used as the solvers.
These parameters are the number of trees in each forest `n_trees`, 
and the maximum proportion of features used when creating each node in the trees `max_features`.

If you want to use your own custom model, after preparation of the model files (explained in the following section),
set-up the custom entry in the `models` dictionary and set the variable `model_name` equal to the key of the custom entry.
Also, if you did not put your model inside the [data](data) folder, set the variables `datasets_path` and `predictions_path` accordingly.

The algorithm reads the data from the `datasets_path` folder and stores the predictions in the `predictions_path` folder.

If you want the number of iterations of GENEREF to be less than the number of datasets, then set the `max_level`
to the required value. Otherwise, make sure that max_lavel is greater than the number of your datasets (or set it to `-1`).

If the variable `skip_existing_preds` is set to `True`, then the algorithm will skip the previously generated predictions. 
This is useful if you want to continue a previously halted algorithm, so you prefer not to re-calculate all the results.  

GENEREF uses two hyper-parameters 𝛼 and 𝛽.
By setting `alpha_log2_values` and `beta_log2_values` triplets in [config.py](config.py) 
you can setup a grid search for these two hyper-parameters.
`alpha_log2_values` holds the logarithmic range of hyper-parameter 𝛼. 
Similarly, `beta_log2_values` holds the logarithmic range of hyper-parameter 𝛽.

To see the hyper-parameter values that GENEREF runs for, you can run the following:

    from config import alpha_log2_values, beta_log2_values
    from sklearn.utils.extmath import cartesian
    combinations = 2 ** cartesian((alpha_log2_values, beta_log2_values))
    print(combinations)

In the following section, there is an extensive description of the fields you can set in the [config.py](config.py) file.

### Configuration Fields

`models[name].name: str`<br/>
The name of a model.

<hr/>

`models[name].networks: List[str]`<br/>
The list of the names of networks in the model.

<hr/>

`models[name].datasets: List[str]`<br/>
The list of the names of the dataset packs used for reconstruction of the model.

<hr/>

`models[name].evaluator: type[Evaluator]`<br/>
The evaluator class for the predictions. (Not used in our algorithm main body).

<hr/>

`models[name].learner_params.n_trees: int`<br/>
Number of trees in each of the random forest regressors.

<hr/>

`models[name].learner_params.max_features: float`<br/>
Maximum number of features used for creating in node in the random forest trees.

<hr/>

`alpha_log2_values: numpy.array`<br/>
1-D array containing the log2 values of the 𝛼 hyper-parameter.
 
<hr/>

`beta_log2_values: numpy.array`<br/>
1-D array containing the log2 values of the 𝛽 hyper-parameter. 

<hr/>

`max_level: int`<br/>
Positive number showing the maximum number of iteration the GENEREF algorithm is going to be run.
Make sure to select a value greater than the number of datasets, if you want all dataset to be involved in the algorithm. 

<hr/>

`max_level: int`<br/>
Positive number showing the maximum number of iteration the GENEREF algorithm is going to be run. 

<hr/>

`model_name: str`<br/>
The current model key. 

<hr/>

`model: SimpleNamespace`<br/>
The current model. By setting `model_name` it will be automatically set. There is no reason to directly modify this line.

<hr/>

`skip_existing_preds: Bool`<br/>
If true, the previously stored predictions will not be calculated.

<hr/>

`datasets_path: os.PathLike`<br/>
The root where datasets of the current model are accessed, by default it is equal to "[data](data)/<model.name>/datasets".

`predictions_path: os.PathLike`<br/>
The root where predictions of the current model are stored, by default it is equal to "[data](data)/<model.name>/predictions".

## Data Management

If you want to use a dataset that is not DREAM4, you will need to prepare the dataset file in the format
that is described in this section. Thwe following section described the DataManager class, that provides an adapter layer between the files 
and the algorithm. If you are not interested in the manipulation of the code, skip this section and go to the Data File Formats section.
From this point, we will talk about dataset as each data file that store gene profiles.

### DataManager Class

The `DataManager` class is used to load a dataset of a network and/or store the predictions 
generated for that network. It manages all the datasets that are related to a network. 
The data that are loaded by this class contain the goldstandard network, the gene
profile files. This class also stores the prediction files. The predictions are the
confidence matrices that show how certainly there is an edge between each pair of genes.

#### Methods and Properties

`DataManager(path: str, network: str, preds_path: str = None)` <br/>
The constructor, that takes the location of the dataset and the targeted network:

* `path: str`<br/>
    The path to the root of the dataset. For example  ['[GENEREF DIR]\data\dream4\datasets\0\'](data/dream4/datasets/0) is the path for the dataset 0 of the DREAM4 networks.
    
* `network: str`<br/>
    The name of the network that is going to be loaded. For example, 'insilico_size100_1'.
    
* `preds_path: str = None`<br/>
    The path where the prediction matrices are going to be saved. If left blank, `path + 'preds'` will be used for the path for the confidence matrices.<br/> 
    This parameter is only used if a prediction is produced and added to the instance of `DataManager`.

<hr/>

`DataManager.goldstandard: UnweightedNetwork`<br/>
Gets the goldstandard network of the loaded network dataset.

<hr/>

`DataManager.experiments: List[Union[Experiment, ExperimentSet]]`<br/>
Gets the list of all (steady-state/knock-out) experiments and (timeseries) experiment-sets

<hr/>

`DataManager.predictions: DataManager.Predictions`<br/>
Gets the list of all predictions

<hr/>

`DataManager.get_steadystates(): List[SteadyStateExperiment]`<br/>
Returns the list of all steady-state experiments

<hr/>

`DataManager.get_timeseries_sets(): List[TimeseriesExperimentSet]`<br/>
Returns the list of all timeseries experiment-sets

### Data File Formats

#### Contents ofFiles
There will be four types of files in a network dataset: steady-state, time-series, goldstandard, and prediction:

* A steady-state data file is a .tsv file in which the steady-state expression levels of all genes are stored. 
Each row in the file contains one measurement of the expression level of all genes. The file can have as many rows as required.
The first row of this file stores the names of the genes. 

* A time-series data file is a .tsv file in which the time-stamped expression levels of all genes are stored.
Unlike a steady-state data file that holds the values of one experiment, a time-series file contains sets of experiments.
Each experiment in this file is stored in consecutive rows. The first column of the row, indicates the time point when the expression levels were captured; The other columns correspond the the expression value of the corresponding genes.
Multiple experiments can be stored in one time-series file, one after another. Similar to the steady-state data files, the first row in this file is labels of the genes, with the first column labeled as "time".

* A goldstandard file is a .tsv file that holds the adjacency matrix of the ground truth network in a sparse format. 
Each column of this files contains three rows.
The first row is the label of the first gene and the second row is the label of the second gene.
The third row can be either 1, indicating that there is a direct link from the first to the second gene, 
or 0, indicating that there is no such link.
All rows that have the third column equal to 0 can be omitted in the file.

*  A prediction file is a .tsv file that stores a confidence matrix.
The first row and first column of this file store the index of genes.
The value stored on column (i+1) row (j+1) is corresponding to the edge from gene i to gene j.  

The file formats are compatible with the standard file formats that are used in the DREAM4 challenge.
That is, for the steady-state files, you can use the format of "*multifactorial.tsv" files, 
and for the times-series files, you can use the format of "*timeseries.tsv" files. 
Just keep in mind that the first row in a data file is used as a compatibility feature with the DREAM4 dataset;
Label your genes using the letter 'G' followed by its index; 
Also, if you have multiple dataset files, make sure that the genes are sorted in all of them -- GENEREF does not read
profiles based on the label column.

#### Naming Input Files

Each file in a dataset folder is equivalent to a dataset.
There are three groups of experiment files: "multifactorial", "knockouts", or "timeseries".
Both "multifactorial" and "knockouts" files contain the information of steady-state experiments and should follow its contents format as discussed in the previous section.
A "timeseries" file contains the information of a **set of** time-series experiments and should follow its contents format.

The general name of a data file is as follows: `<NetworkName>_<ExperimentType>[<FileCounter>].tsv`.
The `<NetworkName>` field is the name of the network. 
For example, if your dataset is the first 100-gene network of DREAM4, it should be named "insilico_size100_1".
The `<ExperimentType>` can be either of "multifactorial", "knockouts", or "timeseries". 
The `<FileCounter>` field is the index of the file starting from zero. Each group of the files have their own indexing.

In addition to the experiment files, there must be a file that contains the goldstandard network.
It is named `<NetworkName>_goldstandard.tsv`.

### Prediction Files

Running [main.py](main.py), stores many prediction files. The general name of a prediction file
is as follows: `<NetworkName>_prediction[(<alphaLogValue>, <betaLogValue>), <DSetIndex1>, <DSetIndex2>, ...].tsv`

`<NetworkName>` will be the name of the network that the prediction is computed for. 
`<alphaLogValue>` and `<betaLogValue>` are the 𝛼 and 𝛽 log2 values that were used.
`<DSetIndex1>, <DSetIndex2>, ...` is the list of the index of datasets that GENEREF was run on.

Please note that these indices are different from the ones that are associated to each
group of dataset files (multifactorial/knockouts/timseries). This is the global indexing based on the
order that datasets are loaded. The algorithm loads multifactorial, knockouts and timeseries datasets respectively. 

### Example

The following is an example of a dataset folder:

* dream4
    * datasets
        * 0
            * insilico_size100_1_goldstandard.tsv
            * insilico_size100_1_multifactorial[0].tsv
            * insilico_size100_1_multifactorial[1].tsv
            * insilico_size100_1_knockouts[0].tsv
            * insilico_size100_1_timeseries[0].tsv
            * insilico_size100_1_timeseries[1].tsv
            * insilico_size100_2_goldstandard.tsv
            * insilico_size100_2_multifactorial[0].tsv
            * insilico_size100_2_multifactorial[1].tsv
            * insilico_size100_2_multifactorial[2].tsv
            * insilico_size100_2_timeseries[0].tsv
            
This folder contains the information of the experiments of two DREAM4 networks: insilico_size100_1 and insilico_size100_2.
To load all the datasets of the first network, it is enough to write the following code:

    from networkdata import DataManager
    data_manager = DataManager(path='dream4/datasets/0/', network='insilico_size100_1')

It loads all files in the folder starting with "insilico_size100_1".
If you want to get the goldstandard network in the programming environment,
it is enough to write:

    goldstandard = data_manager.goldstandard

The following line, returns all the steady-state experiments of this dataset.

    steady_states = data_manager.get_steadystates()
    
Since there are 3 steady-states datasets for network 1 
("insilico_size100_1_multifactorial[0].tsv", "insilico_size100_1_multifactorial[0].tsv", "insilico_size100_1_knockouts[0].tsv"),
`len(steady_states)` will be equal to `3`.

Now, let's make a **random** prediction for this network and store it:

    import numpy as np
    from networkdata import WeightedNetwork
    prediction_data = np.random.rand(100, 100)
    prediction = WeightedNetwork(prediction_data)
    prediction_name = 'random_prediction'
    data_manager.predictions[prediction_name] = prediction
    
The prediction file will be saved in 'dream4/datasets/0/preds/insilico_size100_1_prediction[random_prediction].tsv'

# Details of the Algorithm

In this section, we will provide the details about the important classes of this repo.

## Predictor Class
The `Predictor` class is a regularized random forest class that dedicatedly constructs confidence matrices based on the 
the regularization matrix and one dataset.

### Methods and Properties

`Predictor(num_of_jobs: int = 8, n_trees: int = 100, trunk_size: int = None, max_features: float = 1/7,
                 callback: Callable[[int, int], None] = None)`<br/>
Establishes a predictor object.

* `num_of_jobs: int = 8`<br/>
    The maximum number of parallel jobs that the Predictor will use to solve the problem.
    
* `n_trees: int = 100`<br/>
    Number of trees in each regressor.
    
* `trunk_size: int = None`<br/>
    Number of decomposed datasets before they are given to the regressors. If `None`, the Predictor decomposes the 
    dataset with 𝐺 genes to 𝐺 new datasets. If your memory is limited, you can set this number to a smaller number to
    make a balance between memory consumption and runtime speed.
    
* `max_features: float = 1/7`<br/>
    Maximum number of feature when creating a node in a tree in the random forest. The default value is `1/7`. However,
    you can try the square root of the number of genes for the possibly optimal result. This feature makes a trade-off
    between the accuracy of the regressors and the runtime speed.
    
* `callback: Callable[[int, int], None] = None`<br/>
    The callable that is called after each trunk is solved. The first argument is the number of sub-problems solved
    so far, and the second argument is the total number of sub-problems (the number of genes).

<hr/>

`Predictor.fit(experiment: Union[SteadyStateExperiment, TimeseriesExperimentSet], regularization=None) -> None`<br/>
Gets a steady-state experiment or a time-series experiment set along with a regularization matrix and 
constructs a confidence matrix using regularized random forests.

* `experiment: Union[SteadyStateExperiment, TimeseriesExperimentSet]`<br/>
    The experiment of experiment set.
    
* `regularization: numpy.ndarray = None`<br/>
    The regularization matrix. If none, then the algorithm will be run without regularization.

<hr/>

`Predictor.network: WeightedNetwork`<br/>
Gets the confidence matrix that is produced using the `fit` method.

 ## Evaluator Class
 The `Evaluator` class is an abstract class that calculates performance metrics of a predicted network, based on a goldstandard network.
 Two primarily child classes are `DREAM4S10Evaluator`, `DREAM4S100Evaluator`, and `DREAM5Evaluator`.
 The `DREAM4S10Evaluator` and `DREAM4S100Evaluator` are our own implementation based on [G Stolovitzky et.al](https://nyaspubs.onlinelibrary.wiley.com/doi/abs/10.1111/j.1749-6632.2009.04497.x), 
 while the `DREAM5Evaluator` is the re-implementation of the DREAM5's 
 original evaluation algorithm that is originally implemented in MATLAB.
 
 ### Methods and Properties
 
`Evaluator(network: str)`<br/>
Instantiates the class based the name of the network of interest.
 
* `network: str`<br/>
    The name of the network. In both `DREAM4Evaluator` and `DREAM5Evaluator` the name is the index of the network 
    starting from 1.
    
<hr/>

`Evaluator.fit(labels: numpy.ndarray, scores: numpy.ndarray) -> None`<br/>
Evaluates a prediction based on the goldstandard network.
 
* `labels: numpy.ndarray`<br/>
    A 𝐺×𝐺 matrix (where 𝐺 is the number of genes) indicating the goldstandard network where non-edges are 0 and edges are 1.
    
* `scores: numpy.ndarray`<br/>
    A 𝐺×𝐺 matrix (where 𝐺 is the number of genes) indicating the prediction matrix showing the corresponding value for each pair of genes.
    
<hr/>

`Evaluator.network: str`<br/>
Gets the name of the network.

<hr/>

`Evaluator.auroc: float`<br/>
Gets the AUROC value.

<hr/>

`Evaluator.aupr: float`<br/>
Gets the AUPR value.

<hr/>

`Evaluator.auroc_p_value: float`<br/>
Gets the AUROC p value.

<hr/>

`Evaluator.aupr_p_value: float`<br/>
Gets the AUPR p value.

<hr/>

`Evaluator.score: float`<br/>
Gets the score of the prediction.

<hr/>

`Evaluator.score_aupr: float`<br/>
Gets the score_AUPR of the prediction.

# Acknowledgements
The `rainforest` library in this repo is a fork of [sklearn](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py) RandomForestRegressors.