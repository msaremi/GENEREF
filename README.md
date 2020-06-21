# GENEREF
**GEne NEtwork inference with REgularized Forests**

## Algorithm Setup
You can run the algorithm by running "[main.py](main.py)". 
By running this file, the GENEREF algorithm is run for every network in the current model for every pack of datasets of the model
for every permutations of the datasets in each dataset pack for every 𝛼 and 𝛽 hyper-parameters. 
The current model can be determined either in the command line or in the "[config.yaml](config.yaml)". The simplest way is
using the command line. Just type to following command in the command line:

    >> python.exe main.py "<PathToTheModel>"

If you skip `<PathToTheModel>` the algorithm will use the model refered to in "[config.yaml](config.yaml)".

### Algorithm Specifications
Suppose that: You have 𝐶 networks in your model;
There is 𝑃 dataset packs you want to reconstruct the model 
(𝑃=1 is normally what you need. If you want to calculate the robustness of the algorithm, then you can have more than 1 packs);
In the dataset pack there are 𝑀 datasets;
And, there are 𝑚×𝑛 combinations of 𝛼 and 𝛽. 
Then the complexity of the algorithm that is run in [main.py](main.py) will be 𝑂(𝐶 × 𝑃 × 𝑀 × 𝑚 × 𝑛).

### Algorithm Configurations
The basic configurations that you need for running GENEREF are in "[config.yaml](config.yaml)".
These include the model you use (e.g. DREAM4 or DREAM5), the range of hyper-parameters 𝛼 and 𝛽 for grid search, etc.
Also, each model has its own "config.yaml" file. This file overwrites the settings of "[config.yaml](config.yaml)" and
adds extra information about the model.

Let's take a quick look at the configuration stored in "[data/dream4_size100/config.yaml](data/dream4_size100/config.yaml)", 
which is the configuration file for the DREAM4 size 100 netowkrs:

    ---
    alpha_log2_values:
      start: -6
      stop: 6
      num: 17
    beta_log2_values:
      start: -6
      stop: 6
      num: 17
    model:
      name: dream4_size100
      description: |-
        DREAM4 size100 networks
      folder: dream4_size100
      networks:
        - insilico_size100_1
        - insilico_size100_2
        - insilico_size100_3
        - insilico_size100_4
        - insilico_size100_5
      datasets:
        - '0'
        - '1'
        - '2'
        - '3'
        - '4'
        - '5'
        - '6'
        - '7'
        - '8'
        - '9'
      has_p_values: true
      has_self_loops: false
    learner_params:
      n_trees: 100
      max_features: 0.15
      trunk_size: 100
	
This file has some general settings, as with `alpha_log2_values` and `beta_log2_values`. But it also defines the model that
the algorithm is going to use.
Every model needs a name. In this case the name is "dream4_size100". Hence, we set the value `model\name=dream4_size100`.
The next line is an arbitrary field `model\description` that adds a description to the model.
On the subsequent line, we defined the name of the networks to be used.
There are five networks whose names starts with "insilico_size100_" and is followed by an index.
For this model, we used ten dataset packs that are stored in ten folders ['0'](data/dream4_size100/datasets/0), ..., ['9'](data/dream4_size100/datasets/9).
So we defined the names of these packs in `model\datasets`.
The `model\has_p_values` demonstrate whether or not the AUROC and AUPR metadata information have been defined for this model.
This metadata is not used by the main algorithm. Yet, if one wants to measure, AUROC, AUPR, etc.
they will need to provide the cumulative distribution of p-values of the null distributions for each network in the model
in specific files and set this field equal to `true`. The `model\has_self_loops` indicates if the models have self cycles or not.

The ".yaml" file defines other properties. In summary, the `learner_params` parameter sets the parameters of the algorithm or random forests that are used as the solvers.
These parameters are the number of trees in each forest `learner_params\n_trees`, 
and the maximum proportion of features used when creating each node in the trees `learner_params\max_features`,
and properties related to the parallel solving of the problem `learner_params\trunk_size`, and `learner_params\parallel_jobs`.

You can modify "[config.yaml](config.yaml)" to alter the global properties of GENEREF.
If you want to use your own custom model, after preparation of the model files (explained in the following section),
set-up the custom "config.yaml" file in the root dictionary of the model and set the field `model_name` in
"[config.yaml](config.yaml)" equal to the root folder name of that custom model.
Also, if you did not put your model inside the "[data](data)" folder, 
set the variables `data_root` accordingly. The algorithm reads the data from the model folder and 
stores the predictions in the "predictions" folder.

If you want the number of iterations of GENEREF to be less than the number of datasets, then set the `max_level`
to the required value. Otherwise, make sure that max_lavel is greater than the number of your datasets (or set it to `-1`).

If the field `skip_existing_preds` is set to `true`, then the algorithm will skip the previously generated predictions. 
This is useful if you want to continue a previously halted algorithm, so you prefer not to re-calculate all the results.  

GENEREF uses two hyper-parameters 𝛼 and 𝛽.
By setting `alpha_log2_values` and `beta_log2_values` triplets in "[config.yaml](config.yaml) ""
you can setup a grid search for these two hyper-parameters.
`alpha_log2_values` holds the logarithmic range of hyper-parameter 𝛼. 
Similarly, `beta_log2_values` holds the logarithmic range of hyper-parameter 𝛽.

To see the hyper-parameter values that GENEREF runs for, you can run the following:

    >> python.exe toolset.py report_grid_values

In the following section, there is an inclusive description of the fields you can set in the "[config.yaml](config.yaml)" and
 the model's "config.yaml" files.

### Configuration Fields

`model_name: str` (only in [config.yaml](config.yaml))<br/>
The folder in which the model lies.

<hr/>

`learner_params\n_trees: int`<br/>
Number of trees in each of the random forest regressors.

<hr/>

`learner_params\max_features: Union[float, str]`<br/>
Maximum number of features used for creating in node in the random forest trees. You can use either a proportion in `[0, 1]` or `"sqrt"`.

<hr/>

`learner_params\trunk_size: int`<br/>
Number of sub-problem that can simultaneously exist in the memory. Set it to a lower value if you have limited capacity of the
main memory.

<hr/>

`learner_params\parallel_jobs: int`<br/>
Number of parallel threads of the algorithm. Defaults to 8.

<hr/>

`alpha_log2_values: Union[object, List[float]]`<br/>
If an array, each element is a value for the 𝛼 hyper-parameter.
If object, `start`, `stop` and `num` determine the values for the 𝛼 hyper-parameter.
 
<hr/>

`beta_log2_values: Union[object, List[float]]`<br/>
If an array, each element is a value for the 𝛽 hyper-parameter.
If object, `start`, `stop` and `num` determine the values for the 𝛽 hyper-parameter.

<hr/>

`max_level: int = -1`<br/>
Positive number showing the maximum number of iteration the GENEREF algorithm is going to be run.
Make sure to select a value greater than the number of datasets, if you want all dataset to be involved in the algorithm.
If the number of datasets is unknown, you can set it to `-1`. 

<hr/>

`skip_existing_preds: bool = true`<br/>
If `true`, the previously stored predictions will not be calculated; otherwise, they will be calculated.

<hr/>

`model\name: str` (only in models' .yaml)<br/>
The name of a model.

<hr/>

`model\name: description` (only in models' .yaml)<br/>
An optional description field for the model.

<hr/>

`model\has_p_values: bool = false`<br/>
Determines whether the model defines the p-value metadata.

<hr/>

`model\has_self_loops: bool = false`<br/>
Determines whether the model networks have self-cycles.

<hr/>

`model\evaluator: Type[Evaluator] = Evaluator`<br/>
Name of the class that is used for the model's evaluation.
The class must be a subclass of `evaluation.Evaluator`.
(Not used in our algorithm main body.)

<hr/>

`model\networks: List[str]` (only in models' .yaml)<br/>
The list of the names of networks in the model.

<hr/>

`models\datasets: List[str]` (only in models' .yaml)<br/>
The list of the names of the dataset packs used for reconstruction of the model.

<hr/>

`model\datasets_folder: str = "datasets"`<br/>
The folder name where datasets of the current model are accessed.

<hr/>

`model\predictions_folder: str = "predictions"`<br/>
The folder name where predictions of the current model are stored.

<hr/>

`model\goldstandards_folder: str = "goldstandards"`<br/>
The folder name where goldstandards of the current model are stored.

<hr/>

`model\results_folder: str = "results"`<br/>
The folder name where results of the current model are stored.
(Not used in our algorithm main body.)

<hr/>

`model\p_values_folder: str = "p_values"`<br/>
The folder name where the p-value metadata files of the current model are stored.
(Not used in our algorithm main body.)



## Data Management

If you want to use a dataset that is not DREAM4, you will need to prepare the dataset files in the format and structure them in the way
that is described in this section. The following section described the DataManager class, that provides an adapter layer between the files 
and the algorithm. If you are not interested in the manipulation of the code, skip this section and go to the Data File Formats section.
From this point, we will talk about dataset as each data file that store gene profiles.

### DataManager Class

The [`DataManager`](bionetwork/networkdata.py) class is used to load a dataset of a network and/or store the predictions 
generated for that network. It manages all the datasets that are related to a network. 
The data that are loaded by this class contain the goldstandard network, the gene
profile files. This class also stores the prediction files. The predictions are the
confidence matrices that show how certainly there is an edge between each pair of genes.

#### Methods and Properties

`DataManager(path: str, network: str, preds_path: str = None)` <br/>
The constructor, that takes the location of the dataset and the targeted network:

* `path: str`<br/>
    The path to the root of the dataset. For example  ['[GENEREF DIR]\data\dream4\datasets\0\'](data/dream4_size100/datasets/0) is the path for the dataset 0 of the DREAM4 networks.
    
* `network: str`<br/>
    The name of the network that is going to be loaded. For example, 'insilico_size100_1'.
    
* `preds_path: str = None`<br/>
    The path where the prediction matrices are going to be saved. If left blank, `path + 'datasets' + pack + 'preds'` will be used for the path for the confidence matrices.<br/> 
    This parameter is only used if a prediction is produced and added to the instance of `DataManager`.
    
* `gold_path: str = None`<br/>
    The path where the goldstandards are going to be saved. If left blank, `path + 'goldstandard'` will be used for the path for the confidence matrices.<br/> 
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

#### Contents of Files
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

Besides, there are AUROC and AUPR p-value files that store the CDF of the null distribution of a random predictor. These
are .npy files in which there exists a 2-by-n matrix. The first row of the matrix defines the x value that is in the range of [0, 1]
and the second row defines the corresponding p-value CDF values to the x values.  

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

The general name of the p-value metadata files is as folows: `<NetworkName>_<AUROC_Or_AUPR>.npy`.
For each network two .npy files are created: one for the AUROC values and one for the AUPR values.
You can skip these files, but remember to set `model\has_p_values = false` in the model's config.yaml file.

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

The following is an example of a hypothetical dataset directory:

    data/
    └── dream4_size100_new_dset/
        ├── config.yaml
        ├── datasets/
        │   └── 0/
        │       ├── insilico_size100_1_multifactorial[0].tsv
        │       ├── insilico_size100_1_multifactorial[1].tsv
        │       ├── insilico_size100_1_knockouts[0].tsv
        │       ├── insilico_size100_1_timeseries[0].tsv
        │       ├── insilico_size100_1_timeseries[1].tsv
        │       ├── insilico_size100_2_multifactorial[0].tsv
        │       ├── insilico_size100_2_multifactorial[1].tsv
        │       ├── insilico_size100_2_multifactorial[2].tsv
        │       └── insilico_size100_2_timeseries[0].tsv
        ├── goldstandards/
        │   ├── insilico_size100_1_goldstandard.tsv
        │   └── insilico_size100_2_goldstandard.tsv
        └── p_values/
            ├── insilico_size100_1_auroc.npy
            ├── insilico_size100_1_aupr.npy
            ├── insilico_size100_2_auroc.npy
            └── insilico_size100_2_aupr.npy
            
The "dream4_size100_new_dset" folder contains the information of the experiments of two DREAM4 networks: insilico_size100_1 and insilico_size100_2.
To load all the datasets of the first network, it is enough to write the following code:

    from networkdata import DataManager
    data_manager = DataManager(path='dream4_size100_new_dset/datasets/0/', 
                               network='insilico_size100_1', 
                               gold_path='dream4_size100_new_dset/goldstandard')

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
 
This class also deals with the p-values. For the DREAM4 networks, we used our own implementation based on [G Stolovitzky et.al](https://nyaspubs.onlinelibrary.wiley.com/doi/abs/10.1111/j.1749-6632.2009.04497.x), 
One of its primary child class is `DREAM5Evaluator`. It is the re-implementation of the DREAM5's 
original evaluation algorithm that is originally implemented in MATLAB.
 
 ### Methods and Properties
 
`Evaluator(network: str)`<br/>
Instantiates the class based the name of the network of interest.
 
* `network: str`<br/>
    The name of the network.
* `p_values_path: str`<br/>
    The path in which p-value meta-data files are stored.
* `self_loops: str`<br/>
    Indicates whether the possible self-cycles of the network should be included in the calculation of the metrics
    
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