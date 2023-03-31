<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/WinIT/blob/main/logos/logobox.jpg" width="180"></a>
</p>

## ICLR'23 "Temporal Dependencies in Feature Importance for Time Series Prediction"

Authors: KK Leung, Clayton Rooke, Jonathan Smith, Saba Zuberi, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)  
[[paper](https://openreview.net/forum?id=C0q9oBc3n4)]

## Introduction

This repository contains a full implementation of the WinIT algorithm along with all the other
results for comparison. It includes a notebook that demonstrates the reproducibility of
the figures and graphs.

## Environment

```commandline
conda create -n winit python=3.8.11
conda activate winit
```

One of our dependencies is [TimeSynth](https://github.com/TimeSynth/TimeSynth),
which contains dependencies on symengine. It is possible that there will be errors
in installing symengine version 0.4. Installing TimeSynth from source will fix the issue.

```commandline
git clone https://github.com/TimeSynth/TimeSynth.git
cd TimeSynth
python setup.py install
cd ..
```

After TimeSynth is installed, run the following to install the package.

```commandline
pip install -e .
```

## Dataset

### Synthetic datasets

Our data are generated using the simulated data from [FIT Repo](https://github.com/sanatonek/time_series_explainability). 
The data is already generated and is in the `./data/` directory.

- The original spike dataset (`./data/simulated_spike_data`)
- Four spike datasets with delays of 1 through 4 (`./data/simulated_spike_data_delay_X`).
- The original state dataset (`./data/simulated_state_data`)


### Mimic datasets

MIMIC-III is a private dataset. Refer
to [the official MIMIC-III documentation](https://mimic.mit.edu/iii/gettingstarted/dbsetup/).
(ReadMe and datagen of MIMIC is from [Dynamask Repo](https://github.com/JonathanCrabbe/Dynamask).

- Run this command to acquire the data and store it:
   ```shell
   python -m winit.datagen.icu_mortality --sqluser YOUR_USER --sqlpass YOUR_PASSWORD
   ```
  If everything happens properly, two files named ``adult_icu_vital.gz`` and ``adult_icu_lab.gz``
  are stored in ``./data/mimic``.

- Run this command to preprocess the data:
   ```shell
   python -m fit.data_generator.data_preprocess
   ```
  If everything happens properly, a file ``patient_vital_preprocessed.pkl`` is stored
  in ``./data/mimic``.

## Running the Code

Note that our code is designed to be reproducible. We use `torch.use_deterministic_algorithms(True)`.
For some of the code, this will induce an error. It is suggested that we set the environment
variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`. See [here](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
for details.

### Train the models

Run these commands to train the models. By default, the models are 1-layer GRUs. To explore
different
types of models, use `--numlayers` and `--modeltype` args.

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --train --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --delay 2 --train --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data state --train --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data mimic --train --skipexplain
```

The models will be saved at `./ckpt/[MODELTYPE]/[DATASET]/`

### Train the generators

Run these commands to train the generators. It is expected to take a long time to train all
generators. The MIMIC-III generators take about 36 hours to train for all the CVs and features in
our machines.

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --traingen --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --delay 2 --traingen --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data state --traingen --skipexplain
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data mimic --traingen --skipexplain
```

The generators will be saved at `./ckpt/[MODELTYPE]/[DATASET]/[CV]/feature_generator/` or
`./ckpt/[MODELTYPE]/[DATASET]/[CV]/joint_generator/`

### Compute, save and evaluate the the feature importances

Run these commands to compute the importances. Use `--explainer` args to run different explainers.
You
can run several explainers at a time.

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --eval 
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --delay 2 --eval
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data state --eval
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data mimic --eval
```

The importances will be saved
at `./output/[MODELTYPE]/[DATASET]/[EXPLAINER_NAME]_test_importance_score_[CV].pkl`. The results 
of the evaluation will be saved at `./output/[MODELTYPE]/[DATASET]/results.csv`

### Other baselines
For FIT, we will need a joint feature generator. Thus we have to train the generators for FIT.

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --traingen --skipexplain --explainer fit
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --delay 2 --traingen --skipexplain --explainer fit
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data state --traingen --skipexplain --explainer fit
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data mimic --traingen --skipexplain --explainer fit
```

Then we can compute, save and evaluate the importances. 

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --eval --explainer deeplift gradientshap ig fo afo fit dynamask
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data spike --delay 2 --eval --explainer deeplift gradientshap ig fo afo fit dynamask
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data state --eval --explainer deeplift gradientshap ig fo afo fit dynamask
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run --data mimic --eval --explainer deeplift gradientshap ig fo afo fit dynamask
```

The results of the evaluation will be saved at the same file `./output/[MODELTYPE]/[DATASET]/results.csv`

### Notebooks

After running all the experiments needed, check out the notebook [here](notebooks/Reproduce.ipynb).


## File structure

<details>
<summary>File structure</summary>

```text
.
├── winex
│   ├── Code
├── data
│   ├── simulated_spike_data
│   ├── simulated_spike_data_delay_1
│   ├── ...
│   ├── simulated_state_data
│   └── patient_vital_preprocessed.pkl
├── ckpt
│   ├── gru1layer
│   │   ├── mimic
│   │   │   ├── model files
│   │   │   ├── 0 (cv)
│   │   │   │   ├── feature_generator
│   │   │   │   └── joint_generator
│   │   │   ├── ..
│   │   │   └── 4 (cv)
│   │   │       ├── feature_generator
│   │   │       └── joint_generator
│   │   ├── ...
│   │   └── simulated_spike_delay_2
│   ├── ...
│   └── lstm
│       ├── mimic
│       │   ├── model files
│       │   ├── 0 (cv)
│       │   │   ├── feature_generator
│       │   │   └── joint_generator
│       │   ├── ..
│       │   └── 4 (cv)
│       │       ├── feature_generator
│       │       └── joint_generator
│       ├── ...
│       └── simulated_spike_delay_2
├── output
│   ├── gru1layer
│   │   ├── mimic
│   │   │   ├── importance files
│   │   │   └── result.csv
│   │   ├── ...
│   │   └── simulated_spike_delay_2
│   │       ├── importance files
│   │       └── result.csv
│   ├── ...
│   └── lstm
│       ├── mimic
│       │   ├── importance files
│       │   └── result.csv
│       ├── ...
│       └── simulated_spike_delay_2
│           ├── importance files
│           └── result.csv
├── plots
│   ├── gru1layer
│   │   ├── mimic
│   │   │   ├── box plots
│   │   │   ├── generator_array
│   │   │   │   └── generator training curve arrays
│   │   │   └── array
│   │   │       └── masking numpy arrays
│   │   ├── ...
│   │   └── simulated_spike_delay_2
│   │       ├── box plots
│   │       ├── generator_array
│   │       │   └── generator training curve arrays
│   │       └── array
│   │           └── masking numpy arrays
│   ├── ...
│   └── lstm
│       ├── mimic
│       │   ├── box plots
│       │   ├── generator_array
│       │   │   └── generator training curve arrays
│       │   └── array
│       │       └── masking numpy arrays
│       ├── ...
│       └── simulated_spike_delay_2
│           ├── box plots
│           ├── generator_array
│           │   └── generator training curve arrays
│           └── array
│               └── masking numpy arrays
├── notebooks
│   └── demo notebooks
└── logs
    └── log files
```

</details>
