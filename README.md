How to select predictive model for causal inference
===================================================

Overview
--------
This package contains simulations for causal inference, estimators for ATE and
CATE as well as code for experiments described in the paper : How to select
predictive models for causal inference ? 

Package Features
----------------

The package code is contained in: [caussim](caussim/)

- `estimation` contains CATE and ATE estimators usable with any scikit-learn compatible base estimators and meta-learners such as TLearner, SLearner or RLearner.
- `simulations` simulations with basis expansion (available Nystroem, Splines)
- `experiences` used to run extensive evaluation of causal metrics on ACIC 2016
  and handcrafted simulations.
- `reports` contains the scripts used to derive figures and tables presented in
  the paper. The main results are obtained by launching the 
- `utils.py` plot utils
- `pdistances` naive implementation of MMD, Total Varation and Jensen Shannon Divergences used to measure population overlap

- `demos` contains notebooks used to create toy example and risks maps for the 2D simulations.
- `data` contains utils to load semi-simulated datasets (ACIC 2016, ACIC 2018,
  TWINS). A dedicated [README](data/README.md) is
  available in the root data folder.


Experiences
----------

Experiences outputs are mainly csvs (one for each sampled dataset). To launch an experience, run `python scripts/experiences/<experience.py>` and it should output the csv in a dedidacted folder in the corresponding subfolder `data/experiences/<dataset>/<experience_name>`.

**🔎 Replicate the main experience of the paper (section 5.)**, launch the script
[scripts/experiences/causal_scores_evaluation.py](scripts/experiences/causal_scores_evaluation.py). 
Make sure that the configurations for the datasets at the beginning of the file
is : 

```python
from caussim.experiences.base_config import DATASET_GRID_FULL_EXPES
DATASET_GRID = DATASET_GRID_FULL_EXPES
```

📢 Note that the results of the section 5 are already provided in the zenodo link [`experiences.zip`]([data/experiences.zip](https://zenodo.org/records/13765465?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJmNTFlOWNjLTUxOTYtNGFjNS04YjVjLTIyZWFjMmNhZjQyMyIsImRhdGEiOnt9LCJyYW5kb20iOiJlOTZjZGE4ZmQzNDFkMWUxNTJhYzI0YWI1ZjUxNGViMyJ9.vPuJgBw0A0w02InS9ovWRShKUGTDk4w6k2uwYBZklRiC-p7hlVvZOOyvpg6wsJ6T5MBW30vUCsL_UdBSCmmFMw)).

Reports
-------

Reports outputs are mainly figures for the papers. To obtain the results, run `pytest scripts/reports/<report.py>` and it should output the figures in one or several corresponding folders in `figures/`.

The main report type is a pytest function contained in the `reports/causal_scores_evaluation.py` script. For each macro-dataset, it plot the results of running a given set of candidate estimators with a fixed nuisance estimator on several generation process of the macro-dataset (often hundreds of sampled datasets). 

**🔎 Replicate the main figure of the paper (Figure 5.)**, launch the script
[scripts/reports/causal_scores_evaluation.py](scripts/reports/causal_scores_evaluation.py).
It should take some time because of the high number of simualtions results. Make
sure that the appropriate experiences results exists. The one used in the paper
are provided in [`experiences.zip`](data/experiences.zip).

```
pytest scripts/reports/causal_scores_evaluation.py
```

Installation
============

- We recommand the use of poetry and python>=3.9 to manage dependencies. 

You can install caussim via 
[poetry](https://python-poetry.org/):
 ```shell script
poetry install
```

or

[pip](https://pip.pypa.io/). In this case you also need to install the dependies listed in the `pyproject.toml`:
 ```shell script
pip install caussim
```

Dependencies: 
------------

python = ">=3.9, <3.11"  
python-dotenv = "^0.15.0"  
click = "^8.0.1"  
yapf = "^0.31.0"  
matplotlib = "^3.4.2"
numpy = "^1.20.3"  
seaborn = "^0.11.1"  
jupytext = "^1.11.5"  
rope = "^0.19.0"  
scikit-learn = "^1.0"  
jedi = "^0.18.0"  
tqdm = "^4.62.3"  
tabulate = "^0.8.9"  
statsmodels = "^0.13.1"  
pyarrow = "^6.0.1"  
submitit = "^1.4.1"  
rpy2 = "^3.4.5"  
moepy = "^1.1.4"  

