# Get the Data

## Simulated and semi-simulated datasets

To replicate the paper main experiment, you will need to download the data.

- ACIC 2016: We use the R package associated with the data. You will a version
  of R installed and perform the following steps:
```
Load one acic 2016 simulation based on their R package,
[aciccomp2016](https://github.com/vdorie/aciccomp)
    Pre-requisite:
    ```
    if (require("remotes", quietly = TRUE) == FALSE) {
        install.packages("remotes")
        require("remotes")
    }
    remotes::install_github("vdorie/aciccomp/2016")
``` 

- ACIC 2018: Donwload only the scaling subset of the data at https://www.synapse.org/#!Synapse:syn11738963

- TWINS: The twins dataset can be downloaded with the utilitary function
  `load_twins` from [`caussim.data.loading.py`](/caussim/data/loading.py)

- Caussim Dataset: This dataset is simulation based only. The generator is the class `CausalSimulator` from [`caussim.data.simulations.py`](/caussim/data/simulations.py)

## Paper experiment results

The `experiences.zip` file contain the results of the main experience of the
paper. These results have been obtained by launching as is, the script
[`/scripts/experiences/causal_scores_evaluation.py`](/scripts/experiences/causal_scores_evaluation.py).
The csv contained in the zip should be sufficient to replicate the Figure 5.,
using
[`/scripts/reports/causal_scores_evaluation.py`](/scripts/reports/causal_scores_evaluation.py).
