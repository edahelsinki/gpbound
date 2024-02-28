# Error bounds for any regression model using Gaussian processes with gradient information

This repository contains the source code for running the experiments in the paper: 

> Savvides, Rafael, Hoang Phuc Hau Luu & Kai Puolamäki (2024). Error bounds for any regression model using Gaussian processes with gradient information. AISTATS 2024.

`run.sh` describes how to run the experiments. 

- `experiments/` contains all experiment scripts. To run individual experiments, run the corresponding lines in `run.sh`. 
- `results/` contains all the results from the experiments. 
- `data/` contains all data sets used in the experiments.

Most experiments were run in parallel on a high-performance computing cluster. It is not advised to run all experiments in sequence on a personal computer. 
