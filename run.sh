#!/bin/bash
# This scripts runs everything (installs dependencies, downloads data, runs experiments).
# It is not meant to be run from start to finish but to describe how to reproduce the results.  
# Requirements: conda>=23.9.0, R>=4.3.1.

# Make conda environment and install R dependencies
conda env create -n bound_local --file env_local.yml
conda env create -n bound_cluster --file env_cluster.yml
Rscript scripts/install_dependencies.R

# Get datasets
conda activate bound_local
python data/data.py

# Run bound validity experiment
conda activate bound_cluster
for i in {0..47}; do
    python experiments/run_bound.py "$i" gp_rbf 200
    python experiments/run_bound.py "$i" svm 200
    python experiments/run_bound.py "$i" rf 200
done
python experiments/run_bound.py "-1"

# Run scaling experiment
for i in {0..99}; do
    python experiments/run_scaling.py "$i" bound
    python experiments/run_scaling.py "$i" rbf
    python experiments/run_scaling.py "$i" rq
    python experiments/run_scaling.py "$i" matern
done
python experiments/run_scaling.py "-1"

# Run real data experiment
for i in {0..23}; do
    python experiments/run_bound_real.py "$i" 100
done
python experiments/run_bound_real.py "-1"

# Run drift experiment
for i in {0..23}; do
    python experiments/run_drift.py "$i"
done
python experiments/run_drift.py "-1"
python experiments/run_drift.py "-1" datasets

# Make plots
cd scripts || exit
conda activate bound_local
python make_plots.py
R -e 'suppressPackageStartupMessages({rmarkdown::render("figures.Rmd")})'
