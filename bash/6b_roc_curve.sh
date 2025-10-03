#!/usr/bin/env bash
conda activate tb

python ../CancerGUIDE_Internal/scripts/roc_analysis/generate_roc_plots.py\
     --json_dir ../CancerGUIDE_Internal/results/json_results \
    --output-dir ../CancerGUIDE_Internal/results/figures