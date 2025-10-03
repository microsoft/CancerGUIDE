#!/usr/bin/env bash
conda activate tb

python ../CancerGUIDE/scripts/accuracy_by_consistency_bar_plot.py\
        --outdir ../CancerGUIDE/results/figures\
        --base_path ../CancerGUIDE_Internal