#!/usr/bin/env bash
conda activate tb

MODEL="llama" #("gpt-5-high" "llama" "gpt-4.1" "gpt-5-med" "gpt-5" "o4-mini" "o3" "deepseek")
python ../CancerGUIDE_Internal/scripts/benchmark_evaluate.py --model $MODEL\
 --benchmark_dir ../CancerGUIDE_Internal/data/benchmarks/human_annotations \
 --output_dir ../CancerGUIDE_Internal/results/benchmark_results/$MODEL/human \
 --benchmark_experiment human_annotations