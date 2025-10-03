#!/usr/bin/env bash
conda activate tb

MODEL="llama"  #("gpt-5-high" "llama" "gpt-4.1" "gpt-5-med" "gpt-5" "o4-mini" "o3" "deepseek")
python ../CancerGUIDE/scripts/evaluate.py --model $MODEL\
 --benchmark_dir ../CancerGUIDE/data/benchmarks/synthetic_bench/synthetic_final_unstructured \
 --output_dir ../CancerGUIDE/results/benchmark_results/$MODEL/unstructured \
 --benchmark_experiment unstructured