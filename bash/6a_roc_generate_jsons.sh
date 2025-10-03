#!/usr/bin/env bash
METHOD=holdout

MODELS=("gpt-4.1" "gpt-5" "o4-mini" "gpt-5-med" "deepseek" "o3" "llama" "gpt-5-high")

for MODEL in "${MODELS[@]}"; do
  echo "Running grid search for test model: $MODEL"
  python ../CancerGUIDE_Internal/scripts/roc_analysis/generate_roc_jsons.py \
    --mode grid_search \
    --models gpt-4.1 gpt-5 o4-mini gpt-5-med deepseek o3 llama gpt-5-high\
    --train-sizes 0.25 \
    --feature-sets all internal base base_aggregated aggregated_only\
    --methods $METHOD \
    --testmodel $MODEL \
    --outdir ../CancerGUIDE_Internal/results/json_results/ \
    --heatmap_results ../CancerGUIDE_Internal/results/heatmap_results.json
done