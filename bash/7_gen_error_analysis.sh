#!/usr/bin/env bash
conda activate tb

MODELS=("gpt-5" "gpt-4.1" "o3" "o4-mini" "gpt-5-med" "deepseek" "llama" "gpt-5-high")
RESULTS_BASE="../CancerGUIDE_Internal/results/"

echo "Writing results to: $RESULTS_JSON"

for MODEL in "${MODELS[@]}"; do

DATA_PATH_HUMAN="$RESULTS_BASE/benchmark_results/${MODEL}/human_new_prompt"
OUTPUT_DIR_HUMAN="$RESULTS_BASE/error_analysis/human_analysis_${MODEL}/human_analysis"

DATA_PATH_CONSISTENCY="$RESULTS_BASE/rollout_results/rollout_experiment_${MODEL}"
OUTPUT_DIR_CONSISTENCY="$RESULTS_BASE/error_analysis/path_consistency_${MODEL}/path_consistency"

OUTPUT_DIR_RESULTS="$RESULTS_BASE/figures/${MODEL}_error_analysis"

mkdir -p "$OUTPUT_DIR_HUMAN"
mkdir -p "$OUTPUT_DIR_CONSISTENCY"
mkdir -p "$OUTPUT_DIR_RESULTS"

python ../CancerGUIDE_Internal/scripts/error_analysis/error_analysis_human.py --model $MODEL --data_path $DATA_PATH_HUMAN --output_dir $OUTPUT_DIR_HUMAN
python ../CancerGUIDE_Internal/scripts/error_analysis/error_analysis_consistency.py --model $MODEL --data_path $DATA_PATH_CONSISTENCY --output_dir $OUTPUT_DIR_CONSISTENCY
python ../CancerGUIDE_Internal/scripts/error_analysis/compare_mistakes.py --model $MODEL --pred_dir $OUTPUT_DIR_CONSISTENCY --human_dir $OUTPUT_DIR_HUMAN --output_dir $OUTPUT_DIR_RESULTS --gamma 1
done

# #make heatmap
echo "✅ Done. Appended results to $RESULTS_JSON"