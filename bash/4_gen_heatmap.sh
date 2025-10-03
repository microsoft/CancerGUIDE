#!/usr/bin/env bash
conda activate tb

MODELS=( "llama" "gpt-5-med" "gpt-5" "gpt-4.1" "o3" "o4-mini" "deepseek" "gpt-5-high")

BENCHMARK_DIR_UNSTRUCTURED= "../CancerGUIDE_Internal/data/benchmarks/synthetic_bench/synthetic_final_unstructured"
BENCHMARK_DIR_UNSTRUCTURED_GPT_5= "../CancerGUIDE_Internal/data/benchmarks/synthetic_bench/gpt-5_synthetic_final_unstructured"
BENCHMARK_DIR_STRUCTURED_GPT_5= "../CancerGUIDE_Internal/data/benchmarks/synthetic_bench/gpt-5_synthetic_final_structured"
BENCHMARK_DIR_STRUCTURED= "../CancerGUIDE_Internal/data/benchmarks/synthetic_bench/synthetic_final_structured"

BENCHMARK_DIR_HUMAN= "../CancerGUIDE_Internal/data/benchmarks/human_annotations"

OUTPUT_BASE= "../CancerGUIDE_Internal/results"
RESULTS_JSON="${OUTPUT_BASE}/heatmap_results.json"
RESULTS_BASE= "../CancerGUIDE_Internal/results/benchmark_results"
RESULTS_FIGURE="${OUTPUT_BASE}/figures"
mkdir -p "$RESULTS_FIGURE"
echo "Writing results to: $RESULTS_JSON"

for MODEL in "${MODELS[@]}"; do
  BENCHMARK_DIR_CB_path= "../CancerGUIDE_Internal/data/benchmarks/consistency_bench/${MODEL}/path_filter"
  BENCHMARK_DIR_CB_treatment= "../CancerGUIDE_Internal/data/benchmarks/consistency_bench/${MODEL}/treatment_filter"

  # Each benchmark is stored as "input_dir:subdir_name"
  BENCHMARKS=(
    "$BENCHMARK_DIR_HUMAN:human_new_prompt"
    "$BENCHMARK_DIR_UNSTRUCTURED:unstructured"
    "$BENCHMARK_DIR_STRUCTURED:structured"
    "$BENCHMARK_DIR_CB_path:path_filter"
    "$BENCHMARK_DIR_CB_treatment:treatment_filter"
  )

  for BENCHMARK_PAIR in "${BENCHMARKS[@]}"; do
    IFS=":" read -r BENCHMARK_DIR SUBDIR <<< "$BENCHMARK_PAIR"
    # Override dirs for gpt-4.1
    if [[ "$MODEL" == "gpt-4.1" && "$BENCHMARK_DIR" == "$BENCHMARK_DIR_STRUCTURED" ]]; then
      BENCHMARK_DIR="$BENCHMARK_DIR_STRUCTURED_GPT_5"
      SUBDIR="gpt-5_structured"
    fi
    if [[ "$MODEL" == "gpt-4.1" && "$BENCHMARK_DIR" == "$BENCHMARK_DIR_UNSTRUCTURED" ]]; then
      BENCHMARK_DIR="$BENCHMARK_DIR_UNSTRUCTURED_GPT_5"
      SUBDIR="gpt-5_unstructured"
    fi

    OUTPUT_DIR="$RESULTS_BASE/$MODEL/$SUBDIR"
    mkdir -p "$OUTPUT_DIR"

    echo "=== Model: $MODEL, Benchmark: $SUBDIR, Input: $BENCHMARK_DIR ==="

    python ../CancerGUIDE_Internal/scripts/analyses/benchmark_evaluate.py \
      --model "$MODEL" \
      --benchmark_dir "$BENCHMARK_DIR" \
      --results_json "$RESULTS_JSON" \
      --output_dir "$OUTPUT_DIR" \
      --benchmark_experiment "$SUBDIR" 
    #   --generation #optional flag to generate new outputs if malformed jsons
    
    echo $MODELS
    echo $BENCHMARK_DIR 
    echo $OUTPUT_DIR
  done
done



# aggregated bench
python ../CancerGUIDE_Internal/scripts/benchmark_generation/cross_model_consistency.py --filter_type path\
 --results_json "$RESULTS_JSON" --consistency_bench_aggregated --aggregation_mode \
  --experiment_name "path_aggregation" --model_list "gpt-4.1,gpt-5,o3,o4-mini,gpt-5-med,deepseek,llama,gpt-5-high"
python ../CancerGUIDE_Internal/scripts/benchmark_generation/cross_model_consistency.py --filter_type treatment\
 --results_json "$RESULTS_JSON" --consistency_bench_aggregated --aggregation_mode --experiment_name "treatment_aggregation"\
 --model_list "gpt-4.1,gpt-5,o3,o4-mini,gpt-5-med,deepseek,llama,gpt-5-high"

python ../CancerGUIDE_Internal/scripts/analyses/heatmap_correlations.py --result_json "$RESULTS_JSON" --output_dir "$RESULTS_FIGURE" --one_plot

# #make heatmap
echo "✅ Done. Appended results to $RESULTS_JSON"