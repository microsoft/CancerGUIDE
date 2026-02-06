# CancerGUIDE: Cancer Guideline Understanding via Internal Disagreement Estimation

The **National Comprehensive Cancer Network (NCCN)** provides evidence-based guidelines for cancer treatment. Translating complex patient presentations into guideline-concordant treatment recommendations is time-intensive, requires specialized expertise, and is prone to error. Advances in large language model (LLM) capabilities promise to reduce this burden by enabling accurate, scalable recommendation systems.

**CancerGUIDE** is an LLM agent-based framework for automatically generating guideline-compliant treatment trajectories for patients with **non-small cell lung cancer (NSCLC)**.

https://doi.org/10.48550/arxiv.2509.07325

---

## ✨ Contributions

1. **Novel Dataset**  
   We construct a longitudinal dataset of **121 NSCLC patient cases** containing clinical encounters, diagnostic results, and medical histories. Each case is expertly annotated with the corresponding NCCN guideline trajectories by board-certified oncologists.

2. **Proxy Benchmark Generation**  
   We show that existing LLMs encode domain-specific knowledge sufficient for generating high-quality proxy benchmarks. These benchmarks correlate strongly with expert annotations  
   *(Spearman’s ρ = 0.88, RMSE = 0.08)*.

3. **Hybrid Verification Framework**  
   We introduce a hybrid approach combining expert annotations with model-consistency signals:  
   - An **agent framework** that predicts guideline trajectories.  
   - A **meta-classifier** that verifies predictions with calibrated confidence scores  
     *(AUROC = 0.800)*.  
   This improves interpretability, supports regulatory compliance, and allows practitioners to tailor tradeoffs between annotation cost and accuracy.

---

## 📂 Project Structure


CancerGUIDE/
├── scripts/
│ ├── benchmark_generation/ # Generate benchmarks (self-consistency, synthetic, cross-model)
│ ├── analyses/ # Model evaluation + figure generation
│ ├── data_processing/ # Raw patient note processing
│ └── ...
├── data/
│ ├── patient_data/ # Structured + unstructured patient notes (Protege source)
│ ├── human_labelling/ # Human labels + utilities for json-note alignment
│ └── benchmarks/ # Final benchmark files
├── results/
│ ├── benchmark_results/ # Model predictions
│ ├── figures/ # Paper figures
│ ├── json_results/ # Data for ROC analyses
│ ├── rollout_results/ # k rollouts for consistency analyses
│ └── heatmap_results.json # Aggregated numeric results for heatmaps
├── bash/ # Ordered bash scripts to reproduce paper analyses
└── README.md

---

## 🛠️ Scripts Overview

### `scripts/benchmark_generation/`
- Generate self-consistency, synthetic, and cross-model benchmarks.  
- **Note:** cross-model consistency must be run *after* self-consistency, since it uses rollout information for labels/accuracy scores.

### `scripts/analyses/`
- `benchmark_evaluate.py`: Evaluate a model on a dataset.  
- `heatmap_correlations.py`: Run all models on all datasets → produce heatmap of correlations/RMSE.  
- `clustering_approximation.py`: Unsupervised clustering experiment.  
- `accuracy_by_consistency_bar_plot.py`: Generate bar plots of accuracy by consistency (requires self-consistency benchmarks).  
- `error_analysis/`: Unsupervised error identification scripts.  
- `extra_analysis/`: Ancillary scripts for additional/unused evaluations.  
- `roc_analysis/`: Generate ROC curves using logistic regression.

### `scripts/data_processing/`
- Process raw patient notes and export into `data/patient_data/`.

---

## 📊 Data Layout

- **`data/patient_data/`**: Structured + unstructured patient notes.  
- **`data/human_labelling/`**:  
  - Raw notes sent for third-party expert annotation.  
  - `human_labels.json`: mapping patient IDs to labels.  
  - `utils/human_json_to_dir.py`: Converts `human_labels.json` + raw notes into note/label pairs for evaluation.  
- **`data/benchmarks/`**: Pre-built benchmark datasets for evaluation.

---

## 📈 Results Directory

- **`benchmark_results/`**: Raw predictions of each model on each dataset.  
- **`figures/`**: Plots and figures from the paper.  
- **`json_results/`**: Data used in ROC analyses.  
- **`rollout_results/`**: k-rollouts for consistency analysis.  
- **`heatmap_results.json`**: Aggregated results for heatmap visualization.

---

## 🔧 Reproducibility

- **`bash/`** contains ordered shell scripts to reproduce experiments and figures.  
- Scripts are aligned with analyses presented in the paper for straightforward replication.
