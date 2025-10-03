import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import argparse
import os
from pathlib import Path
from collections import defaultdict

feature_set_list=['All', 'Internal', 'Base', 'Base_aggregated', 'Aggregated_only']

def load_and_aggregate_json_files(json_dir):
    """
    Load all JSON files from directory and aggregate by configuration index.
    
    Args:
        json_dir (str): Directory containing JSON files
        
    Returns:
        dict: Individual file data and aggregated data
    """
    json_files = list(Path(json_dir).glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    all_file_data = {}
    config_aggregates = defaultdict(list)
    
    # Load all files
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_file_data[json_file.stem] = data
            
            # Group by configuration index
            for config_id, roc_data in data.items():
                config_aggregates[config_id].append(roc_data)
    
    # Aggregate by configuration
    aggregated_data = {}
    for config_id, roc_list in config_aggregates.items():
        aggregated_data[config_id] = aggregate_roc_curves(roc_list)
    
    return all_file_data, aggregated_data

def fix_infinity_threshold(threshold):
    """Convert string 'Infinity' to numpy inf"""
    if threshold == "Infinity" or threshold == "Infinity":
        return np.inf
    return threshold

def aggregate_roc_curves(roc_list):
    """
    Aggregate multiple ROC curves by averaging FPR, TPR, and thresholds.
    
    Args:
        roc_list (list): List of ROC curve dictionaries
        
    Returns:
        dict: Aggregated ROC curve data
    """
    if len(roc_list) == 1:
        # Fix infinity thresholds
        fixed_thresholds = [fix_infinity_threshold(t) for t in roc_list[0]['roc_thresholds']]
        return {
            'fpr': roc_list[0]['fpr'],
            'tpr': roc_list[0]['tpr'],
            'roc_thresholds': fixed_thresholds
        }
    
    # Find common length (use the shortest curve)
    min_length = min(len(roc['fpr']) for roc in roc_list)
    
    # Aggregate arrays
    fpr_arrays = []
    tpr_arrays = []
    threshold_arrays = []
    
    for roc in roc_list:
        # Interpolate to common length if needed
        if len(roc['fpr']) != min_length:
            # Use uniform sampling for simplicity
            indices = np.linspace(0, len(roc['fpr'])-1, min_length, dtype=int)
            fpr_arrays.append(np.array(roc['fpr'])[indices])
            tpr_arrays.append(np.array(roc['tpr'])[indices])
            thresholds = [fix_infinity_threshold(t) for t in roc['roc_thresholds']]
            threshold_arrays.append(np.array(thresholds)[indices])
        else:
            fpr_arrays.append(np.array(roc['fpr']))
            tpr_arrays.append(np.array(roc['tpr']))
            thresholds = [fix_infinity_threshold(t) for t in roc['roc_thresholds']]
            threshold_arrays.append(np.array(thresholds))
    
    # Calculate means
    mean_fpr = np.mean(fpr_arrays, axis=0)
    mean_tpr = np.mean(tpr_arrays, axis=0)
    mean_thresholds = np.mean(threshold_arrays, axis=0)
    
    return {
        'fpr': mean_fpr.tolist(),
        'tpr': mean_tpr.tolist(),
        'roc_thresholds': mean_thresholds.tolist()
    }

def plot_base_aggregated_across_models(all_file_data, output_file):
    """
    Plot ROC curves for each model using only the Base_aggregated feature set.
    Each JSON file corresponds to one model.
    """
    plt.figure(figsize=(10, 8))
    
    base_idx = feature_set_list.index("Base_aggregated")
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_file_data)))
    
    print("\nBase_aggregated AUC Summary:")
    print("-" * 35)
    
    for idx, (filename, data) in enumerate(all_file_data.items()):
        if str(base_idx) not in data:
            print(f"Skipping {filename} (no Base_aggregated data)")
            continue
        
        roc_data = data[str(base_idx)]
        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        
        roc_auc = auc(fpr, tpr)
        model_name = filename.split('_')[1]  # adjust if your naming differs
        label_map = {"o3": "o3", "gpt-5": "GPT-5-Minimal", "gpt-5-med": "GPT-5-Medium", "gpt-4.1": "GPT-4.1", "o4-mini": "o4-mini", "deepseek": "DeepSeek-R1", "llama": "LLaMA-3.3-70B-Instr.", "gpt-5-high": "GPT-5-High"}

        model_name= label_map[model_name]
        
        plt.plot(fpr, tpr, color=colors[idx], linewidth=3,
                 alpha=0.9, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        print(f"{model_name}: AUC = {roc_auc:.4f}")
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, 
             label='Random Classifier (AUC = 0.5)')
    
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.title("AUROC Across All Models:\n Base + Aggregated Features", fontsize=26, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBase_aggregated plot saved: {output_file}")
    plt.close()


def plot_individual_files(all_file_data, output_dir):
    """Plot ROC curves for each individual JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for filename, data in all_file_data.items():
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
        
        for idx, (config_id, roc_data) in enumerate(data.items()):
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx], linewidth=2.5, 
                    marker='o', markersize=4, alpha=0.8, 
                    label=f'{feature_set_list[int(config_id)]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, 
                 label='Random Classifier (AUC = 0.5)')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        title_file = filename.split('_')[1] if '_' in filename else filename
        plt.title(f'Average Performance by Feature Set - {title_file}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        output_file = output_dir / f'roc_{filename}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plot saved: {output_file}")

def plot_aggregated_curves(aggregated_data, output_file, title):
    """Plot aggregated ROC curves"""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(aggregated_data)))
    
    print("\nAggregated AUC Summary:")
    print("-" * 35)
    
    for idx, (config_id, roc_data) in enumerate(aggregated_data.items()):
        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], linewidth=3, 
                marker='o', markersize=5, alpha=0.8, 
                label=f'{feature_set_list[int(config_id)]} (AUC = {roc_auc:.3f})')
        
        print(f"Config {config_id}: AUC = {roc_auc:.4f}")
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, 
             label='Random Classifier (AUC = 0.5)')

    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.title(title, fontsize=26, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nAggregated plot saved: {output_file}")
    
    plt.show()

def save_aggregated_json(aggregated_data, output_file):
    """Save aggregated data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    print(f"Aggregated JSON saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot ROC curves from multiple JSON files")
    parser.add_argument('--json_dir', help='Directory containing JSON files')
    parser.add_argument('--output-dir', '-o', default='./roc_output', 
                       help='Output directory for plots')
    parser.add_argument('--title', '-t', default="Average Performance by Feature Set", 
                       help='Title for aggregated plot')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load and aggregate data
        print(f"Loading JSON files from: {args.json_dir}")
        all_file_data, aggregated_data = load_and_aggregate_json_files(args.json_dir)
        
        print(f"Found {len(all_file_data)} JSON files")
        print(f"Configurations found: {list(aggregated_data.keys())}")
        
        # Plot individual files
        print("\nGenerating individual plots...")
        plot_individual_files(all_file_data, output_dir / "individual")
        
        # Plot aggregated curves
        print("\nGenerating aggregated plot...")
        aggregated_plot_file = output_dir / "aggregated_roc_curves.png"
        plot_aggregated_curves(aggregated_data, aggregated_plot_file, args.title)
        
        # Save aggregated data
        aggregated_json_file = output_dir / "aggregated_roc_data.json"
        save_aggregated_json(aggregated_data, aggregated_json_file)
        
        print(f"\nAll outputs saved to: {output_dir}")
        plot_base_aggregated_across_models(all_file_data, output_dir / "base_aggregated_roc_curves.png")

    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())