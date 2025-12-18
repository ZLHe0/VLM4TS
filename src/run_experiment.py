"""
General-purpose script to run ViT4TS on a specified dataset and evaluate performance across multiple alpha thresholds.
Usage: python src/run_experiment.py <DatasetName>
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Add src to path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.vit4ts import ViT4TS
from evaluation.evaluate import evaluate_intervals

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_anomalies():
    """Load and parse anomalies.csv."""
    anomalies_path = os.path.join(DATA_DIR, 'anomalies.csv')
    if not os.path.exists(anomalies_path):
        raise FileNotFoundError(f"{anomalies_path} not found.")
    
    anomalies_dict = {}
    with open(anomalies_path, 'r') as f:
        # Skip header
        lines = f.readlines()[1:]
        
    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
        
        signal = parts[0]
        try:
            # Parse the list of list string
            events = ast.literal_eval(parts[1].strip('"'))
            anomalies_dict[signal] = events
        except Exception as e:
            print(f"Error parsing anomalies for {signal}: {e}")
            
    return anomalies_dict

def run_experiment(dataset_name):
    # Check if dataset directory exists
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print(f"Please run 'python src/preprocessing/download_data.py {dataset_name}' first.")
        return

    # Load ground truth
    print("Loading ground truth anomalies...")
    ground_truth = load_anomalies()
    
    # Define alphas to test
    alphas = [0.1, 0.01, 0.001]
    
    # Iterate over alphas
    for alpha in alphas:
        print(f"\n--- Running experiment with alpha={alpha} ---")
        results_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'results_{dataset_name}_alpha_{alpha}.csv')
        
        # Initialize detector with current alpha
        detector = ViT4TS(
            window_size=100,
            window_step_ratio=4.0,
            model_name='ViT-B-16',
            image_size=(224, 224),
            alpha=alpha,
            verbose=False
        )
        
        results = []
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        print(f"Processing {dataset_name} ({len(files)} signals)...")
        
        for file_name in tqdm(files):
            signal_name = file_name.replace('.csv', '')
            file_path = os.path.join(dataset_dir, file_name)
            
            # Load data
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
            if data.empty:
                print(f"File {file_name} is empty. Skipping.")
                continue

            # Run detection
            try:
                # Suppress vision model warnings during batch processing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anomalies = detector.detect(data)
            except Exception as e:
                print(f"Error processing {signal_name}: {e}")
                results.append({
                    'dataset': dataset_name,
                    'signal': signal_name,
                    'alpha': alpha,
                    'status': 'failed',
                    'error': str(e)
                })
                continue
            
            # Evaluate
            if signal_name in ground_truth:
                gt_intervals = ground_truth[signal_name]
                detected_intervals = anomalies[['start', 'end']].values.tolist()
                
                metrics = evaluate_intervals(gt_intervals, detected_intervals)
                
                results.append({
                    'dataset': dataset_name,
                    'signal': signal_name,
                    'alpha': alpha,
                    'status': 'success',
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['F1'],
                    'n_detected': len(detected_intervals),
                    'n_ground_truth': len(gt_intervals)
                })
            else:
                results.append({
                    'dataset': dataset_name,
                    'signal': signal_name,
                    'alpha': alpha,
                    'status': 'no_gt',
                    'n_detected': len(anomalies)
                })

        # Save results for this alpha
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(results_file, index=False)
            print(f"\nResults for alpha={alpha} saved to {results_file}")
            
            # Print summary
            success_df = df_results[df_results['status'] == 'success']
            if not success_df.empty:
                print(f"\nPerformance Summary (alpha={alpha}):")
                print(success_df[['precision', 'recall', 'f1']].mean())
        else:
            print(f"No results to save for alpha={alpha}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT4TS experiment on a specific dataset across multiple alphas.")
    parser.add_argument("dataset", help="Name of the dataset (e.g., SMAP, MSL)")
    args = parser.parse_args()
    
    run_experiment(args.dataset)
