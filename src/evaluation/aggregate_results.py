"""
Script to aggregate results and compute F1max statistics.
"""

import os
import glob
import pandas as pd
from pathlib import Path

def aggregate_results(results_dir='results'):
    # Find all result files
    search_pattern = os.path.join(results_dir, 'results_*_alpha_*.csv')
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"No result files found in {results_dir}.")
        return

    # Parse result files and organize by dataset and model
    data = []

    for file in result_files:
        # Parse filename: results_{dataset}_{model}_alpha_{alpha}.csv
        # or results_{dataset}_alpha_{alpha}.csv (old format, model=vit4ts)
        parts = Path(file).stem.split('_')

        # Find alpha position
        try:
            alpha_idx = parts.index('alpha')
            alpha = float(parts[alpha_idx + 1])
        except ValueError:
            print(f"Skipping file {file}: 'alpha' not found or invalid.")
            continue

        # Determine model and dataset
        if 'vlm4ts' in file or 'vit4ts' in file:
            # New format: results_{dataset}_{model}_alpha_{alpha}.csv
            # We assume model is immediately before 'alpha'
            # But wait, my script saves as results_{dataset}_{model}_alpha_{alpha}.csv
            # So dataset is between 'results' and {model}
            # results is parts[0]
            # alpha is parts[alpha_idx]
            # model is parts[alpha_idx - 1]
            
            # Check if model part is valid
            possible_model = parts[alpha_idx - 1]
            if possible_model in ['vit4ts', 'vlm4ts']:
                model = possible_model
                dataset = '_'.join(parts[1:alpha_idx-1])
            else:
                 # Maybe simpler parsing logic is needed if dataset has underscores
                 # But sticking to assumption for now.
                 # Let's handle the case where dataset has underscores more robustly if needed
                 # For now, SMAP/MSL don't have underscores.
                 model = possible_model
                 dataset = '_'.join(parts[1:alpha_idx-1])

        else:
            # Old format: results_{dataset}_alpha_{alpha}.csv
            model = 'vit4ts'
            dataset = '_'.join(parts[1:alpha_idx])  # dataset is between 'results' and 'alpha'

        # Read the file
        try:
            df = pd.read_csv(file)

            # Filter successful results
            if 'status' in df.columns:
                df_success = df[df['status'] == 'success'].copy()
            else:
                # If no status column, assume success? Or maybe older files differ
                df_success = df.copy()

            if df_success.empty:
                continue

            # Add metadata if not present (or overwrite to be consistent)
            df_success['dataset'] = dataset
            # Use 'model' column if present (from my updated script), else use parsed model
            if 'model' not in df_success.columns:
                df_success['model'] = model
            
            # Use 'alpha' column if present, else use parsed alpha
            if 'alpha' not in df_success.columns:
                df_success['alpha'] = alpha

            data.append(df_success)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not data:
        print("No successful results found.")
        return

    # Combine all data
    all_results = pd.concat(data, ignore_index=True)

    # Compute F1max for each signal across alphas
    # Group by dataset, model, and signal, then take max F1 across alphas
    # We group by 'dataset' and 'model' and 'signal'
    # Note: 'model' column in df distinguishes vit4ts vs vlm4ts even within same file
    f1max_per_signal = all_results.groupby(['dataset', 'model', 'signal'])['f1'].max().reset_index()
    f1max_per_signal.rename(columns={'f1': 'f1_max'}, inplace=True)

    # Compute average F1max per dataset and model
    summary = f1max_per_signal.groupby(['dataset', 'model'])['f1_max'].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ['Dataset', 'Model', 'Avg_F1max', 'Std_F1max', 'N_Signals']

    # Round for readability
    summary['Avg_F1max'] = summary['Avg_F1max'].round(4)
    summary['Std_F1max'] = summary['Std_F1max'].round(4)

    # Sort by dataset and model
    summary = summary.sort_values(['Dataset', 'Model'])
    
    # Print summary
    print("\n--- Summary Statistics (F1max) ---")
    print(summary.to_string(index=False))

    # Save summary
    summary_path = os.path.join(results_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    aggregate_results()
