"""
General-purpose script to run ViT4TS on a specified dataset and evaluate performance across multiple alpha thresholds.
Usage: python src/run_experiment.py <DatasetName> [--n_workers N] [--model {vit4ts,vlm4ts}]
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import multiprocessing
from functools import partial
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.vit4ts import ViT4TS
from models.vlm4ts import VLM4TS
from evaluation.evaluate import evaluate_intervals

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Global variables for worker processes
global_detector = None
global_ground_truth = None
global_model_type = None

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

def init_worker(ground_truth, model_type):
    """Initialize global variables in worker process."""
    global global_detector
    global global_ground_truth
    global global_model_type
    
    global_ground_truth = ground_truth
    global_model_type = model_type
    
    # Initialize detector
    try:
        if model_type == 'vit4ts':
            global_detector = ViT4TS(
                window_size=240,
                window_step_ratio=4.0,
                model_name='ViT-B-16',
                image_size=(224, 224),
                alpha=0.01,
                verbose=False
            )
        elif model_type == 'vlm4ts':
            # Check for API key
            if not os.getenv('OPENAI_API_KEY'):
                logger.warning("OPENAI_API_KEY not set. VLM4TS may fail if key is not provided otherwise.")
                
            global_detector = VLM4TS(
                vit4ts_params={
                    'window_size': 240,
                    'window_step_ratio': 4.0,
                    'model_name': 'ViT-B-16',
                    'image_size': (224, 224),
                    'alpha': 0.01,
                    'verbose': False
                },
                alpha=0.01,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error initializing detector ({model_type}) in worker: {e}")

def process_file_task(args):
    """
    Worker task to process a single file. 
    
    args: (dataset_name, file_name, alphas)
    """
    dataset_name, file_name, alphas = args
    
    # Reconstruct paths
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    file_path = os.path.join(dataset_dir, file_name)
    signal_name = file_name.replace('.csv', '')
    
    logger.info(f"Starting processing for signal: {signal_name}")
    
    results = []
    
    # Load data
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return [{
            'dataset': dataset_name,
            'signal': signal_name,
            'model': global_model_type,
            'alpha': a,
            'status': 'failed',
            'error': f"Read error: {str(e)}"
        } for a in alphas]
        
    if data.empty:
        return []

    # Run Inference (Once)
    scores = None
    timestamps = None
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # For VLM4TS, we access the underlying vit4ts to compute scores
            if global_model_type == 'vlm4ts':
                scores, timestamps = global_detector.vit4ts.predict_scores(data)
            else:
                scores, timestamps = global_detector.predict_scores(data)
    except Exception as e:
        return [{
            'dataset': dataset_name,
            'signal': signal_name,
            'model': global_model_type,
            'alpha': a,
            'status': 'failed',
            'error': f"Inference error: {str(e)}"
        } for a in alphas]
        
    # Evaluate for each alpha
    for alpha in alphas:
        try:
            # 1. Get initial intervals (ViT4TS screening)
            vit_intervals = None
            if global_model_type == 'vlm4ts':
                vit_intervals = global_detector.vit4ts.get_intervals(scores, timestamps, alpha=alpha)
                # 2. Verify with VLM
                vlm_intervals = global_detector.verify_intervals(data, vit_intervals)
                # We will process both below
                methods_to_eval = [('vit4ts', vit_intervals), ('vlm4ts', vlm_intervals)]
            else:
                vit_intervals = global_detector.get_intervals(scores, timestamps, alpha=alpha)
                methods_to_eval = [('vit4ts', vit_intervals)]
            
            for model_name, intervals in methods_to_eval:
                if signal_name in global_ground_truth:
                    gt_intervals = global_ground_truth[signal_name]
                    detected_intervals = intervals[['start', 'end']].values.tolist()
                    
                    metrics = evaluate_intervals(gt_intervals, detected_intervals)
                    
                    results.append({
                        'dataset': dataset_name,
                        'signal': signal_name,
                        'model': model_name,
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
                        'model': model_name,
                        'alpha': alpha,
                        'status': 'no_gt',
                        'n_detected': len(intervals)
                    })
        except Exception as e:
            results.append({
                'dataset': dataset_name,
                'signal': signal_name,
                'model': global_model_type,
                'alpha': alpha,
                'status': 'failed',
                'error': f"Eval error: {str(e)}"
            })
            
    return results

def run_experiment(dataset_name, n_workers=1, model_type='vit4ts'):
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
    
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    print(f"Processing {dataset_name} ({len(files)} signals) with {n_workers} workers using {model_type}...")
    
    # Prepare tasks
    tasks = [(dataset_name, f, alphas) for f in files]
    
    # Run parallel processing
    all_results = []
    
    if n_workers > 1:
        # Use multiprocessing
        # Note: 'spawn' is default on MacOS, 'fork' on Linux. 
        # CUDA requires 'spawn'. 
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=init_worker, initargs=(ground_truth, model_type)) as pool:
            # Use imap_unordered for better tqdm responsiveness
            for res_list in tqdm(pool.imap_unordered(process_file_task, tasks), total=len(tasks)):
                all_results.extend(res_list)
    else:
        # Run sequentially
        init_worker(ground_truth, model_type)
        for task in tqdm(tasks):
            res_list = process_file_task(task)
            all_results.extend(res_list)
            
    # Organize and save results by alpha
    if not all_results:
        print("No results generated.")
        return

    df_all = pd.DataFrame(all_results)
    
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for alpha in alphas:
        print(f"\n--- Results for alpha={alpha} ---")
        results_file = os.path.join(results_dir, f'results_{dataset_name}_{model_type}_alpha_{alpha}.csv')
        
        df_alpha = df_all[df_all['alpha'] == alpha]
        
        if not df_alpha.empty:
            df_alpha.to_csv(results_file, index=False)
            print(f"Saved to {results_file}")
            
            # Print summary
            success_df = df_alpha[df_alpha['status'] == 'success']
            if not success_df.empty:
                print(f"Performance Summary (alpha={alpha}):")
                print(success_df[['precision', 'recall', 'f1']].mean())
        else:
            print(f"No results for alpha={alpha}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment on a specific dataset across multiple alphas.")
    parser.add_argument("dataset", help="Name of the dataset (e.g., SMAP, MSL)")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--model", type=str, default='vit4ts', choices=['vit4ts', 'vlm4ts'], help="Model to use (default: vit4ts)")
    args = parser.parse_args()
    
    # Set start method for multiprocessing to spawn to be safe with CUDA/PyTorch
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    run_experiment(args.dataset, args.n_workers, args.model)