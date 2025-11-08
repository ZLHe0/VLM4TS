import os
import argparse
import ast
import pandas as pd

import sys
src_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, src_path)
os.chdir(src_path)

import psutil
import subprocess

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main script for inference."
    )

    parser.add_argument("--data_dir", type=str, default="../data/raw/",
                        help="Directory of raw data (e.g., CSV files and metadata).")
    parser.add_argument("--dataset_name", type=str, default="MSL",
                        help="Name of the dataset (default: 'MSL').")
    parser.add_argument("--file_list", type=str, nargs="+", default=None,
                        help="List of file names (without extension) to process. If not provided, all files from metadata will be used.")
    parser.add_argument("--window_step_ratio", type=float, default=4.0,
                        help="Window step ratio (default: 4).")
    parser.add_argument("--agg_percent", type=float, default=0.25,
                        help="Aggregation percentage for anomaly map reduction (default: 0.10).")
    parser.add_argument("--no_anomaly", action="store_true", default=False,
                        help="Flag indicating that no anomaly labels are available (default: False).")
    parser.add_argument("--plot_types", type=str, nargs="+", default=["line"],
                        help="List of plot types (e.g., 'line', 'gaf'; default: ['line']).")
    parser.add_argument("--n_shot", type=str, default="all",
                        help="Number of shots ('all' or a number as string; default: 'all').")

    args = parser.parse_args()

    # If file_list is not provided, load it from the metadata in datasets.csv.
    if args.file_list is None:
        meta_path = os.path.join(args.data_dir, "datasets_multivariate.csv")
        datasets_meta = pd.read_csv(meta_path, header=None, names=['dataset', 'files'])
        dataset_meta = datasets_meta[datasets_meta['dataset'].str.strip() == args.dataset_name]
        if dataset_meta.empty:
            print(f"No metadata found for dataset: {args.dataset_name}")
            exit(1)
        file_list = ast.literal_eval(dataset_meta.iloc[0]['files'])
    else:
        file_list = args.file_list

    # Call process_dataset with the provided parameters (other parameters are set to defaults).
    for file_name in file_list:
        print(f"[Before loading {file_name}] Memory: {memory_usage_mb():.2f} MB")
        cmd = [
            "python", "models/vision_inference_subproc.py",
            "--data_dir",         args.data_dir,
            "--dataset_name",     args.dataset_name,
            "--file_name",        file_name,
            "--window_step_ratio", str(args.window_step_ratio),
            "--agg_percent",      str(args.agg_percent),
            "--plot_types",       *args.plot_types,
            "--n_shot",           args.n_shot,
            "--no_anomaly"        if args.no_anomaly else ""
        ]
        cmd = [c for c in cmd if c]

        print(f"Running subprocess for {file_name}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Processing failed for {file_name}")
        else:
            print(f"Processing completed for {file_name}")
        print(f"[After processing {file_name}] Memory: {memory_usage_mb():.2f} MB")





