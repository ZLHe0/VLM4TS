import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import argparse
import json

src_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, src_path)
os.chdir(src_path)
from evaluation_utils import *

def main(
    data_dir,
    dataset_name,
    file_list,
    n_windows=None,
    window_size=240,
    window_step_ratio=4.0,
    alpha_list=[0.001, 0.01, 0.1],
    evaluate_vlm=False,
    global_output=True,
    model_name="gpt"
):
    """
    Process each file by loading the CSV, computing window parameters, loading the final anomaly vector,
    aligning it with the full time series, and then generating detection intervals from the aligned anomaly vector.
    Finally, both the true anomaly intervals (if available) and the detected intervals are passed into the 
    PlotTimeSeriesWithAnomalies() function.
    
    Parameters
    ----------
    data_dir : str
        Directory where the raw CSV files and metadata are stored.
    dataset_name : str
        Name of the dataset to process.
    file_list : list of str
        List of file names (without extension) to process.
    n_windows : int
        Desired number of windows.
    window_step_ratio : float
        Ratio to compute window size.
    alpha : float, optional
        Upper quantile level (e.g., 0.05) for detection intervals.
    
    Returns
    -------
    None
    """
    if global_output:
        all_window_metrics = []
    for file_name in file_list:
        # Construct full file path and load the time series.
        file_path = os.path.join(data_dir, file_name + ".csv")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, file_name)
        ts_df = pd.read_csv(file_path)
        # For univariate, assume a column 'value'
        raw_series = ts_df['value'].tolist()
        time_series_proc = np.array(raw_series, dtype=float)
        if 'timestamp' in ts_df.columns:
            time_points_proc = ts_df['timestamp'].tolist()
        else:
            time_points_proc = np.arange(len(time_series_proc)).tolist()
        
        # Load true anomaly intervals (if available).
        anomalies_df = pd.read_csv(os.path.join(data_dir, 'anomalies.csv'))
        anomalies_df['events'] = anomalies_df['events'].apply(ast.literal_eval)
        anomaly_dict = dict(zip(anomalies_df['signal'], anomalies_df['events']))
        anomaly_intervals = anomaly_dict.get(file_name, None)
        
        # Convert intervals into indices form.
        interval_indices = []
        for interval in anomaly_intervals:
            start_idx = np.searchsorted(time_points_proc, interval[0], side='left')
            end_idx = np.searchsorted(time_points_proc, interval[1], side='right') - 1
            interval_indices.append((int(start_idx), int(end_idx)))
        anomaly_intervals = interval_indices

        # Determine windowing parameters.
        T_full = len(time_points_proc)

        # Determine windowing parameters.
        T_full = len(time_points_proc)
        if n_windows is not None:
            window_size = int(window_step_ratio * T_full / (window_step_ratio + n_windows - 1))
            step_size   = int(window_size / window_step_ratio)
            n_wins = n_windows
            print(f"Computed window_size: {window_size}, step_size: {step_size}")
        else:
            step_size = int(window_size / window_step_ratio)
            n_wins = int((T_full - window_size) / step_size) + 1
            print(f"Using provided window_size: {window_size}, step_size: {step_size}")
        
        # Load the final anomaly vector from file.
        results_dir = os.path.join("../results", dataset_name, file_name)

        if evaluate_vlm:
            # load the single JSON that VLM produced (one top‑level alpha key)
            vlm_path = os.path.join(results_dir, f"{file_name}_{model_name}_detections.json")
            with open(vlm_path, "r") as vf:
                vlm_out = json.load(vf)

            file_window_metrics = {}

            try:
                for alpha_str, info in vlm_out.items():
                    # ground‑truth intervals
                    true_fmt = format_intervals(anomaly_intervals)
                    # detected intervals from VLM
                    det_fmt  = format_intervals(info.get("interval_index", []))

                    # window‑wise records, via your helpers
                    win_rec   = make_window_eval_record(true_fmt, det_fmt)

                    # attach confidence if present
                    conf = info.get("confidence", [])

                    file_window_metrics[alpha_str] = {**win_rec,   "confidence": conf}
            except Exception as e:
                print(f"Error processing VLM output for {file_name}: {e}")
                continue

            # save per‑file
            with open(os.path.join(results_dir, f"{file_name}_{model_name}_metrics_windowwise.json"), "w") as f:
                json.dump(file_window_metrics, f, indent=4)

            if global_output:
                all_window_metrics.append(file_window_metrics)

            print("Saved VLM‑based evaluation metrics for", file_name)

        else:
            final_filename = os.path.join(results_dir, f"{file_name}_agg_allshot.npy")
            if not os.path.exists(final_filename):
                print(f"Vision result for '{file_name}' missing, skipping.")
                continue
            final_anomaly_vector = np.load(final_filename)
            
            # Align the final anomaly vector with the full time series.
            step_size = int(window_size / window_step_ratio)
            aligned_anomaly_vector = align_anomaly_vector(final_anomaly_vector, T_full, window_size, step_size, n_wins)
            
            eval_results_window = evaluate_detections_window(anomaly_intervals, aligned_anomaly_vector, alpha_list)

            with open(os.path.join(results_dir, f"{file_name}_vision_metrics_windowwise.json"), "w") as f:
                json.dump(eval_results_window, f, indent=4)

            if global_output:
                all_window_metrics.append(eval_results_window)

            print("Saved vision-based evaluation metrics for", file_name)

    if global_output:
        # Aggregate metrics across files for window‐wise evaluations.
        agg_window = aggregate_metrics(all_window_metrics)  # Summed metrics per alpha.
        
        n_files = len(all_window_metrics)  # Number of files processed.
        
        overall_window = {}
        pr_points_window = []
        for alpha, metrics in agg_window.items():
            # Compute the mean for each numeric metric.
            metrics = {k: metrics[k] for k in ("TP", "FP", "FN", "total_detected_length")}
            pr = compute_precision_recall_f1(metrics)  # returns dict with "precision", "recall", "F1", "F0.5"
            overall_window[alpha] = {**metrics, **pr, "threshold": metrics.get("threshold")}
            pr_points_window.append((pr["recall"], pr["precision"]))
            
        pr_auc_window = compute_pr_auc(pr_points_window)
        overall_window["PR_AUC"] = pr_auc_window

        # Save global aggregated metrics for both window.
        global_results_dir = os.path.join("../results", dataset_name)
        os.makedirs(global_results_dir, exist_ok=True)
        
        suffix = "vlm" if evaluate_vlm else "vision"
        window_fname = f"{suffix}_{model_name}_global_metrics_window.json"

        with open(os.path.join(global_results_dir, window_fname), "w") as f:
            json.dump(overall_window, f, indent=4)
        
        print("Global aggregated detection metrics saved for both window and point.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a dataset to generate windowed images for CLIP anomaly detection."
    )
    parser.add_argument("--data_dir", type=str, default="../data/raw/",
                        help="Directory of raw data (e.g., CSV files and metadata).")
    parser.add_argument("--results_base_dir", type=str, default="../results/",
                        help="Base directory where results will be saved.")
    parser.add_argument("--dataset_name", type=str, default="MSL",
                        help="Name of the dataset to process.")
    parser.add_argument("--file_list", type=str, nargs="+", default=None,
                        help="List of file names (without extension) to process. If not provided, all files from metadata will be used.")
    parser.add_argument("--transform_types", type=str, nargs="+", default=["line"],
                        help="List of transformation types to use (e.g., 'line', 'gaf').")
    parser.add_argument("--n_windows", type=int, default=None,
                        help="Desired number of windows. If not set, --window_size will be used.")
    parser.add_argument("--window_size", type=int, default=240,
                        help="Size of each window when --n_windows is not provided.")
    parser.add_argument("--window_step_ratio", type=float, default=4.0,
                        help="Window step ratio (window_size/step_size).")
    parser.add_argument(
        "--evaluate_vlm", action="store_true", default=False,
        help="If set, load VLM JSON results and compute window- and point-wise metrics from them."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt",
        help="Name of the model to use for evaluation. Options: 'gemini', 'gpt'."
    )

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

    main(
        args.data_dir,
        args.dataset_name,
        file_list,
        n_windows=args.n_windows,
        window_size=args.window_size,
        window_step_ratio=args.window_step_ratio,
        evaluate_vlm=args.evaluate_vlm,
        model_name=args.model_name
    )

