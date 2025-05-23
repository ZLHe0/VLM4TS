import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import argparse
import pdb
import json

src_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, src_path)
os.chdir(src_path)
from evaluation_utils import *

def main(data_dir, dataset_name, file_list, n_windows=None, window_size=240,
          window_step_ratio=4.0, alpha=None, save_subfigures=False,
          evaluate_vlm=False):
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
        results_dir   = os.path.join("../results", dataset_name, file_name)
        base_filename = os.path.join(results_dir, file_name)

        # prepare a list of plotting jobs: (aligned_vec, det_intervals, suffix)
        plot_jobs = []

        if not evaluate_vlm:
            # VISION-BASED branch: load & align
            final_path = os.path.join(results_dir, f"{file_name}_agg_allshot.npy")
            if not os.path.exists(final_path):
                print(f"Vision result for '{final_path}' missing, skipping.")
                continue
            final_vec = np.load(final_path)
            step_size = int(window_size / window_step_ratio)
            aligned_vec = align_anomaly_vector(final_vec, T_full, window_size, step_size, n_wins)
            # compute intervals, threshold, and get back (optionally smoothed) aligned_vec
            det_intervals, thresh, aligned_vec = compute_detection_intervals(aligned_vec, alpha)
            print(f"[{file_name}] vision  alpha={alpha} → thresh={thresh}, dets={det_intervals}")

            plot_jobs.append(
                (aligned_vec, det_intervals, f"_vision")
            )
        else:
            # VLM-BASED branch: load its JSON output
            vlm_path = os.path.join(results_dir, f"{file_name}_vlm_detections.json")
            if not os.path.exists(vlm_path):
                print(f"Vision result for '{vlm_path}' missing, skipping.")
                continue
            vlm_out = json.load(open(vlm_path))

            # iterate over each α key in that JSON
            for alpha_str, info in vlm_out.items():
                det_intervals = format_intervals(info.get("interval_index", []))
                print(f"[{file_name}] VLM  α={alpha_str} → dets={det_intervals}")

                plot_jobs.append(
                    (None, det_intervals, f"_vlm")
                )
                break # only plot for the first alpha (presumbly 0.1)

        # now do all plotting in one place
        for aligned_vec, det_intervals, suffix in plot_jobs:
            plot_time_series_with_anomalies(
                time_points_proc,
                time_series_proc,
                aligned_vec,
                anomaly_intervals,
                det_intervals,
                file_name,
                base_filename + suffix,
                show_labels=False,
                plot_indices=True,
                save_ts_original=False,
                dpi=100,
                max_width_px=1200,
                max_height_px=685
            )

            plot_time_series_with_anomalies(
                time_points_proc,
                time_series_proc,
                aligned_vec,
                anomaly_intervals,
                det_intervals,
                file_name,
                base_filename + suffix,
                show_labels=True,
                plot_indices=True,
                save_ts_original=False,
                dpi=100,
                max_width_px=1200,
                max_height_px=685
            )

        # Plot time series with true anomalies
        plot_time_series_with_anomalies(
            time_points_proc,
            time_series_proc,
            aligned_anomaly_vector=None,
            anomaly_intervals=anomaly_intervals,
            detection_intervals=None,
            file_name=file_name,
            base_filename=base_filename,
            show_labels=True,
            plot_indices=True,
            dpi=100,
            max_width_px=1200,
            max_height_px=685
        )

def plot_time_series_with_anomalies(
    time_points,                # array-like, shape (T,)
    time_series,                # array-like, shape (T,) for univariate (or multivariate handled separately)
    aligned_anomaly_vector,     # array-like, shape (T,)
    anomaly_intervals,          # list of [start, end] pairs (timestamps) for true anomaly, or None
    detection_intervals,        # list of [start, end] pairs (timestamps) for detected anomalies, or None
    file_name,                  # identifier for the series (displayed in title)
    base_filename,              # base path for saving the figure
    show_labels=True,
    plot_indices=True,
    save_subfigures=False,
    save_ts_with_window=True,
    save_ts_original=True,
    color="black",
    num_windows=5,
    dpi=100,
    max_width_px=1200,
    max_height_px=685
):
    """
    Plot a univariate time series along with its anomaly score.
    
    If show_labels is True and anomaly_intervals is provided, the true anomaly intervals
    are overlaid with the label "True Anomaly". If show_labels is False but detection_intervals
    is provided, the detected intervals are overlaid with the label "Detected Anomaly".
    
    The x-axis is determined by time_points. If plot_indices is True, x-axis is set as indices (0...T-1)
    and the intervals are converted accordingly.
    """
    T = len(time_points)
    # Determine x-axis and interval conversion.
    if plot_indices:
        x_axis = np.arange(T)
        if show_labels and anomaly_intervals is not None:
            interval_indices = []
            for interval in anomaly_intervals:
                start_idx = np.searchsorted(time_points, interval[0], side='left')
                end_idx = np.searchsorted(time_points, interval[1], side='right') - 1
                interval_indices.append((start_idx, end_idx))
            label_to_use = "True Anomaly"
        elif (not show_labels) and (detection_intervals is not None):
            interval_indices = detection_intervals # The detection intervals are already in index form
            # for interval in detection_intervals:
            #     start_idx = np.searchsorted(time_points, interval[0], side='left')
            #     end_idx = np.searchsorted(time_points, interval[1], side='right') - 1
            #     interval_indices.append((start_idx, end_idx))
            label_to_use = "Detected Anomaly"
        else:
            interval_indices = None
    else:
        x_axis = np.array(time_points)
        if show_labels and anomaly_intervals is not None:
            interval_indices = anomaly_intervals
            label_to_use = "True Anomaly"
        elif (not show_labels) and (detection_intervals is not None):
            interval_indices = detection_intervals
            label_to_use = "Detected Anomaly"
        else:
            interval_indices = None

    # Calculate figure size in inches based on desired pixel dimensions.
    fig_width = max_width_px / dpi
    fig_height = max_height_px / dpi

    # Set font sizes for readability.
    title_fontsize = max_width_px // 100
    label_fontsize = max_width_px // 100
    tick_fontsize = max_width_px // 120
    legend_fontsize = max_width_px // 120

    if aligned_anomaly_vector is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(fig_width, fig_height), dpi=dpi)

        # Plot the time series.
        ax1.plot(x_axis, time_series, label='Time Series')
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        
        if interval_indices is not None:
            first = True
            for (s, e) in interval_indices:
                if first:
                    ax1.axvspan(s, e, color='red', alpha=0.3, label=label_to_use)
                    first = False
                else:
                    ax1.axvspan(s, e, color='red', alpha=0.3)
        
        ax1.set_xlabel("Time", fontsize=label_fontsize)
        ax1.set_ylabel("Value", fontsize=label_fontsize)
        ax1.set_title(f"Time Series: {file_name}", fontsize=title_fontsize)
        ax1.legend(fontsize=legend_fontsize)

        # Plot the anomaly score.
        ax2.plot(x_axis, aligned_anomaly_vector, 'r-', linewidth=1.5, label='Anomaly Score')
        ax2.set_xlabel("Time", fontsize=label_fontsize)
        ax2.set_ylabel("Anomaly Score", fontsize=label_fontsize)
        ax2.set_title("Vision-based Anomaly Score", fontsize=title_fontsize)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.legend(fontsize=legend_fontsize)
        
        plt.tight_layout()

        save_path = f"{base_filename}_{'with_labels' if show_labels else 'raw'}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()

        print(f"Saved aligned score plot as {save_path}")

    if save_ts_with_window:
        # ----- Individual Figure for Time Series Panel -----
        fig1, ax = plt.subplots(figsize=(fig_width, fig_height/2), dpi=dpi)
        ax.plot(x_axis, time_series, label='Time Series', color=color)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        if interval_indices is not None:
            first = True
            for (s, e) in interval_indices:
                if first:
                    ax.axvspan(s, e, color='red', alpha=0.3, label=label_to_use)
                    first = False
                else:
                    ax.axvspan(s, e, color='red', alpha=0.3)
                    
        # Adjust ticks
        orig_locs = ax.get_xticks()
        if len(orig_locs) > 1:
            xmin, xmax = ax.get_xlim()
            orig_locs = orig_locs[(orig_locs >= xmin) & (orig_locs <= xmax)]
            mid_locs = (orig_locs[:-1] + orig_locs[1:]) / 2
            mid_locs = mid_locs[(mid_locs > xmin) & (mid_locs < xmax)]
            new_locs = np.sort(np.concatenate([orig_locs, mid_locs]))
            ax.set_xticks(new_locs)
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        ax.set_title(f"Time Series: {file_name}", fontsize=title_fontsize)
        ax.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        save_path1 = f"{base_filename}_timeseries_{'with_labels' if show_labels else 'with_detection'}.png"
        plt.savefig(save_path1, bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"Saved individual time series plot as {save_path1}")

    if save_ts_original:
        # ----- Individual Figure for Time Series Panel -----
        fig1, ax = plt.subplots(figsize=(fig_width, fig_height/2), dpi=dpi)
        ax.plot(x_axis, time_series, label='Time Series', color=color)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Adjust ticks
        orig_locs = ax.get_xticks()
        if len(orig_locs) > 1:
            xmin, xmax = ax.get_xlim()
            orig_locs = orig_locs[(orig_locs >= xmin) & (orig_locs <= xmax)]
            mid_locs = (orig_locs[:-1] + orig_locs[1:]) / 2
            mid_locs = mid_locs[(mid_locs > xmin) & (mid_locs < xmax)]
            new_locs = np.sort(np.concatenate([orig_locs, mid_locs]))
            ax.set_xticks(new_locs)
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        ax.set_title(f"Time Series: {file_name}", fontsize=title_fontsize)
        ax.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        save_path1 = f"{base_filename}_timeseries_raw.png"
        plt.savefig(save_path1, bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"Saved individual time series plot as {save_path1}")

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
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Anomaly threshold (upper quantile).")
    parser.add_argument("--save_subfigures", action="store_true",
                        help="Save overlapping subfigures.")
    parser.add_argument(
        "--evaluate_vlm", action="store_true", default=False,
        help="If set, load VLM JSON results and compute window- and point-wise metrics from them."
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
        alpha=args.alpha,
        save_subfigures=args.save_subfigures,
        evaluate_vlm=args.evaluate_vlm
    )