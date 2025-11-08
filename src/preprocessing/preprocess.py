import os
import sys
import ast
import random
import pandas as pd
import numpy as np
import numpy as np
from scipy.signal import detrend
import argparse
import warnings
import math
from fractions import Fraction
import pdb

src_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, src_path)
os.chdir(src_path)
from preprocessing.draw_image import draw_windowed_images

def preprocess_time_series(time_series):
    """
    Preprocess a time series by detrending and min–max standardization.
    
    Parameters
    ----------
    time_series : array-like
        The raw time series values.
    
    Returns
    -------
    preprocessed_series : np.ndarray
        The detrended and min–max normalized time series.
    """
    # Convert to a NumPy array (in case it's not already).
    ts = np.array(time_series, dtype=float)
    
    # Detrend the time series.
    ts_detrended = detrend(ts)
    
    # Min–max normalization.
    ts_min = np.min(ts_detrended)
    ts_max = np.max(ts_detrended)
    if ts_max - ts_min > 0:
        ts_normalized = (ts_detrended - ts_min) / (ts_max - ts_min)
    else:
        # In case the series is constant after detrending, return zeros.
        ts_normalized = np.zeros_like(ts_detrended)
    
    return ts_normalized


def extend_time_series(time_series, extra_length):
    """
    Extend a 1D time series by a given number of extra time points using linear extrapolation.

    Parameters
    ----------
    time_series : np.ndarray
         1D array of time series values.
    extra_length : int
         The number of additional data points to generate by extrapolation.

    Returns
    -------
    extended_series : np.ndarray
         The extended time series (original values followed by the extrapolated values).
    """
    T_orig = len(time_series)
    if extra_length <= 0:
        return time_series.copy()
    
    # Compute slope from the last two points (if available)
    if T_orig >= 2:
        slope = time_series[-1] - time_series[-2]
    else:
        slope = 0.0
    
    # Generate extra_length points using linear extrapolation
    extra_points = time_series[-1] + slope * np.arange(1, extra_length + 1)
    
    # Concatenate original series with the extra points.
    extended_series = np.concatenate([time_series, extra_points])
    return extended_series

    

def process_dataset(dataset_name, data_dir, results_base_dir, transform_types,
                    n_windows=None, window_step_ratio=4, window_size=None,
                    plot_step_size=None, file_list=None, override=True, dpi=100,
                    image_size=(240,240),
                    plot_params=('-', 1, '*', 2, 'black', None), save_image=False,
                    use_count_as_timepoints=True, standardization=True):
    """
    Process a dataset by drawing windowed images for each time series file.

    For each file, the time series is loaded and windowed using a sliding window.
    The CSV is assumed to have columns: 'value' and 'timestamp'.

    Time points, window sizes, and anomaly intervals are determined from the CSV data.
    The processed time series (shape [T]) is fed into draw_windowed_images() to produce windowed images.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to process (e.g. 'artificialWithAnomaly').
    data_dir : str
        Directory where the raw CSV files are stored.
    results_base_dir : str
        Base directory where the results will be saved.
    transform_types : list of str
        List of transformation types to use (e.g., ['line', 'gaf']).
    n_windows : int or None, optional
        If provided, automatically compute window and step sizes to yield ~n_windows windows.
    window_size : int or None, optional
        Fixed window size (used if n_windows is None).
    plot_step_size : int or None, optional
        Fixed step size (used if n_windows is None).
    file_list : list of str, optional
        List of file names to process. If None, datasets_multivariate.csv must exist to load file list.
    override : bool, optional
        Whether to overwrite existing files (default: True).
    dpi : int, optional
        Dots per inch for the saved images.
    image_size : tuple, optional
        Output image size in pixels (height, width); default is (240,240).
    plot_params : tuple, optional
        Plot styling parameters for a line plot: (linestyle, linewidth, marker, markersize, color, y_scale).
    use_count_as_timepoints : bool, optional
        If True, ignore the 'timestamp' column and use a count (0 to T-1) as time points.
    standardization : bool, optional
        If True, perform detrending and min–max standardization.

    Returns
    -------
    None
    """
    # If file_list is not provided, load it from the datasets metadata.
    if file_list is None:
        meta_path = os.path.join(data_dir, 'datasets_multivariate.csv')
        if not os.path.exists(meta_path):
            print(f"Error: datasets_multivariate.csv not found at {meta_path} and no file_list provided.")
            return
        datasets_meta = pd.read_csv(meta_path, header=None, names=['dataset', 'files'])

        # Filter for the specified dataset.
        dataset_meta = datasets_meta[datasets_meta['dataset'].str.strip() == dataset_name]
        if dataset_meta.empty:
            print(f"No metadata found for '{dataset_name}' dataset.")
            return

        file_list = ast.literal_eval(dataset_meta.iloc[0]['files'])

    # Process each file in file_list.
    for file_name in file_list:
        # Define a results directory for this file.
        file_results_dir = os.path.join(results_base_dir, dataset_name, file_name)
        if not os.path.exists(file_results_dir):
            os.makedirs(file_results_dir)
        
        # Construct the file path.
        file_path = os.path.join(data_dir, file_name + '.csv')
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} not found, skipping...")
            continue
        
        # Load the time series CSV file.
        try:
            ts_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        if 'value' not in ts_df.columns:
            print(f"Column 'value' not found in {file_name}. Skipping...")
            continue
        raw_series = ts_df['value'].tolist()
        if standardization:
            time_series_proc = preprocess_time_series(raw_series)
            plot_params_list = list(plot_params)
            plot_params_list[-1] = (0, 1)
            plot_params_mod = tuple(plot_params_list)
        else:
            time_series_proc = np.array(raw_series, dtype=float)
            plot_params_mod = plot_params
        time_points_proc = (np.arange(len(time_series_proc)).tolist() if use_count_as_timepoints 
                else ts_df['timestamp'].tolist())
        
        # Use the file name (without extension) as the base series ID.
        base_series_id = os.path.splitext(file_name)[0]
        print(f"Processing {file_name} with {len(time_series_proc)} data points...")
        
        # Determine window and step sizes.
        T = len(time_series_proc)
        S = window_step_ratio
        if n_windows is not None:
            # calculate both window and step sizes to fit exactly n_windows
            window_size = int(S * T / (S + n_windows - 1))
            step_size   = int(window_size / S)
            covered_length = window_size + (n_windows - 1) * step_size
            print(f"Calculated window size: {window_size}, step size: {step_size} for {n_windows} windows, covering {covered_length} data points.")
        else:
            # derive step size from the given window_size and ratio S
            step_size = int(window_size / S)
            print(f"Using window size: {window_size}, derived step size: {step_size} based on window-step ratio {S}.")
        
        # --- Load anomaly intervals ---
        anomalies_csv = os.path.join(data_dir, 'anomalies.csv')
        if os.path.exists(anomalies_csv):
            anomalies_df = pd.read_csv(anomalies_csv)
            anomalies_df['events'] = anomalies_df['events'].apply(ast.literal_eval)
            anomaly_dict = dict(zip(anomalies_df['signal'], anomalies_df['events']))
            anomaly_intervals = anomaly_dict.get(file_name, None)
        else:
            anomaly_intervals = None
        
        # Process anomaly intervals.
        if anomaly_intervals is not None and use_count_as_timepoints:
            if 'timestamp' in ts_df.columns:
                original_timestamps = ts_df['timestamp'].tolist()
                anomaly_intervals_proc = []
                for interval in anomaly_intervals:
                    start_time, end_time = interval[0], interval[1]
                    start_idx = np.searchsorted(original_timestamps, start_time, side='left')
                    end_idx = np.searchsorted(original_timestamps, end_time, side='right') - 1
                    anomaly_intervals_proc.append((start_idx, end_idx))
            else:
                anomaly_intervals_proc = anomaly_intervals
        else:
            anomaly_intervals_proc = anomaly_intervals
        
        # --- Draw windowed images ---
        for transform in transform_types:
            draw_windowed_images(
                base_series_id=base_series_id,
                save_path=file_results_dir,
                time_series=time_series_proc,
                time_points=time_points_proc,
                anomaly_intervals=anomaly_intervals_proc,
                window_size=window_size,
                step_size=step_size,
                override=override,
                save_image=save_image,
                image_size=image_size,
                dpi=dpi,
                plot_params=plot_params_mod,
                transform_type=transform
            )
    return True

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
                        help="Desired number of windows. If not provided, uses --window_size.")
    parser.add_argument("--window_size", type=int, default=240,
                        help="Size of each window when --n_windows is not set.")
    parser.add_argument("--window_step_ratio", type=float, default=4.0,
                        help="Window step ratio (window_size/step_size).")
    parser.add_argument("--save_image", action="store_true", default=False,
                        help="If set, save the generated images.")

    args = parser.parse_args()

    # If file_list is not provided, load it from the metadata in datasets_multivariate.csv.
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

    # Call process_dataset with the provided parameters.
    process_dataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        results_base_dir=args.results_base_dir,
        transform_types=args.transform_types,
        n_windows=args.n_windows,
        window_size=args.window_size,
        window_step_ratio=args.window_step_ratio,
        file_list=file_list,
        override=True,
        dpi=100,
        image_size=(240, 240),
        plot_params=('-', 1, '*', 0.1, 'black', None), 
        use_count_as_timepoints=True,
        standardization=True,
        save_image=args.save_image
    )