"""
General-purpose script to download data from Sintel Orion S3 bucket.
Can download specific datasets or all available datasets.
"""

import os
import ast
import sys
import argparse
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
BASE_URL = "https://sintel-orion.s3.us-east-2.amazonaws.com"

def download_file(url, save_path):
    """Download a file from a URL to a local path."""
    if os.path.exists(save_path):
        return  # Skip if already exists

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets from Sintel Orion S3.")
    parser.add_argument("datasets", nargs="*", help="List of dataset names to download. If empty, downloads all.")
    args = parser.parse_args()

    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Check for datasets.csv
    datasets_path = os.path.join(DATA_DIR, 'datasets.csv')
    if not os.path.exists(datasets_path):
        print("Downloading datasets.csv...")
        download_file(f"{BASE_URL}/datasets.csv", datasets_path)

    # Check for anomalies.csv
    anomalies_path = os.path.join(DATA_DIR, 'anomalies.csv')
    if not os.path.exists(anomalies_path):
        print("Downloading anomalies.csv...")
        download_file(f"{BASE_URL}/anomalies.csv", anomalies_path)

    # Parse datasets.csv
    with open(datasets_path, 'r') as f:
        lines = f.readlines()

    target_datasets = set(args.datasets) if args.datasets else None

    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
            
        dataset_name = parts[0]
        
        # Filter if targets are specified
        if target_datasets and dataset_name not in target_datasets:
            continue
            
        # Extract signal names from the tuple string
        try:
            signals = ast.literal_eval(parts[1].strip('"'))
        except Exception as e:
            print(f"Error parsing signals for {dataset_name}: {e}")
            continue

        print(f"Downloading {len(signals)} signals for {dataset_name}...")
        
        # Create dataset directory
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Download each signal
        for signal in tqdm(signals):
            file_name = f"{signal}.csv"
            # Files are in the root of the bucket
            url = f"{BASE_URL}/{file_name}"
            save_path = os.path.join(dataset_dir, file_name)
            
            download_file(url, save_path)

    print("Download complete.")

if __name__ == "__main__":
    main()