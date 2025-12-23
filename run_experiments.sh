#!/bin/bash

# Run experiments for multiple datasets across multiple alpha thresholds
echo "Starting VLM4TS experiments with 10 workers..."
echo "Start time: $(date)"

# Number of workers
N_WORKERS=10

# List of datasets to process
DATASETS=(
    "realTraffic"
    "realTweets"
    "SMAP"
    "MSL"
    "artificialWithAnomaly"
    "realAdExchange"
    "realAWSCloudwatch"
)

# Download all datasets if not already present
echo ""
echo "Checking and downloading datasets if needed..."
python src/preprocessing/download_data.py "${DATASETS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download datasets."
    exit 1
fi

echo "Dataset download/check complete."

# Run experiments for each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running $dataset (alpha=0.1, 0.01, 0.001) with VLM4TS..."
    echo "=========================================="

    python src/run_experiment.py "$dataset" --n_workers $N_WORKERS --model vlm4ts

    if [ $? -ne 0 ]; then
        echo "Warning: Experiment failed for $dataset"
    else
        echo "Completed $dataset successfully"
    fi
done

echo ""
echo "Aggregating results..."
python src/evaluation/aggregate_results.py

echo ""
echo "=========================================="
echo "All experiments completed."
echo "End time: $(date)"
echo "=========================================="
