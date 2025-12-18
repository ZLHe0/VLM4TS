#!/bin/bash

# Run experiments for NASA datasets across multiple alpha thresholds
echo "Starting NASA experiments..."

echo "Running SMAP (alpha=0.1, 0.01, 0.001)..."
python src/run_experiment.py SMAP

echo "Running MSL (alpha=0.1, 0.01, 0.001)..."
python src/run_experiment.py MSL

echo "Experiments completed."