#!/bin/bash
# Start the job
echo " "
echo "Running ViT4TS started at $(date)"
echo " "

# datasets=("realAdExchange", "realTweets" "realTraffic" "realAWSCloudwatch"  "artificialWithAnomaly")
# datasets=("YAHOOA1" "YAHOOA2" "YAHOOA3" "YAHOOA4")
# datasets=("MSL" "SMAP")
# Loop through each dataset
for dataset in "${datasets[@]}"
do
    echo "Processing $dataset dataset..."
    # Preprocessing step
    cd preprocessing
    python preprocess.py --dataset_name "$dataset"
    # Model inference step
    cd ../models
    python vision_inference.py --dataset_name "$dataset"
    # Visualization step
    cd ../evaluation
    python visu
    alization.py --dataset_name "$dataset"
    python evaluation.py --dataset_name "$dataset"
    cd ..
    echo "$dataset processing completed."
done