#!/bin/bash
# Please make sure that you have run run_ViT4TS.sh
# Start the job
echo " "
echo "Running VLM4TS started at $(date)"
echo " "

# datasets=("realAdExchange", "realTweets" "realTraffic" "realAWSCloudwatch"  "artificialWithAnomaly")
# datasets=("YAHOOA1" "YAHOOA2" "YAHOOA3" "YAHOOA4")
# datasets=("MSL" "SMAP")
# Loop through each alpha
alpha_list=(0.1 0.01 0.001)
# Loop through each dataset
for dataset in "${datasets[@]}"
do
    echo "Processing $dataset dataset..."
    cd models
    python VLM_inference.py --dataset_name "$dataset" --alpha_list "${alpha_list[@]}"
    cd ../evaluation
    python evaluation.py --dataset_name "$dataset" --evaluate_vlm
    cd ..
    echo "$dataset processing completed."
done