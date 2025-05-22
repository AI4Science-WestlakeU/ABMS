#! /bin/bash

real_folder=./images/celeb-a-hq_subset@1k
gen_folder=./exp/image_samples/arcface_face_ours_t@100_stop@100_rho@100.0_guidance@0.08
available_gpu_ids=(0 1 2 3 4 5 6 7)
faceid_results_dir=$gen_folder/faceid_loss_results

# Calculate number of available GPUs
num_gpus=${#available_gpu_ids[@]}

# Validate input folders
if [ ! -d "$real_folder" ]; then
    echo "Error: Real folder $real_folder does not exist"
    exit 1
fi

if [ ! -d "$gen_folder" ]; then
    echo "Error: Generated folder $gen_folder does not exist"
    exit 1
fi

# Create results directory
mkdir -p $faceid_results_dir

# Get list of image pairs
real_images=($(ls $real_folder/*.png))
gen_images=($(ls $gen_folder/*.png))

# Validate that we have matching numbers of images
if [ ${#real_images[@]} -ne ${#gen_images[@]} ]; then
    echo "Error: Number of real images (${#real_images[@]}) does not match number of generated images (${#gen_images[@]})"
    exit 1
fi

# Process image pairs in parallel
for i in "${!real_images[@]}"; do
    real_path="${real_images[$i]}"
    gen_path="${gen_images[$i]}"
    
    # Assign GPU ID in round-robin fashion
    gpu_id=${available_gpu_ids[$((i % num_gpus))]}
    
    echo "Evaluating FaceID loss for image pair $real_path and $gen_path on GPU $gpu_id"

    # Run FaceID evaluation in background
    export CUDA_VISIBLE_DEVICES=$gpu_id && python evaluate_faceid_loss.py \
        --real_path "$real_path" \
        --gen_path "$gen_path" \
        --output_dir "$faceid_results_dir" \
        --gpu_id 0 &
    
    # Limit number of parallel processes
    if (( (i + 1) % num_gpus == 0 )); then
        wait
    fi
done

# Wait for remaining processes
wait

# Aggregate FaceID results
echo "Aggregating FaceID results..."
python -c "
import os
import json
import sys
from evaluate_faceid_loss import aggregate_results

results_dir = '$faceid_results_dir'
mean_norm = aggregate_results(results_dir)
print(f'\nFaceID Evaluation Results:')
print(f'Mean FaceID Score: {mean_norm:.4f}')
"

# run FID/KID evaluation
echo "Running FID/KID evaluation..."
python evaluate_fid_kid.py --real_folder $real_folder --gen_folder $gen_folder