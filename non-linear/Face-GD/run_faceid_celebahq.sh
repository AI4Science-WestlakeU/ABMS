#!/bin/bash

# face
method="ours" # dsg, freedom, ours
guidance_rate=0.04
rho_scale=100
process_start=0
process_end=999

# Retry settings
max_retries=3
retry_delay=5

# Available GPUs
available_cuda=(1 2 3 4 5 6)
num_gpus=${#available_cuda[@]}

# Image directory - using absolute path
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
image_dir="${script_dir}/images/celeb-a-hq_subset@1k"

# Verify image directory exists
if [ ! -d "$image_dir" ]; then
    echo "Error: Image directory $image_dir does not exist"
    exit 1
fi

# Get list of all images
image_list=($(ls ${image_dir}/*.png))
if [ ${#image_list[@]} -eq 0 ]; then
    echo "Error: No PNG images found in $image_dir"
    exit 1
fi

total_images=${#image_list[@]}
echo "Total images to process: $total_images"

# Create a temporary file to track processed images
temp_file=$(mktemp)
echo "0" > "$temp_file"

# Function to process a single image
process_image() {
    local gpu_id=$1
    local image_path=$2
    local retry_count=0
    
    echo "Processing $image_path on GPU $gpu_id"
    
    while [ $retry_count -lt $max_retries ]; do
        # Create a temporary file for error output
        local error_file=$(mktemp)
        
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py -i arcface_face -s arc_ddim --doc celeba_hq \
            --timesteps 100 --rho_scale $rho_scale --guidance_rate $guidance_rate --seed 1234 --stop 100 \
            --ref_path "$image_path" --batch_size 1 --gpu 0 --method "$method" 2> "$error_file"
            
        # Check if error file contains RuntimeError
        if grep -q "RuntimeError" "$error_file"; then
            echo "RuntimeError processing $image_path on GPU $gpu_id (attempt $((retry_count + 1))/$max_retries):"
            cat "$error_file"
            retry_count=$((retry_count + 1))
        else
            # If no RuntimeError, consider it successful
            echo "Successfully processed $image_path on GPU $gpu_id"
            rm "$error_file"
            return 0
        fi
        
        rm "$error_file"
        
        if [ $retry_count -lt $max_retries ]; then
            echo "Retrying in $retry_delay seconds..."
            sleep $retry_delay
        fi
    done
    
    echo "Failed to process $image_path after $max_retries attempts"
    return 1
}

# Function to get next image index
get_next_image() {
    local current=$(cat "$temp_file")
    if [ $current -lt $total_images ]; then
        echo $((current + 1)) > "$temp_file"
        echo $current
    else
        echo -1
    fi
}

# Main processing loop
while true; do
    # Launch one process per GPU
    for gpu_idx in $(seq 0 $((num_gpus-1))); do
        gpu_id=${available_cuda[$gpu_idx]}
        image_idx=$(get_next_image)
        
        if [ $image_idx -eq -1 ]; then
            # No more images to process
            break 2
        fi
        
        process_image $gpu_id "${image_list[$image_idx]}" &
    done
    
    # Wait for current batch to complete
    wait
done

# Clean up
rm "$temp_file"

echo "All processing complete!"
