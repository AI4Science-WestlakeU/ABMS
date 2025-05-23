#!/bin/bash

# face
method="ours" # dsg, freedom, ours
guidance_rate=1.0
rho_scale=100
process_start=0
process_end=999

# Retry settings
max_retries=3
retry_delay=5

# Available GPUs
available_cuda=(0 1 2 3 4 5 6 7)
num_gpus=${#available_cuda[@]}

# Image directory - using absolute path
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
image_dir="${script_dir}/images/ffhq_256_subset@1k"

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

# Verify process_start and process_end are valid
if [ $process_start -lt 0 ] || [ $process_end -ge ${#image_list[@]} ] || [ $process_start -gt $process_end ]; then
    echo "Error: Invalid process_start ($process_start) or process_end ($process_end)"
    echo "Valid range is 0 to $((${#image_list[@]} - 1))"
    exit 1
fi

total_images=$((process_end - process_start + 1))
echo "Processing images from index $process_start to $process_end (total: $total_images images)"

# Create a temporary file to track processed images
temp_file=$(mktemp)
echo "$process_start" > "$temp_file"

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
    if [ $current -le $process_end ]; then
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
