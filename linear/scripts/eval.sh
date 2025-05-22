#!/bin/bash

# Define available CUDA devices
CUDA_DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")

# Define result and ground truth root pairs
declare -A RESULT_GT_PAIRS=(
    ["/data/dengwenhao/20250423_guidance_bench_1/Linear_Inverse_Problems/total_results_ours_full_DDIM100/ours_full_interval_1_guidance_0.1_n_guidance_directions_1_imagenet"]="./data/imagenet_256_subset@50"
    ["/data/dengwenhao/20250423_guidance_bench_1/Linear_Inverse_Problems/total_results_ours_full_DDIM100/ours_full_interval_1_guidance_0.2_n_guidance_directions_1_imagenet"]="./data/imagenet_256_subset@50"
)

# Create a temporary directory for storing intermediate results
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Function to run evaluation for a single pair
run_evaluation() {
    local result_root=$1
    local gt_root=$2
    local device=$3
    local output_file="$TEMP_DIR/$(basename $result_root)_metrics.json"
    local log_file="$TEMP_DIR/$(basename $result_root)_log.txt"
    
    echo "Running evaluation for $result_root on $device"
    # Run the evaluation and capture both stdout and stderr
    python evaluate_all_tasks.py --result_root "$result_root" --gt_root "$gt_root" --device "$device" > "$log_file" 2>&1
    
    # Check if the metrics file was created
    if [ -f "$result_root/metrics.json" ]; then
        cp "$result_root/metrics.json" "$output_file"
    else
        echo "Error: metrics.json not created for $result_root" > "$output_file"
    fi
}

# Get all pairs into an array
declare -a PAIRS
for result_root in "${!RESULT_GT_PAIRS[@]}"; do
    PAIRS+=("$result_root:${RESULT_GT_PAIRS[$result_root]}")
done

# Calculate number of rounds needed
num_pairs=${#PAIRS[@]}
num_devices=${#CUDA_DEVICES[@]}
num_rounds=$(( (num_pairs + num_devices - 1) / num_devices ))

# Run evaluations in round-robin fashion
for ((round=0; round<num_rounds; round++)); do
    echo "Starting round $((round + 1)) of $num_rounds"
    
    # Start parallel jobs for this round
    for ((i=0; i<num_devices; i++)); do
        pair_idx=$((round * num_devices + i))
        if [ $pair_idx -lt $num_pairs ]; then
            IFS=':' read -r result_root gt_root <<< "${PAIRS[$pair_idx]}"
            device=${CUDA_DEVICES[$i]}
            run_evaluation "$result_root" "$gt_root" "$device" &
        fi
    done
    
    # Wait for all jobs in this round to complete
    wait
done

# Collect and combine results using pandas
echo "Collecting and combining results..."
python - <<EOF
import pandas as pd
import json
import os
import glob

# Read all JSON files from temp directory
results = []
for json_file in glob.glob('$TEMP_DIR/*_metrics.json'):
    try:
        with open(json_file, 'r') as f:
            result = json.load(f)
            # Extract the result root from the filename
            result_root = os.path.basename(json_file).replace('_metrics.json', '')
            # Add result root to each task's metrics
            for task, metrics in result.items():
                metrics['result_root'] = result_root
                metrics['task'] = task
                results.append(metrics)
    except json.JSONDecodeError:
        print(f"Error reading {json_file}")
        continue

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('evaluation_results.csv', index=False)
print("Results saved to evaluation_results.csv")

# Print summary
print("\nSummary of results:")
print(df.to_string())
EOF

# Clean up temporary directory
rm -rf "$TEMP_DIR"
echo "Cleaned up temporary directory"

