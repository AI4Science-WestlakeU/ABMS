#!/bin/bash

task="style"
image_paths=("./style_images/137_le_village.jpg" "./style_images/198_clouds_over_bor.jpg" "./style_images/39_vertical_and_horizontal_composition.jpg" "./style_images/81_free_&_easy.jpg" "./style_images/99_Machiner.jpg" "./style_images/333_Irregular_Horizontal_Bands_of_Equal_Width_Starting_at_Bottom.jpg" "./style_images/376_two_women_with_the_armchair.jpg")

methods=("DSG" "ours" "freedom")
available_devices=(1 3 5 7)
# scale=20.0
# guidance_rate=0.5
scale=10.0
guidance_rate=0.2

outdir="outputs/txt2img-samples/"

# create output directory for all tasks
for method in "${methods[@]}"; do
    outdir_path="$outdir/${method}_style_guidance"
    mkdir -p "$outdir_path"
    echo "Created output directory: $outdir_path"
done

# Function to run a single task with the specified parameters
run_task() {
    local image_path=$1
    local method=$2
    local cuda_device=$3
    
    echo "Running task with image: $image_path, method: $method, GPU: $cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "$image_path" --ddim_steps 100 --n_iter 1 --n_samples 1 \
    --H 512 --W 512 --scale "$scale" --method "$method" --task "$task" --outdir "$outdir/${method}_style_guidance" --ddim_eta 1 --uncondition --guidance_rate "$guidance_rate"
}


# Create an array to hold processes for each device
declare -A device_pids
for device in "${available_devices[@]}"; do
    device_pids[$device]=""
done

# Create a queue of all tasks
declare -a task_queue
for method in "${methods[@]}"; do
    for image_path in "${image_paths[@]}"; do
        task_queue+=("$method:$image_path")
    done
done

# Process the task queue
task_index=0
total_tasks=${#task_queue[@]}
completed_tasks=0

echo "Starting processing of $total_tasks tasks..."

while [ $completed_tasks -lt $total_tasks ]; do
    # Check if any device is free
    for device in "${available_devices[@]}"; do
        # If device has a process ID, check if it's still running
        if [ "${device_pids[$device]}" != "" ]; then
            if ! kill -0 ${device_pids[$device]} 2>/dev/null; then
                # Process completed
                echo "Task on GPU $device completed"
                device_pids[$device]=""
                ((completed_tasks++))
            fi
        fi
        
        # If device is free and there are tasks remaining, assign a new task
        if [ "${device_pids[$device]}" == "" ] && [ $task_index -lt $total_tasks ]; then
            # Parse the task from the queue
            IFS=':' read -r method image_path <<< "${task_queue[$task_index]}"
            
            # Run the task in the background
            run_task "$image_path" "$method" "$device" &
            
            # Save the background process ID
            device_pids[$device]=$!
            
            echo "Assigned task $((task_index+1))/$total_tasks to GPU $device (PID: ${device_pids[$device]})"
            
            # Move to the next task
            ((task_index++))
        fi
    done
    
    # Short sleep to avoid excessive CPU usage
    sleep 1
done

echo "All tasks completed successfully!"