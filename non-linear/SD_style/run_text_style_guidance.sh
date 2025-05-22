#!/bin/bash

task="style"
methods=("DSG" "ours" "freedom")
prompt="A corgi wearing a wizard hat."

image_paths=(
  "./text_style_images/xingkong.jpg"
  "./text_style_images/xing.jpg"
  "./text_style_images/xiangrikui.jpg"
  "./text_style_images/tan.jpg"
  "./text_style_images/nahan.jpeg"
  "./text_style_images/jojo.jpeg"
  "./text_style_images/bw.jpg"
  "./text_style_images/bijiasuo.jpeg"
  "./text_style_images/1.png"
  "./text_style_images/brice-marden_6-red-rock-1-2002.jpg"
)
available_devices=(1 3 5 7)
scale=15.0
guidance_rate=0.2
outdir="outputs/txt2img-samples"

# create output directories for all methods
for method in "${methods[@]}"; do
    outdir_path="${outdir}/${method}_text_style_guidance"
    mkdir -p "$outdir_path"
    echo "Created output directory: $outdir_path"
done

run_task() {
    local image_path=$1
    local method=$2
    local cuda_device=$3

    echo "Running task with image: $image_path, method: $method, GPU: $cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python txt2img.py \
      --prompt "$prompt" \
      --style_ref_img_path "$image_path" \
      --ddim_steps 100 \
      --n_iter 1 \
      --n_samples 1 \
      --seed 42 \
      --H 512 \
      --W 512 \
      --scale "$scale" \
      --method "$method" \
      --task "$task" \
      --outdir "${outdir}/${method}_text_style_guidance" \
      --ddim_eta 1 \
      --guidance_rate "$guidance_rate"
}

declare -A device_pids
for device in "${available_devices[@]}"; do
    device_pids[$device]=""
done

declare -a task_queue
for method in "${methods[@]}"; do
    for image_path in "${image_paths[@]}"; do
        task_queue+=("$method:$image_path")
    done
done

task_index=0
total_tasks=${#task_queue[@]}
completed_tasks=0

echo "Starting text-style guidance for $total_tasks tasks..."

while [ $completed_tasks -lt $total_tasks ]; do
    for device in "${available_devices[@]}"; do
        if [ "${device_pids[$device]}" != "" ]; then
            if ! kill -0 ${device_pids[$device]} 2>/dev/null; then
                echo "Task on GPU $device completed"
                device_pids[$device]=""
                ((completed_tasks++))
            fi
        fi

        if [ "${device_pids[$device]}" == "" ] && [ $task_index -lt $total_tasks ]; then
            IFS=':' read -r method image_path <<< "${task_queue[$task_index]}"
            run_task "$image_path" "$method" "$device" &
            device_pids[$device]=$!
            echo "Assigned task $((task_index+1))/$total_tasks to GPU $device (PID: ${device_pids[$device]})"
            ((task_index++))
        fi
    done
    sleep 1
done

echo "All text-style guidance tasks completed successfully!"

