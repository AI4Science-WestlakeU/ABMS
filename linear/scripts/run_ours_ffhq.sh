#!/bin/bash


#1. Choose The Task
#task_config=("configs/super_resolution_config.yaml" "configs/inpainting_config.yaml" "configs/gaussian_deblur_config.yaml")
task_config=("configs/inpainting_config.yaml")

#2. Choose the root of dataset
data_root="./data/ffhq_256_subset@1k"

# Set DATA_ROOT environment variable for all subsequent commands
export DATA_ROOT="$data_root"

#3. Choose the hyperparameters
guidance_scales=(0.3)
intervals=(1)

#4. Choose the DDIM steps and available GPUs
DDIM=100
GPUS=(5 6 7)  # List your available GPUs here

#5. Choose the diffusion model config
config="model_config.yaml"
#config="imagenet_model_config.yaml"

save_root="total_results_ours_DDIM"$DDIM

# Function to run a single job
run_job() {
  local interval=$1
  local guidance_scale=$2
  local yaml_file=$3
  local gpu=$4
  local save_dir="./${save_root}/ours_interval_${interval}_guidance_${guidance_scale}_ffhq"
  CUDA_VISIBLE_DEVICES=$gpu python sample_condition_same_inputs.py \
    --model_config configs/"$config" \
    --diffusion_config configs/diffusion_ddim"${DDIM}"_config.yaml \
    --task_config "$yaml_file" \
    --gpu 0 \
    --interval="$interval" \
    --save_dir="$save_dir" \
    --method "ours" \
    --guidance_scale "$guidance_scale"
}

# Main loop
job=0
for interval in "${intervals[@]}"; do
  for guidance_scale in "${guidance_scales[@]}"; do
    for yaml_file in "${task_config[@]}"; do
      echo "Running job $job with interval $interval, guidance scale $guidance_scale, and yaml file $yaml_file on GPU ${GPUS[$((job % ${#GPUS[@]}))]}"
      gpu=${GPUS[$((job % ${#GPUS[@]}))]}
      run_job "$interval" "$guidance_scale" "$yaml_file" "$gpu" &
      ((job++))
      # Optional: Limit number of concurrent jobs to number of GPUs
      if (( job % ${#GPUS[@]} == 0 )); then
        wait
      fi
    done
  done
done

wait  # Wait for all background jobs to finish
