method="ours"
task="style"
# image_paths=("./style_images/brice-marden_6-red-rock-1-2002.jpg" "./style_images/1.png" "./style_images/bijiasuo.jpeg" "./style_images/bw.jpg" "./style_images/jojo.jpeg" "./style_images/nahan.jpeg" "./style_images/tan.jpg" "./style_images/xiangrikui.jpg" "./style_images/xing.jpg" "./style_images/xingkong.jpg")
image_paths=("./style_images/jojo.jpeg")

available_devices=(1 2 3 5)

scales=(5.0 10.0 15.0 20.0 25.0 30.0)
guidance_rates=(0.05 0.1 0.2 0.5 1.0)

outdir="outputs/txt2img-samples/search_params_${method}_style_guidance"

# Create an array to hold all task configurations
declare -a jobs
for image_path in "${image_paths[@]}"; do
  for scale in "${scales[@]}"; do
    for guidance_rate in "${guidance_rates[@]}"; do
      jobs+=("$image_path $scale $guidance_rate")
    done
  done
done

# create output directory for all jobs
for job in "${jobs[@]}"; do
  IFS=' ' read -r img_path scale g_rate <<< "$task"
  outdir_path="$outdir/scale_${scale}_guidance_rate_${g_rate}"
  # Create directory if it doesn't exist, -p ensures no error if already exists
  mkdir -p "$outdir_path"
  echo "Created output directory: $outdir_path"
done


# Function to run a single task
run_task() {
  local image_path=$1
  local scale=$2
  local guidance_rate=$3
  local device=$4
  local task=$5
  local method=$6
  
  # Create the output directory path
  local outdir_path="$outdir/scale_${scale}_guidance_rate_${guidance_rate}"

  echo "Running on CUDA:$device - Scale: $scale, Guidance rate: $guidance_rate, Image: $image_path"
  CUDA_VISIBLE_DEVICES=$device python txt2img.py \
    --prompt "a cat wearing glasses." \
    --style_ref_img_path "$image_path" \
    --ddim_steps 100 \
    --n_iter 1 \
    --n_samples 1 \
    --H 512 \
    --W 512 \
    --scale "$scale" \
    --method "$method" \
    --task "$task" \
    --outdir "$outdir_path" \
    --ddim_eta 1 \
    --uncondition \
    --guidance_rate "$guidance_rate"
}


# echo "Running task with image: $image_path, method: $method, GPU: $cuda_device"
# CUDA_VISIBLE_DEVICES=$cuda_device python txt2img.py \
# --prompt "a cat wearing glasses." \
# --style_ref_img_path "$image_path" \
# --ddim_steps 100 \
# --n_iter 1 \
# --n_samples 1 \
# --H 512 \
# --W 512 \
# --scale "$scale" \
# --method "$method" \
# --task "$task" \
# --outdir "$outdir/${method}_style_guidance" \
# --ddim_eta 1 \
# --uncondition \
# --guidance_rate "$guidance_rate"

# Launch jobs in parallel with round-robin device assignment
device_count=${#available_devices[@]}
pids=()

for i in "${!jobs[@]}"; do
  # Get device index using modulo for round-robin assignment
  device_idx=$((i % device_count))
  device=${available_devices[$device_idx]}
  
  # Parse task parameters
  IFS=' ' read -r img_path scale g_rate <<< "${jobs[$i]}"
  
  # Run task in background
  run_task "$img_path" "$scale" "$g_rate" "$device" "$task" "$method" &
  pids+=($!)
  
  # Limit max concurrent processes to number of available devices
  if (( (i + 1) % device_count == 0 )); then
    echo "Waiting for batch of processes to complete..."
    wait "${pids[@]}"
    pids=()
  fi
done

# Wait for any remaining processes
if [ ${#pids[@]} -gt 0 ]; then
  echo "Waiting for remaining processes to complete..."
  wait "${pids[@]}"
fi

echo "All jobs completed!"

# Call the collection script to generate visualizations
echo "Generating result visualizations..."
python collect_style_guide_results.py --outdir "$outdir" --scales "${scales[@]}" --guidance_rates "${guidance_rates[@]}"
echo "Visualizations complete!"



# for image_path in "${image_paths[@]}"; do
#   # DSG
  
#   CUDA_VISIBLE_DEVICES=7 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "$image_path" --ddim_steps 100 --n_iter 1 --n_samples 1 \
#   --H 512 --W 512 --scale 5.0 --method "$method" --task "$task" --outdir "$outdir" --ddim_eta 1 --uncondition --guidance_rate 0.1
# done