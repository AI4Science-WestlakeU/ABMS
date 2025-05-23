import os
import random
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

HUGGINGFACE_DATASET_NAME = "your/dataset/name" # find your ffhq-256 dataset name on huggingface

# Make sure the cache directory exists
os.makedirs("./cache", exist_ok=True)

seed = 42
random.seed(seed)

# 1. Load the dataset (ffhq-256 has only one split: 'train')
dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train", cache_dir="./cache")

# 2. Randomly select 1000 samples
indices = random.sample(range(len(dataset)), 1000)
subset = dataset.select(indices)

# 3. Prepare output directory
output_dir = "./Face-GD/images/ffhq_256_subset@1k"
os.makedirs(output_dir, exist_ok=True)

# 4. Save images as PNG
for i, item in enumerate(tqdm(subset, desc="Saving images")):
    img = item["image"]
    # Ensure it's a PIL Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img.save(os.path.join(output_dir, f"{i:04d}.png"))

print(f"Saved 1000 PNG images to {output_dir}")
