import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import argparse
from functions.arcface.model import IDLoss
import torch

def compute_faceid(real_paths, gen_paths):
    """
    Compute FaceID similarity between real and generated faces using ArcFace
    
    Args:
        real_images: List of real face images
        generated_images: List of generated face images
    """
    norms = []
    
    def transform(image):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])(image)

    for real_path, gen_path in tqdm(zip(real_paths, gen_paths), desc="Computing ArcFace similarities"):
        idloss = IDLoss(ref_path=real_path).cuda()
        gen_image = Image.open(gen_path).convert('RGB')
        
        gen_tensor = transform(gen_image).unsqueeze(0).cuda()
        
        residual = idloss.get_residual(gen_tensor)
        norm = torch.linalg.norm(residual)
        norms.append(norm.item())
        
    return float(np.mean(norms)) if norms else 0.0

def load_images_from_folder(folder_path, return_paths_only=True):
    """Load all PNG images from a folder and convert to tensors"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    images = []
    image_paths = []
    
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading images from {folder_path}"):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                if not return_paths_only:
                    images.append(transform(img))
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    if return_paths_only:
        return image_paths
    else:
        return torch.stack(images), image_paths

def compute_fid_kid(real_folder, gen_folder):
    """Compute FID and KID metrics between real and generated images"""
    metrics = calculate_metrics(
        input1=real_folder,
        input2=gen_folder,
        fid=True,
        kid=True,
        verbose=False
    )
    return metrics['frechet_inception_distance'], metrics['kernel_inception_distance_mean']

def evaluate_folders(real_folder, gen_folder):
    """Evaluate metrics between real and generated images in specified folders"""
    
    print("Loading real images...")
    real_paths = load_images_from_folder(real_folder)
    
    print("Loading generated images...")
    gen_paths = load_images_from_folder(gen_folder)
    
    print(f"Loaded {len(real_paths)} real images and {len(gen_paths)} generated images")
    
    # Compute FID and KID
    print("Computing FID and KID metrics...")
    fid_score, kid_score = compute_fid_kid(real_folder, gen_folder)
    
    # Compute FaceID using ArcFace
    print("Computing FaceID metric (ArcFace)...")
    faceid_arcface_score = compute_faceid(real_paths, gen_paths)
    
    results = {
        'fid': fid_score,
        'kid': kid_score,
        'faceid_arcface': faceid_arcface_score
    }
    
    print("\nEvaluation Results:")
    print(f"FID Score: {fid_score:.4f}")
    print(f"KID Score: {kid_score:.4f}")
    print(f"FaceID Score (ArcFace): {faceid_arcface_score:.4f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate image generation metrics')
    parser.add_argument('--real_folder', type=str, required=True, help='Path to folder containing real images')
    parser.add_argument('--gen_folder', type=str, required=True, help='Path to folder containing generated images')
    args = parser.parse_args()
    
    evaluate_folders(args.real_folder, args.gen_folder) 