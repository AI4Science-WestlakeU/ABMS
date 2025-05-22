import os
import torch
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
from functions.arcface.model import IDLoss
from tqdm import tqdm

def compute_faceid_pair(real_path, gen_path, output_dir, gpu_id):
    """
    Compute FaceID similarity between a single pair of real and generated faces using ArcFace
    
    Args:
        real_path: Path to real face image
        gen_path: Path to generated face image
        output_dir: Directory to save results
        gpu_id: GPU ID to use
    """
    # Set GPU device
    torch.cuda.set_device(gpu_id)
    
    # Transform for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Initialize IDLoss with real image
    idloss = IDLoss(ref_path=real_path).cuda()
    
    # Load and process generated image
    gen_image = Image.open(gen_path).convert('RGB')
    gen_tensor = transform(gen_image).unsqueeze(0).cuda()
    
    # Compute residual and norm
    residual = idloss.get_residual(gen_tensor)
    norm = torch.linalg.norm(residual).item()
    
    # Create result dictionary
    result = {
        'real_path': real_path,
        'gen_path': gen_path,
        'norm': float(norm)
    }
    
    # Save result to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"faceid_{os.path.basename(real_path).split('.')[0]}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    return norm

def aggregate_results(results_dir):
    """Aggregate all individual FaceID results and compute mean"""
    norms = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                result = json.load(f)
                norms.append(result['norm'])
    
    mean_norm = sum(norms) / len(norms) if norms else 0.0
    return mean_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate FaceID loss for a single image pair')
    parser.add_argument('--real_path', type=str, required=True, help='Path to real image')
    parser.add_argument('--gen_path', type=str, required=True, help='Path to generated image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    args = parser.parse_args()
    
    compute_faceid_pair(args.real_path, args.gen_path, args.output_dir, args.gpu_id) 