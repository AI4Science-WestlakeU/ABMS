import argparse
from torch_fidelity import calculate_metrics

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate FID and KID metrics')
    parser.add_argument('--real_folder', type=str, required=True, help='Path to folder containing real images')
    parser.add_argument('--gen_folder', type=str, required=True, help='Path to folder containing generated images')
    args = parser.parse_args()
    
    fid_score, kid_score = compute_fid_kid(args.real_folder, args.gen_folder)
    
    print("\nFID/KID Evaluation Results:")
    print(f"FID Score: {fid_score:.4f}")
    print(f"KID Score: {kid_score:.4f}") 