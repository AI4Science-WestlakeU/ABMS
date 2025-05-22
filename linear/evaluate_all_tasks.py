import argparse
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import lpips
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_fidelity import calculate_metrics


def evaluate_task(result_dir, gt_dir, device):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    result_images = sorted([p for p in result_dir.glob('*.png')])
    psnr_list, ssim_list, lpips_list = [], [], []
    for res_path in tqdm(result_images, desc=f"Evaluating {result_dir.parent.name}"):
        fname = res_path.name
        # NOTE: the filename is one more zero in the front then the ground truth png
        gt_fname = fname[1:]
        gt_path = gt_dir / gt_fname
        if not gt_path.exists():
            print(f"Warning: Ground truth {gt_path} not found, skipping.")
            continue
        res_img = plt.imread(res_path)[..., :3]
        gt_img = plt.imread(gt_path)[..., :3]
        # PSNR
        psnr = peak_signal_noise_ratio(gt_img, res_img)
        psnr_list.append(psnr)
        # SSIM
        ssim_val = ssim(gt_img, res_img, channel_axis=2, data_range=1.0)
        ssim_list.append(ssim_val)
        # LPIPS
        res_tensor = torch.from_numpy(res_img).permute(2, 0, 1).to(device).float()
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).to(device).float()
        res_tensor = res_tensor.view(1, 3, res_tensor.shape[1], res_tensor.shape[2]) * 2. - 1.
        gt_tensor = gt_tensor.view(1, 3, gt_tensor.shape[1], gt_tensor.shape[2]) * 2. - 1.
        lpips_val = loss_fn_vgg(res_tensor, gt_tensor).item()
        lpips_list.append(lpips_val)
    # FID computation using torch-fidelity
    fid_result = calculate_metrics(
        input1=str(result_dir),
        input2=str(gt_dir),
        cuda=device.startswith('cuda'),
        isc=False,
        kid=False,
        fid=True,
        verbose=False
    )
    fid = fid_result['frechet_inception_distance']
    
    # Convert numpy types to Python native types
    metrics = {
        'psnr': float(np.mean(psnr_list)) if psnr_list else float('nan'),
        'ssim': float(np.mean(ssim_list)) if ssim_list else float('nan'),
        'lpips': float(np.mean(lpips_list)) if lpips_list else float('nan'),
        'fid': float(fid),
        'count': int(len(psnr_list))
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for all tasks.")
    parser.add_argument('--result_root', type=str, required=True, help='Root directory of results, e.g. data/samples')
    parser.add_argument('--gt_root', type=str, required=True, help='Root directory of ground truth images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for LPIPS computation')
    args = parser.parse_args()

    tasks = ['super_resolution', 'inpainting', 'gaussian_blur']
    all_metrics = {}
    for task in tasks:
        result_dir = Path(args.result_root) / task / 'recon'
        gt_dir = Path(args.gt_root)
        if not result_dir.exists():
            print(f"Result directory {result_dir} does not exist, skipping {task}.")
            continue
        metrics = evaluate_task(result_dir, gt_dir, args.device)
        all_metrics[task] = metrics
        print(f"\nTask: {task}")
        print(f"  Images evaluated: {metrics['count']}")
        print(f"  PSNR:  {metrics['psnr']:.4f}")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"  FID:   {metrics['fid']:.4f}")
    
    import json
    with open(args.result_root + '/metrics.json', 'w') as f:
        json.dump(all_metrics, f)

if __name__ == '__main__':
    main() 