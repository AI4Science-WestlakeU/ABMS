from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from torch_fidelity import calculate_metrics


device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'SR'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'/media/harry/tomo/FFHQ/256_1000')

delta_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps/{scale}/recon')
normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

psnr_delta_list = []
psnr_normal_list = []

lpips_delta_list = []
lpips_normal_list = []

ssim_delta_list = []
ssim_normal_list = []

for idx in tqdm(range(150)):
    fname = str(idx).zfill(5)

    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]

    psnr_delta = peak_signal_noise_ratio(label, delta_recon)
    psnr_normal = peak_signal_noise_ratio(label, normal_recon)

    psnr_delta_list.append(psnr_delta)
    psnr_normal_list.append(psnr_normal)

    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)
    normal_d = loss_fn_vgg(normal_recon, label)

    lpips_delta_list.append(delta_d)
    lpips_normal_list.append(normal_d)
    
    ssim_delta = ssim(label, delta_recon, channel_axis=2, data_range=1.0)
    ssim_normal = ssim(label, normal_recon, channel_axis=2, data_range=1.0)
    ssim_delta_list.append(ssim_delta)
    ssim_normal_list.append(ssim_normal)

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

ssim_delta_avg = sum(ssim_delta_list) / len(ssim_delta_list)
ssim_normal_avg = sum(ssim_normal_list) / len(ssim_normal_list)

print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')

print(f'Normal PSNR: {psnr_normal_avg}')
print(f'Normal LPIPS: {lpips_normal_avg}')

print(f'Delta SSIM: {ssim_delta_avg}')
print(f'Normal SSIM: {ssim_normal_avg}')

# --- FID computation using torch-fidelity ---
fid_delta = calculate_metrics(
    input1=str(delta_recon_root),
    input2=str(label_root),
    cuda=True,
    isc=False,
    kid=False,
    fid=True,
    verbose=False
)['frechet_inception_distance']

fid_normal = calculate_metrics(
    input1=str(normal_recon_root),
    input2=str(label_root),
    cuda=True,
    isc=False,
    kid=False,
    fid=True,
    verbose=False
)['frechet_inception_distance']

print(f'Delta FID: {fid_delta}')
print(f'Normal FID: {fid_normal}')