#!/usr/bin/env python
# Script to collect style guidance results and create visualization plots

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Collect style guidance results and create visualization plots')
    parser.add_argument('--outdir', type=str, default="outputs/txt2img-samples/search_params_DSG_style_guidance",
                        help='Output directory for the style guidance results')
    parser.add_argument('--scales', type=float, nargs='+', default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                        help='List of scale values used in the style guidance')
    parser.add_argument('--guidance_rates', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5, 1.0],
                        help='List of guidance rate values used in the style guidance')
    parser.add_argument('--style_images_dir', type=str, default="./style_images",
                        help='Directory containing the style reference images')
    return parser.parse_args()

# Configuration from arguments
args = parse_args()
OUTDIR = args.outdir
SCALES = args.scales
GUIDANCE_RATES = args.guidance_rates
STYLE_IMAGES_DIR = args.style_images_dir

def extract_method_name(output_dir):
    """Extract method name (ours or DSG) from the output directory path."""
    if "search_params_DSG_style_guidance" in output_dir:
        return "DSG"
    elif "search_params_ours_style_guidance" in output_dir:
        return "Ours"
    else:
        # Try to extract from directory name
        dir_name = os.path.basename(output_dir)
        if "DSG" in dir_name:
            return "DSG"
        elif "ours" in dir_name:
            return "Ours"
        else:
            return "Unknown Method"

def collect_style_results():
    """Collect style guidance results and organize them by reference image."""
    results = {}
    
    # Get all style reference images
    style_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        style_images.extend(glob.glob(os.path.join(STYLE_IMAGES_DIR, ext)))
    
    style_image_names = {os.path.splitext(os.path.basename(img))[0]: img for img in style_images}
    
    print(f"Found {len(style_images)} style reference images: {list(style_image_names.keys())}")
    
    # Process each scale and guidance rate combination
    for scale in SCALES:
        for guidance_rate in GUIDANCE_RATES:
            dir_name = f"scale_{scale}_guidance_rate_{guidance_rate}"
            output_dir = os.path.join(OUTDIR, dir_name)
            samples_dir = os.path.join(output_dir, "samples")
            
            if not os.path.exists(samples_dir):
                print(f"Warning: Directory not found: {samples_dir}")
                continue
            
            # Get all generated images in this directory
            generated_images = glob.glob(os.path.join(samples_dir, "*.png"))
            
            for img_path in generated_images:
                # Extract reference image name from filename
                ref_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Skip if this isn't one of our known reference images
                if ref_name not in style_image_names:
                    continue
                
                # Initialize the results structure if this is a new reference
                if ref_name not in results:
                    results[ref_name] = {}
                
                if scale not in results[ref_name]:
                    results[ref_name][scale] = {}
                
                # Store the generated image path indexed by scale and guidance rate
                results[ref_name][scale][guidance_rate] = img_path
    
    return results, style_image_names

def create_visualization(results, style_image_names):
    """Create visualization plots for each reference image."""
    # Create output directory for plots
    plots_dir = os.path.join(OUTDIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract method name from output directory
    method_name = extract_method_name(OUTDIR)
    
    for ref_name, scales in results.items():
        # Get the reference image path
        if ref_name not in style_image_names:
            print(f"Warning: Reference image not found for {ref_name}")
            continue
            
        ref_image_path = style_image_names[ref_name]
        
        # Create a figure with subplots for each scale and guidance rate
        fig = plt.figure(figsize=(4*len(GUIDANCE_RATES) + 2, 4*len(SCALES) + 1), dpi=100)
        
        # Create grid for the main matrix of results
        grid_size = (len(SCALES), len(GUIDANCE_RATES))
        grid = plt.GridSpec(grid_size[0], grid_size[1], left=0.2, right=0.98, 
                           bottom=0.05, top=0.85, wspace=0.1, hspace=0.2)
        
        # Add title with reference image name and method
        fig.suptitle(f"Style Transfer Results for Reference: {ref_name} ({method_name})", fontsize=20)
        
        # Add the reference image at the top left, outside the main grid
        ref_img = Image.open(ref_image_path)
        ax_ref = fig.add_axes([0.02, 0.88, 0.15, 0.10])  # [left, bottom, width, height]
        ax_ref.imshow(np.array(ref_img))
        ax_ref.set_title("Reference")
        ax_ref.axis("off")
        
        # Add column headers for guidance rates
        for j, guidance_rate in enumerate(GUIDANCE_RATES):
            ax = fig.add_subplot(grid[0, j])
            ax.set_title(f"Rate: {guidance_rate}", fontsize=12)
            ax.axis('off')
        
        # Create array to store axes for the main matrix
        axes = np.empty(grid_size, dtype=object)
        
        # Plot each generated image
        for i, scale in enumerate(SCALES):
            # Add row label for scale
            row_label = fig.add_subplot(grid[i, 0])
            row_label.text(-0.5, 0.5, f"Scale: {scale}", fontsize=12, 
                          ha='right', va='center', transform=row_label.transAxes)
            row_label.axis('off')
            
            for j, guidance_rate in enumerate(GUIDANCE_RATES):
                ax = fig.add_subplot(grid[i, j])
                axes[i, j] = ax
                
                if scale in scales and guidance_rate in scales[scale]:
                    img_path = scales[scale][guidance_rate]
                    try:
                        img = Image.open(img_path)
                        ax.imshow(np.array(img))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No result", ha='center', va='center')
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
        
        # Save the figure
        output_path = os.path.join(plots_dir, f"{ref_name}_results.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close(fig)

def create_summary_visualization(results, style_image_names):
    """Create a summary visualization with all reference images and their best results."""
    if not results:
        print("No results to create summary visualization.")
        return
        
    plots_dir = os.path.join(OUTDIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract method name
    method_name = extract_method_name(OUTDIR)
    
    # Determine the grid size based on number of reference images
    n_refs = len(results)
    n_cols = min(3, n_refs)  # Max 3 columns
    n_rows = (n_refs + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with more space
    fig = plt.figure(figsize=(7*n_cols, 4*n_rows), dpi=100)
    
    # Create grid for subplot placement
    grid = plt.GridSpec(n_rows, n_cols*2, wspace=0.3, hspace=0.4)
    
    fig.suptitle(f"Style Transfer Summary Results ({method_name})", fontsize=24)
    
    # Find best result for each reference
    # Define "best" as scale=15.0 and guidance_rate=0.2 if available, or closest available
    target_scale = 15.0
    target_guidance_rate = 0.2
    
    for idx, (ref_name, scales) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        # Get reference image
        ref_image_path = style_image_names[ref_name]
        ref_img = Image.open(ref_image_path)
        
        # Add reference image
        ref_ax = fig.add_subplot(grid[row, col*2])
        ref_ax.imshow(np.array(ref_img))
        ref_ax.set_title(f"Reference: {ref_name}", fontsize=14)
        ref_ax.axis('off')
        
        # Find best result (closest to target parameters)
        best_img_path = None
        best_distance = float('inf')
        best_params = None
        
        for scale in scales:
            for guidance_rate in scales[scale]:
                # Calculate distance from target parameters
                distance = (scale - target_scale)**2 + (guidance_rate - target_guidance_rate)**2
                if distance < best_distance:
                    best_distance = distance
                    best_img_path = scales[scale][guidance_rate]
                    best_params = (scale, guidance_rate)
        
        # Add best result
        result_ax = fig.add_subplot(grid[row, col*2+1])
        if best_img_path:
            try:
                img = Image.open(best_img_path)
                result_ax.imshow(np.array(img))
                result_ax.set_title(f"Scale={best_params[0]}, Rate={best_params[1]}", fontsize=12)
            except Exception as e:
                print(f"Error loading image {best_img_path}: {e}")
                result_ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        else:
            result_ax.text(0.5, 0.5, "No result", ha='center', va='center')
        
        result_ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title
    
    # Save the summary figure
    output_path = os.path.join(plots_dir, f"summary_results_{method_name}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved summary visualization to {output_path}")
    plt.close(fig)

def compute_similarity_to_reference(generated_img_path, reference_img_path):
    """Compute similarity between generated image and reference style image using SSIM."""
    try:
        # Load images and convert to grayscale
        gen_img = np.array(Image.open(generated_img_path).convert('L'))
        ref_img = np.array(Image.open(reference_img_path).convert('L'))
        
        # Resize reference to match generated if needed
        if ref_img.shape != gen_img.shape:
            ref_img = np.array(Image.open(reference_img_path).convert('L').resize(
                (gen_img.shape[1], gen_img.shape[0]), Image.LANCZOS))
        
        # Compute SSIM
        similarity = ssim(gen_img, ref_img)
        return similarity
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0  # Return lowest similarity on error

def create_parameter_surface_plots(results, style_image_names):
    """Create 3D surface plots to visualize the parameter space."""
    if not results:
        print("No results to create parameter surface plots.")
        return
        
    plots_dir = os.path.join(OUTDIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract method name
    method_name = extract_method_name(OUTDIR)
    
    # For each reference style
    for ref_name, scales in results.items():
        if ref_name not in style_image_names:
            continue
            
        ref_image_path = style_image_names[ref_name]
        
        # Create a grid for scale and guidance rate
        X, Y = np.meshgrid(SCALES, GUIDANCE_RATES)
        Z = np.zeros_like(X)
        
        # Calculate similarity for each parameter combination
        for i, scale in enumerate(SCALES):
            for j, guidance_rate in enumerate(GUIDANCE_RATES):
                if scale in scales and guidance_rate in scales[scale]:
                    img_path = scales[scale][guidance_rate]
                    similarity = compute_similarity_to_reference(img_path, ref_image_path)
                    Z[j, i] = similarity  # Note: j, i because meshgrid returns with first axis being Y
        
        # Create the 3D surface plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add color bar and labels
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Similarity to Reference Style')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Guidance Rate')
        ax.set_zlabel('Style Similarity')
        ax.set_title(f'Parameter Space Exploration for {ref_name} ({method_name})', fontsize=16)
        
        # Add annotations for the maximum similarity point
        max_idx = np.unravel_index(Z.argmax(), Z.shape)
        max_guidance_rate = GUIDANCE_RATES[max_idx[0]]
        max_scale = SCALES[max_idx[1]]
        max_similarity = Z[max_idx]
        
        ax.text(max_scale, max_guidance_rate, max_similarity, 
                f'Best: Scale={max_scale}, Rate={max_guidance_rate}\nSimilarity={max_similarity:.3f}', 
                color='red', fontsize=10)
        
        # Save the figure
        output_path = os.path.join(plots_dir, f"{ref_name}_parameter_surface_{method_name}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved parameter surface plot to {output_path}")
        plt.close(fig)
        
        # Create a 2D heatmap version as well
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(Z, cmap='viridis', origin='lower', 
                     extent=[min(SCALES), max(SCALES), min(GUIDANCE_RATES), max(GUIDANCE_RATES)])
        
        # Add colorbar and labels
        plt.colorbar(im, ax=ax, label='Similarity to Reference Style')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Guidance Rate')
        ax.set_title(f'Parameter Heatmap for {ref_name} ({method_name})', fontsize=16)
        
        # Mark the best parameters
        ax.plot(max_scale, max_guidance_rate, 'rx', markersize=10)
        ax.annotate(f'Best: Scale={max_scale}, Rate={max_guidance_rate}\nSimilarity={max_similarity:.3f}',
                   (max_scale, max_guidance_rate), xytext=(max_scale+2, max_guidance_rate+0.1),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Save the figure
        output_path = os.path.join(plots_dir, f"{ref_name}_parameter_heatmap_{method_name}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved parameter heatmap to {output_path}")
        plt.close(fig)

def main():
    print("Collecting style guidance results...")
    print(f"Using configuration:")
    print(f"  Output directory: {OUTDIR}")
    print(f"  Scales: {SCALES}")
    print(f"  Guidance rates: {GUIDANCE_RATES}")
    print(f"  Style images directory: {STYLE_IMAGES_DIR}")
    
    # Extract method name from output directory
    method_name = extract_method_name(OUTDIR)
    print(f"  Method detected: {method_name}")
    
    results, style_image_names = collect_style_results()
    
    if not results:
        print("No results found. Make sure the experiment has completed.")
        return
    
    print(f"Found results for {len(results)} reference images")
    
    print("Creating visualization plots...")
    create_visualization(results, style_image_names)
    
    print("Creating summary visualization...")
    create_summary_visualization(results, style_image_names)
    
    print("Creating parameter surface plots...")
    create_parameter_surface_plots(results, style_image_names)
    
    print("Done!")

if __name__ == "__main__":
    main() 