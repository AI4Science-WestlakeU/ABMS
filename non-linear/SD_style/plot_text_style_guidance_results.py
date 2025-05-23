import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse

def find_all_images(base_dirs, filter_keyword=None):
    """Find all images across directories, optionally filtering by keyword in filename."""
    results = {
        "Reference": [],
        "Freedom": [],
        "DSG": [],
        "Ours": []
    }
    
    # If filter_keyword is provided, convert to lowercase for case-insensitive matching
    keyword_lower = filter_keyword.lower() if filter_keyword else None
    
    # Find Reference images from text_style_images directory
    ref_path = base_dirs["Reference"]
    if os.path.exists(ref_path):
        ref_files = sorted(glob.glob(os.path.join(ref_path, "*.png")) + 
                          glob.glob(os.path.join(ref_path, "*.jpg")) +
                          glob.glob(os.path.join(ref_path, "*.jpeg")))
        for file in ref_files:
            if not keyword_lower or (keyword_lower and keyword_lower in os.path.basename(file).lower()):
                results["Reference"].append(file)
    
    # Find DSG images
    dsg_path = base_dirs["DSG"]
    if os.path.exists(dsg_path):
        dsg_files = sorted(glob.glob(os.path.join(dsg_path, "*.png")))
        for file in dsg_files:
            if not keyword_lower or (keyword_lower and keyword_lower in os.path.basename(file).lower()):
                results["DSG"].append(file)
    
    # Find Freedom images
    freedom_path = base_dirs["Freedom"]
    if os.path.exists(freedom_path):
        freedom_files = sorted(glob.glob(os.path.join(freedom_path, "*.png")))
        for file in freedom_files:
            if not keyword_lower or (keyword_lower and keyword_lower in os.path.basename(file).lower()):
                results["Freedom"].append(file)
    
    # Find Ours images
    ours_path = base_dirs["Ours"]
    if os.path.exists(ours_path):
        ours_files = sorted(glob.glob(os.path.join(ours_path, "*.png")))
        for file in ours_files:
            if not keyword_lower or (keyword_lower and keyword_lower in os.path.basename(file).lower()):
                results["Ours"].append(file)
    
    # Print debug info
    for method, files in results.items():
        print(f"{method}: Found {len(files)} images")
        if len(files) > 0:
            print(f"  Sample path: {files[0]}")
        else:
            print(f"  Directory: {base_dirs[method]}")
            print(f"  Directory exists: {os.path.exists(base_dirs[method])}")
    
    return results

def create_visualization(images, prompt, output_path=None):
    """Create a 2D matrix visualization of the images."""
    # Determine the number of columns (use the maximum number of images in any category)
    max_images = max(len(imgs) for imgs in images.values())
    if max_images == 0:
        print("No images found! Cannot create visualization.")
        return
    
    # Set up the figure
    fig, axes = plt.subplots(4, max_images, figsize=(4*max_images, 16))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Labels for y-axis
    methods = ["Reference", "Freedom", "DSG", "Ours"]
    
    # Ensure axes is 2D even if there's only one column
    if max_images == 1:
        axes = axes.reshape(-1, 1)
    
    # Load and display images
    for row, method in enumerate(methods):
        method_images = images[method]
        
        # Display each image in this row
        for col in range(max_images):
            # Turn off axis for all subplots
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
            # Remove all borders
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            
            # If there's an image for this column, display it
            if col < len(method_images) and os.path.exists(method_images[col]):
                try:
                    img = Image.open(method_images[col])
                    axes[row, col].imshow(np.array(img))
                    print(f"Loaded image: {method_images[col]}")
                except Exception as e:
                    print(f"Error loading image {method_images[col]}: {str(e)}")
                    axes[row, col].set_visible(False)
            else:
                # Empty cell
                axes[row, col].set_visible(False)
        
        # Set y-axis label for the first column only
        if max_images > 0:
            axes[row, 0].set_ylabel(method, rotation=90, labelpad=15, 
                              fontsize=24, verticalalignment='center')
    
    # Set the title
    if prompt:
        plt.suptitle(f"Text Style Guidance of '{prompt}'", fontsize=48, y=0.98)
    
    # Save or show
    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        except Exception as e:
            print(f"Error saving visualization to {output_path}: {str(e)}")
    else:
        print("Displaying visualization...")
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Collect and visualize text-to-image results")
    parser.add_argument("--filter", type=str, help="Optional keyword to filter images by filename")
    parser.add_argument("--output_name", type=str, default="text_style_guidance_comparison.png", help="Output path for visualization")
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output_name)
    
    # query user for prompt of text style guidance
    prompt = input("Enter the prompt for text style guidance: ")
    
    # Define base directories relative to the script location
    base_dirs = {
        "Reference": os.path.join(script_dir, "text_style_images"),
        "DSG": os.path.join(script_dir, "outputs/txt2img-samples/DSG_text_style_guidance/samples"),
        "Freedom": os.path.join(script_dir, "outputs/txt2img-samples/freedom_text_style_guidance/samples"),
        "Ours": os.path.join(script_dir, "outputs/txt2img-samples/ours_text_style_guidance/samples")
    }
    
    # Ensure output directories exist
    for method, path in base_dirs.items():
        if method != "Reference" and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    print(f"Script directory: {script_dir}")
    for method, path in base_dirs.items():
        print(f"{method} path: {path}")
        print(f"{method} path exists: {os.path.exists(path)}")
    
    # Find images, optionally filtering by filename keyword
    images = find_all_images(base_dirs, args.filter)
    
    # Check if any images were found
    total_images = sum(len(imgs) for imgs in images.values())
    if total_images == 0:
        print("No images found in any directory! Please check the directories and image files.")
        return
    
    # Create visualization
    create_visualization(images, prompt, output_path)

if __name__ == "__main__":
    main() 