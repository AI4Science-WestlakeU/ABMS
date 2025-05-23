import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import rcParams

# Set paths
reference_dir = "style_images"
dsg_dir = "outputs/txt2img-samples/DSG_style_guidance"
freedom_dir = "outputs/txt2img-samples/freedom_style_guidance"
ours_dir = "outputs/txt2img-samples/ours_style_guidance"

# Get image file names from reference directory
image_files = sorted([f for f in os.listdir(reference_dir) if f.endswith('.jpg')])

# Create figure
n_examples = len(image_files)
fig, axes = plt.subplots(4, n_examples, figsize=(3*n_examples, 12))

# Row labels
row_labels = ['Reference', 'Freedom', 'DSG', 'Ours']

# Mapping from base names to grid file pattern
style_ids = {
    '376_two_women_with_the_armchair': '376_two_women_with_the_armchair',
    '333_Irregular_Horizontal_Bands_of_Equal_Width_Starting_at_Bottom': '333_Irregular_Horizontal_Bands_of_Equal_Width_Starting_at_Bottom',
    '99_Machiner': '99_Machiner',
    '198_clouds_over_bor': '198_clouds_over_bor',
    '137_le_village': '137_le_village',
    '39_vertical_and_horizontal_composition': '39_vertical_and_horizontal_composition',
    '81_free_&_easy': '81_free_&_easy'
}

# For each example
for col_idx, img_file in enumerate(image_files):
    base_name = os.path.splitext(img_file)[0]
    style_id = style_ids.get(base_name, base_name)
    
    # Reference image
    ref_img_path = os.path.join(reference_dir, img_file)
    if os.path.exists(ref_img_path):
        ref_img = imread(ref_img_path)
        axes[0, col_idx].imshow(ref_img)
    
    # Freedom image - NOTE: Currently empty, will leave blank
    # If images become available, this will display them
    freedom_files = [f for f in os.listdir(freedom_dir) if style_id in f and f.endswith('.png')]
    if freedom_files:
        freedom_img_path = os.path.join(freedom_dir, freedom_files[0])
        freedom_img = imread(freedom_img_path)
        axes[1, col_idx].imshow(freedom_img)
    
    # DSG image - use grid files
    dsg_files = [f for f in os.listdir(dsg_dir) if style_id in f and f.startswith('grid-') and f.endswith('.png')]
    if dsg_files:
        dsg_img_path = os.path.join(dsg_dir, dsg_files[0])
        dsg_img = imread(dsg_img_path)
        axes[2, col_idx].imshow(dsg_img)
    else:
        # Try sample directory as fallback
        dsg_sample_path = os.path.join(dsg_dir, 'samples', base_name + '.png')
        if os.path.exists(dsg_sample_path):
            dsg_img = imread(dsg_sample_path)
            axes[2, col_idx].imshow(dsg_img)
    
    # Ours image - use grid files
    ours_files = [f for f in os.listdir(ours_dir) if style_id in f and f.startswith('grid-') and f.endswith('.png')]
    if ours_files:
        ours_img_path = os.path.join(ours_dir, ours_files[0])
        ours_img = imread(ours_img_path)
        axes[3, col_idx].imshow(ours_img)
    else:
        # Try sample directory as fallback
        ours_sample_path = os.path.join(ours_dir, 'samples', base_name + '.png')
        if os.path.exists(ours_sample_path):
            ours_img = imread(ours_sample_path)
            axes[3, col_idx].imshow(ours_img)

# Remove ticks for all plots
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(left=0.12)  # Reduced margin to decrease space

# Add row labels using figure text, positioned just outside the matrix (after layout)
first_ax_pos = axes[0, 0].get_position()
label_x = first_ax_pos.x0 - 0.01  # margin outside the matrix
for idx, label in enumerate(row_labels):
    pos = axes[idx, 0].get_position()
    y = pos.y0 + pos.height / 2
    fig.text(label_x, y, label, va='center', ha='right', rotation=90, fontsize=24)

# Save the figure
plt.savefig('style_guidance_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as style_guidance_comparison.png") 