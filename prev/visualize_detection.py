"""
Visualize detected aneurysm locations on 2D slices for manual verification
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def visualize_aneurysm_detection(nifti_path, coords_json_path, output_dir="detection_slices"):
    """Create 2D slice visualizations showing detected aneurysm locations."""
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    image_data = nii_img.get_fdata()
    
    # Load aneurysm coordinates
    with open(coords_json_path, 'r') as f:
        coords_data = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Image shape: {image_data.shape}")
    print(f"Total aneurysms detected: {coords_data['total_aneurysms_found']}")
    
    # Process each detected aneurysm
    for aneurysm_id, aneurysm_info in coords_data['aneurysms'].items():
        centroid = aneurysm_info['centroid_voxel']
        
        print(f"\n{aneurysm_id.upper()}:")
        print(f"  Centroid (voxel): ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        
        # Create visualizations for this aneurysm
        center_x, center_y, center_z = int(centroid[0]), int(centroid[1]), int(centroid[2])
        
        # Create figure with 3 subplots (axial, sagittal, coronal)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{aneurysm_id} Detection - Centroid: ({center_x}, {center_y}, {center_z})', fontsize=14)
        
        # Axial slice (XY plane, fixed Z)
        if 0 <= center_z < image_data.shape[2]:
            axial_slice = image_data[:, :, center_z]
            axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')
            axes[0].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3)
            axes[0].set_title(f'Axial (Z={center_z})')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
        
        # Sagittal slice (YZ plane, fixed X)  
        if 0 <= center_x < image_data.shape[0]:
            sagittal_slice = image_data[center_x, :, :]
            axes[1].imshow(sagittal_slice.T, cmap='gray', origin='lower')
            axes[1].plot(center_y, center_z, 'r+', markersize=15, markeredgewidth=3)
            axes[1].set_title(f'Sagittal (X={center_x})')
            axes[1].set_xlabel('Y')
            axes[1].set_ylabel('Z')
        
        # Coronal slice (XZ plane, fixed Y)
        if 0 <= center_y < image_data.shape[1]:
            coronal_slice = image_data[:, center_y, :]
            axes[2].imshow(coronal_slice.T, cmap='gray', origin='lower')
            axes[2].plot(center_x, center_z, 'r+', markersize=15, markeredgewidth=3)
            axes[2].set_title(f'Coronal (Y={center_y})')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Z')
        
        # Save the figure
        output_path = f"{output_dir}/{aneurysm_id}_detection.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
    
    print(f"\nAll visualizations saved in: {output_dir}/")


if __name__ == "__main__":
    nifti_file = " example_data/01_MRA1_seg.nii.gz"
    coords_file = "aneurysm_coords_final.json"
    
    print("=== Visualizing Aneurysm Detection ===")
    visualize_aneurysm_detection(nifti_file, coords_file) 