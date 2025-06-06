"""
Create STL file centered on detected aneurysm for visualization
"""

import sys
sys.path.append('processing')

import json
import nibabel as nib
import numpy as np
from NIfTI_to_stl import nifti_to_stl, create_mesh_from_volume, save_stl
from pathlib import Path


def create_aneurysm_centered_stl(nifti_path, coords_json_path, output_stl_path=None, crop_factor=0.25):
    """
    Create an STL file centered on the detected aneurysm.
    
    Args:
        nifti_path (str): Path to original NIfTI file
        coords_json_path (str): Path to aneurysm coordinates JSON
        output_stl_path (str): Output STL file path
        crop_factor (float): Fraction of original size for the crop (default: 0.25 = 1/4)
    """
    
    # Load aneurysm coordinates
    with open(coords_json_path, 'r') as f:
        coords_data = json.load(f)
    
    if coords_data['total_aneurysms_found'] == 0:
        raise ValueError("No aneurysms found in the coordinates file")
    
    # Get the first aneurysm
    aneurysm_info = list(coords_data['aneurysms'].values())[0]
    centroid = aneurysm_info['centroid_voxel']
    
    print(f"Aneurysm centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    print(f"Aneurysm volume: {aneurysm_info['volume_voxels']} voxels")
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    image_data = nii_img.get_fdata()
    affine = nii_img.affine
    header = nii_img.header
    
    original_shape = image_data.shape
    print(f"Original image shape: {original_shape}")
    
    # Calculate crop dimensions (crop_factor of original size)
    crop_size = [int(dim * crop_factor) for dim in original_shape]
    print(f"Crop size: {crop_size}")
    
    # Calculate crop boundaries centered on aneurysm
    half_crop = [size // 2 for size in crop_size]
    
    crop_min = [
        max(0, int(centroid[i] - half_crop[i]))
        for i in range(3)
    ]
    
    crop_max = [
        min(original_shape[i], int(centroid[i] + half_crop[i]))
        for i in range(3)
    ]
    
    # Adjust if crop goes outside boundaries
    for i in range(3):
        if crop_max[i] - crop_min[i] < crop_size[i]:
            # Try to extend the other direction
            needed = crop_size[i] - (crop_max[i] - crop_min[i])
            if crop_min[i] >= needed:
                crop_min[i] -= needed
            else:
                crop_max[i] = min(original_shape[i], crop_max[i] + needed)
    
    print(f"Crop boundaries: [{crop_min[0]}:{crop_max[0]}, {crop_min[1]}:{crop_max[1]}, {crop_min[2]}:{crop_max[2]}]")
    
    # Extract the cropped region
    cropped_data = image_data[
        crop_min[0]:crop_max[0],
        crop_min[1]:crop_max[1], 
        crop_min[2]:crop_max[2]
    ]
    
    print(f"Cropped shape: {cropped_data.shape}")
    print(f"Cropped data range: [{np.min(cropped_data):.3f}, {np.max(cropped_data):.3f}]")
    print(f"Non-zero voxels in crop: {np.count_nonzero(cropped_data)}")
    
    # Create output filename if not provided
    if output_stl_path is None:
        base_name = Path(nifti_path).stem
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_stl_path = f"{base_name}_aneurysm_crop.stl"
    
    # Check if there's any aneurysm data in the crop
    if np.max(cropped_data) <= 0.5:
        print("Warning: No aneurysm data found in the cropped region. Expanding crop...")
        
        # Expand the crop size
        crop_size = [int(dim * crop_factor * 2) for dim in original_shape]  # Double the crop size
        half_crop = [size // 2 for size in crop_size]
        
        crop_min = [
            max(0, int(centroid[i] - half_crop[i]))
            for i in range(3)
        ]
        
        crop_max = [
            min(original_shape[i], int(centroid[i] + half_crop[i]))
            for i in range(3)
        ]
        
        print(f"Expanded crop boundaries: [{crop_min[0]}:{crop_max[0]}, {crop_min[1]}:{crop_max[1]}, {crop_min[2]}:{crop_max[2]}]")
        
        # Extract the expanded cropped region
        cropped_data = image_data[
            crop_min[0]:crop_max[0],
            crop_min[1]:crop_max[1], 
            crop_min[2]:crop_max[2]
        ]
        
        print(f"Expanded cropped shape: {cropped_data.shape}")
        print(f"Non-zero voxels in expanded crop: {np.count_nonzero(cropped_data)}")
    
    # Create mesh using marching cubes
    print("Creating mesh...")
    
    # Use a threshold that captures the aneurysm data
    threshold = 0.5
    vertices, faces, normals, values = create_mesh_from_volume(
        cropped_data, threshold=threshold, step_size=1
    )
    
    print(f"Mesh created: {len(vertices)} vertices, {len(faces)} faces")
    
    # Adjust vertices to world coordinates
    # First, offset vertices by crop position
    vertices_adjusted = vertices.copy()
    for i in range(3):
        vertices_adjusted[:, i] += crop_min[i]
    
    # Apply affine transformation if available
    if affine is not None:
        vertices_homo = np.column_stack([vertices_adjusted, np.ones(len(vertices_adjusted))])
        vertices_world = vertices_homo.dot(affine.T)[:, :3]
        vertices_final = vertices_world
    else:
        vertices_final = vertices_adjusted
    
    # Save STL
    save_stl(vertices_final, faces, output_stl_path, binary=True)
    
    print(f"\nSTL file created: {output_stl_path}")
    print(f"File contains {len(vertices)} vertices and {len(faces)} faces")
    
    return output_stl_path


def main():
    """Main function to create aneurysm-centered STL."""
    
    nifti_file = " example_data/01_MRA1_seg.nii.gz"
    coords_file = "aneurysm_coords_final.json"
    output_file = "aneurysm_visualization.stl"
    
    print("=== Creating Aneurysm Visualization STL ===")
    print(f"NIfTI file: {nifti_file}")
    print(f"Coordinates: {coords_file}")
    print(f"Output STL: {output_file}")
    print()
    
    try:
        result_path = create_aneurysm_centered_stl(
            nifti_path=nifti_file,
            coords_json_path=coords_file,
            output_stl_path=output_file,
            crop_factor=0.25  # 1/4 size
        )
        
        print(f"\nSuccess! Aneurysm visualization saved to: {result_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 