"""
Validate manually specified aneurysm coordinates and create visualizations
"""

import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def validate_manual_coordinates(nifti_path, manual_coords_path, output_dir="manual_validation"):
    """
    Validate manual aneurysm coordinates and create visualizations.
    
    Args:
        nifti_path (str): Path to NIfTI file
        manual_coords_path (str): Path to manual coordinates JSON
        output_dir (str): Directory to save validation images
    """
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    image_data = nii_img.get_fdata()
    
    # Load manual coordinates
    with open(manual_coords_path, 'r') as f:
        manual_data = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print("=== Manual Aneurysm Coordinate Validation ===")
    print(f"NIfTI file: {nifti_path}")
    print(f"Manual coordinates: {manual_coords_path}")
    print(f"Image shape: {image_data.shape}")
    print(f"Image range: [{np.min(image_data):.3f}, {np.max(image_data):.3f}]")
    
    # Validate image dimensions match
    expected_dims = manual_data['image_info']['dimensions']
    if list(image_data.shape) != expected_dims:
        print(f"WARNING: Image dimensions {list(image_data.shape)} don't match expected {expected_dims}")
    
    valid_aneurysms = []
    
    # Process each manual aneurysm
    for i, aneurysm in enumerate(manual_data['aneurysms']):
        aneurysm_id = aneurysm['aneurysm_id']
        annotation_type = aneurysm.get('annotation_type', 'centroid')
        
        print(f"\n{aneurysm_id.upper()}:")
        print(f"  Annotation type: {annotation_type}")
        print(f"  Description: {aneurysm['description']}")
        
        # Handle different annotation types
        points_to_validate = []
        
        if annotation_type == 'centroid':
            centroid = aneurysm['centroid_voxel']
            points_to_validate.append(('centroid', centroid))
            print(f"  Centroid: {centroid}")
            
        elif annotation_type == 'multiple_points':
            key_points = aneurysm['key_points']
            for point_name, coords in key_points.items():
                if coords != [0, 0, 0]:  # Skip placeholder coordinates
                    points_to_validate.append((point_name, coords))
                    print(f"  {point_name}: {coords}")
                    
        elif annotation_type == 'bounding_box':
            bbox = aneurysm['bounding_box']
            min_coords = bbox['min_coords']
            max_coords = bbox['max_coords']
            centroid = aneurysm.get('centroid_voxel', [0, 0, 0])
            
            points_to_validate.append(('min_coords', min_coords))
            points_to_validate.append(('max_coords', max_coords))
            if centroid != [0, 0, 0]:
                points_to_validate.append(('centroid', centroid))
            
            print(f"  Bounding box: {min_coords} to {max_coords}")
            if centroid != [0, 0, 0]:
                print(f"  Centroid: {centroid}")
        
        # Validate all points
        valid_points = []
        for point_name, coords in points_to_validate:
            x, y, z = coords
            
            validation_errors = []
            if not (0 <= x < image_data.shape[0]):
                validation_errors.append(f"{point_name} X coordinate {x} out of range [0, {image_data.shape[0]-1}]")
            if not (0 <= y < image_data.shape[1]):
                validation_errors.append(f"{point_name} Y coordinate {y} out of range [0, {image_data.shape[1]-1}]")
            if not (0 <= z < image_data.shape[2]):
                validation_errors.append(f"{point_name} Z coordinate {z} out of range [0, {image_data.shape[2]-1}]")
            
            if validation_errors:
                print(f"  ❌ VALIDATION ERRORS for {point_name}:")
                for error in validation_errors:
                    print(f"     {error}")
                continue
            
            # Check if coordinates point to aneurysm tissue (non-zero values)
            voxel_value = image_data[x, y, z]
            print(f"  {point_name} voxel value: {voxel_value:.3f}")
            
            if voxel_value < 0.5:
                print(f"  ⚠️  WARNING: {point_name} voxel value is low ({voxel_value:.3f}). May not be aneurysm tissue.")
            else:
                print(f"  ✅ {point_name} points to aneurysm tissue")
            
            valid_points.append({
                'point_name': point_name,
                'coordinates': coords,
                'voxel_value': float(voxel_value)
            })
        
        # Create visualization (use centroid if available, otherwise first valid point)
        viz_centroid = None
        if annotation_type == 'centroid':
            viz_centroid = aneurysm['centroid_voxel']
        elif annotation_type == 'multiple_points' and 'centroid' in aneurysm['key_points']:
            viz_centroid = aneurysm['key_points']['centroid']
        elif annotation_type == 'bounding_box' and 'centroid_voxel' in aneurysm:
            viz_centroid = aneurysm['centroid_voxel']
        elif valid_points:
            viz_centroid = valid_points[0]['coordinates']
        
        if viz_centroid and viz_centroid != [0, 0, 0]:
            create_manual_validation_plot(image_data, viz_centroid, aneurysm_id, aneurysm['description'], output_dir)
        
        if valid_points:
            valid_aneurysms.append({
                'aneurysm_id': aneurysm_id,
                'annotation_type': annotation_type,
                'points': valid_points,
                'description': aneurysm['description'],
                'confidence': aneurysm.get('confidence', 'unknown')
            })
    
    # Create summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total aneurysms specified: {len(manual_data['aneurysms'])}")
    print(f"Valid aneurysms: {len(valid_aneurysms)}")
    
    # Save validation results
    validation_results = {
        'nifti_file': nifti_path,
        'manual_coords_file': manual_coords_path,
        'validation_date': manual_data.get('annotation_date', 'unknown'),
        'annotator': manual_data.get('annotator', 'unknown'),
        'image_shape': list(image_data.shape),
        'total_specified': len(manual_data['aneurysms']),
        'valid_aneurysms': len(valid_aneurysms),
        'aneurysms': valid_aneurysms
    }
    
    validation_output = f"{output_dir}/validation_results.json"
    with open(validation_output, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation results saved: {validation_output}")
    print(f"Validation images saved in: {output_dir}/")
    
    return validation_results


def create_manual_validation_plot(image_data, centroid, aneurysm_id, description, output_dir):
    """Create validation plot for manually specified aneurysm."""
    
    x, y, z = centroid
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{aneurysm_id} - Manual Annotation\n{description}', fontsize=12)
    
    # Axial slice (XY plane, fixed Z)
    if 0 <= z < image_data.shape[2]:
        axial_slice = image_data[:, :, z]
        im1 = axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')
        axes[0].plot(x, y, 'g+', markersize=20, markeredgewidth=4, label='Manual annotation')
        axes[0].set_title(f'Axial (Z={z})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend()
        
        # Add colorbar
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Sagittal slice (YZ plane, fixed X)
    if 0 <= x < image_data.shape[0]:
        sagittal_slice = image_data[x, :, :]
        im2 = axes[1].imshow(sagittal_slice.T, cmap='gray', origin='lower')
        axes[1].plot(y, z, 'g+', markersize=20, markeredgewidth=4, label='Manual annotation')
        axes[1].set_title(f'Sagittal (X={x})')
        axes[1].set_xlabel('Y')
        axes[1].set_ylabel('Z')
        axes[1].legend()
        
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Coronal slice (XZ plane, fixed Y)
    if 0 <= y < image_data.shape[1]:
        coronal_slice = image_data[:, y, :]
        im3 = axes[2].imshow(coronal_slice.T, cmap='gray', origin='lower')
        axes[2].plot(x, z, 'g+', markersize=20, markeredgewidth=4, label='Manual annotation')
        axes[2].set_title(f'Coronal (Y={y})')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Z')
        axes[2].legend()
        
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Save the figure
    output_path = f"{output_dir}/{aneurysm_id}_manual_validation.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Validation image saved: {output_path}")


def main():
    """Command line interface for manual coordinate validation."""
    parser = argparse.ArgumentParser(description='Validate manual aneurysm coordinates')
    parser.add_argument('manual_coords', help='Manual coordinates JSON file')
    parser.add_argument('-n', '--nifti', default=' example_data/01_MRA1_seg.nii.gz',
                       help='NIfTI file path')
    parser.add_argument('-o', '--output', default='manual_validation',
                       help='Output directory for validation images')
    
    args = parser.parse_args()
    
    try:
        validate_manual_coordinates(args.nifti, args.manual_coords, args.output)
        print("\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 