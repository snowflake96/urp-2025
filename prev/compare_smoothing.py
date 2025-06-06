#!/usr/bin/env python3
"""
Compare original vs smoothed files to diagnose smoothing issue
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

def compare_files(original_path, smoothed_path):
    """Compare two NIfTI files and show statistics"""
    print(f"\n=== Comparing Files ===")
    print(f"Original: {original_path.name}")
    print(f"Smoothed: {smoothed_path.name}")
    
    # Load both files
    original_img = nib.load(original_path)
    smoothed_img = nib.load(smoothed_path)
    
    original_data = original_img.get_fdata()
    smoothed_data = smoothed_img.get_fdata()
    
    print(f"\nData shapes:")
    print(f"Original: {original_data.shape}")
    print(f"Smoothed: {smoothed_data.shape}")
    
    # Compare voxel counts
    original_nonzero = np.sum(original_data > 0)
    smoothed_nonzero = np.sum(smoothed_data > 0)
    
    print(f"\nNon-zero voxels:")
    print(f"Original: {original_nonzero}")
    print(f"Smoothed: {smoothed_nonzero}")
    
    if smoothed_nonzero == 0:
        print("❌ CRITICAL: Smoothed file is completely empty!")
        return original_nonzero, smoothed_nonzero
    
    print(f"Ratio: {smoothed_nonzero/original_nonzero:.3f}")
    
    # Compare intensity distributions
    original_values = original_data[original_data > 0]
    smoothed_values = smoothed_data[smoothed_data > 0]
    
    print(f"\nIntensity statistics:")
    print(f"Original - min: {original_values.min():.3f}, max: {original_values.max():.3f}, mean: {original_values.mean():.3f}")
    print(f"Smoothed - min: {smoothed_values.min():.3f}, max: {smoothed_values.max():.3f}, mean: {smoothed_values.mean():.3f}")
    
    # Check if smoothed file appears to be binary
    unique_smoothed = np.unique(smoothed_values)
    print(f"\nUnique values in smoothed data: {len(unique_smoothed)}")
    if len(unique_smoothed) <= 10:
        print(f"Unique values: {unique_smoothed}")
    
    # Show central slices
    center_z = original_data.shape[2] // 2
    original_slice = original_data[:, :, center_z]
    smoothed_slice = smoothed_data[:, :, center_z]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_slice.T, cmap='gray', origin='lower')
    plt.title(f'Original\n{original_nonzero} voxels')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(smoothed_slice.T, cmap='gray', origin='lower')
    plt.title(f'Smoothed\n{smoothed_nonzero} voxels')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow((original_slice - smoothed_slice).T, cmap='RdBu', origin='lower')
    plt.title('Difference\n(Original - Smoothed)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'comparison_{original_path.stem}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return original_nonzero, smoothed_nonzero

def main():
    # Define paths
    original_dir = Path("/home/jiwoo/urp/data/uan/original")
    smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed")
    
    # Compare a few representative files
    test_files = [
        ("01_MRA1_seg.nii.gz", "01_MRA1_seg_smoothed.nii.gz"),
        ("01_MRA2_seg.nii.gz", "01_MRA2_seg_smoothed.nii.gz"),
        ("02_MRA1_seg.nii.gz", "02_MRA1_seg_smoothed.nii.gz")
    ]
    
    print("=== Smoothing Quality Analysis ===")
    
    total_original_voxels = 0
    total_smoothed_voxels = 0
    empty_files = 0
    
    for original_name, smoothed_name in test_files:
        original_path = original_dir / original_name
        smoothed_path = smoothed_dir / smoothed_name
        
        if original_path.exists() and smoothed_path.exists():
            orig_voxels, smooth_voxels = compare_files(original_path, smoothed_path)
            total_original_voxels += orig_voxels
            total_smoothed_voxels += smooth_voxels
            if smooth_voxels == 0:
                empty_files += 1
        else:
            print(f"Missing files: {original_name} or {smoothed_name}")
    
    print(f"\n=== Overall Summary ===")
    print(f"Total original voxels: {total_original_voxels}")
    print(f"Total smoothed voxels: {total_smoothed_voxels}")
    print(f"Empty smoothed files: {empty_files}/{len(test_files)}")
    
    if total_smoothed_voxels == 0:
        print("❌ CRITICAL ISSUE: ALL smoothed files are empty!")
        print("The smoothing algorithm is removing all data instead of smoothing it.")
        print("This needs to be fixed with a proper smoothing implementation.")
    elif total_smoothed_voxels/total_original_voxels < 0.8:
        print("⚠️  WARNING: Smoothing is removing too much data (>20% loss)")
    elif total_smoothed_voxels/total_original_voxels > 0.95:
        print("⚠️  WARNING: Smoothing may be too conservative (minimal change)")
    else:
        print("✓ Smoothing appears reasonable")

if __name__ == "__main__":
    main() 