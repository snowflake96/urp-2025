#!/usr/bin/env python3
"""
Compare original vs properly smoothed files to verify the fix
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

def compare_files(original_path, proper_path, conservative_path):
    """Compare original vs both smoothing methods"""
    print(f"\n=== Comparing Files ===")
    print(f"Original: {original_path.name}")
    print(f"Proper: {proper_path.name}")
    print(f"Conservative: {conservative_path.name}")
    
    # Load all files
    original_img = nib.load(original_path)
    proper_img = nib.load(proper_path)
    conservative_img = nib.load(conservative_path)
    
    original_data = original_img.get_fdata()
    proper_data = proper_img.get_fdata()
    conservative_data = conservative_img.get_fdata()
    
    # Compare voxel counts
    original_nonzero = np.sum(original_data > 0)
    proper_nonzero = np.sum(proper_data > 0)
    conservative_nonzero = np.sum(conservative_data > 0)
    
    print(f"\nNon-zero voxels:")
    print(f"Original:     {original_nonzero}")
    print(f"Proper:       {proper_nonzero} (ratio: {proper_nonzero/original_nonzero:.3f})")
    print(f"Conservative: {conservative_nonzero} (ratio: {conservative_nonzero/original_nonzero:.3f})")
    
    # Show central slices
    center_z = original_data.shape[2] // 2
    original_slice = original_data[:, :, center_z]
    proper_slice = proper_data[:, :, center_z]
    conservative_slice = conservative_data[:, :, center_z]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original_slice.T, cmap='gray', origin='lower')
    plt.title(f'Original\n{original_nonzero} voxels')
    plt.colorbar()
    
    plt.subplot(1, 4, 2)
    plt.imshow(proper_slice.T, cmap='gray', origin='lower')
    plt.title(f'Proper Smoothed\n{proper_nonzero} voxels\n(+{(proper_nonzero-original_nonzero)/original_nonzero*100:.1f}%)')
    plt.colorbar()
    
    plt.subplot(1, 4, 3)
    plt.imshow(conservative_slice.T, cmap='gray', origin='lower')
    plt.title(f'Conservative\n{conservative_nonzero} voxels\n({(conservative_nonzero-original_nonzero)/original_nonzero*100:.1f}%)')
    plt.colorbar()
    
    plt.subplot(1, 4, 4)
    # Show difference between proper and original
    diff_slice = proper_slice - original_slice
    plt.imshow(diff_slice.T, cmap='RdBu', origin='lower')
    plt.title('Proper - Original\n(Added smoothing)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'fixed_comparison_{original_path.stem}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return original_nonzero, proper_nonzero, conservative_nonzero

def main():
    # Define paths
    original_dir = Path("/home/jiwoo/urp/data/uan/original")
    fixed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_fixed")
    
    # Find test files
    test_files = [
        "30_MRA2_seg.nii.gz",
        "30_MRA1_seg.nii.gz", 
        "76_MRA1_seg.nii.gz"
    ]
    
    print("=== Fixed Smoothing Quality Analysis ===")
    
    total_original = 0
    total_proper = 0
    total_conservative = 0
    
    for original_name in test_files:
        original_path = original_dir / original_name
        proper_name = original_name.replace('.nii.gz', '_smoothed_proper.nii.gz')
        conservative_name = original_name.replace('.nii.gz', '_smoothed_conservative.nii.gz')
        proper_path = fixed_dir / proper_name
        conservative_path = fixed_dir / conservative_name
        
        if all(p.exists() for p in [original_path, proper_path, conservative_path]):
            orig, prop, cons = compare_files(original_path, proper_path, conservative_path)
            total_original += orig
            total_proper += prop
            total_conservative += cons
        else:
            print(f"Missing files for: {original_name}")
    
    print(f"\n=== Overall Summary ===")
    print(f"Total original voxels:     {total_original}")
    print(f"Total proper voxels:       {total_proper} (ratio: {total_proper/total_original:.3f})")
    print(f"Total conservative voxels: {total_conservative} (ratio: {total_conservative/total_original:.3f})")
    
    print(f"\n=== Quality Assessment ===")
    proper_change = (total_proper - total_original) / total_original * 100
    conservative_change = (total_conservative - total_original) / total_original * 100
    
    print(f"Proper smoothing: {proper_change:+.1f}% volume change")
    print(f"Conservative smoothing: {conservative_change:+.1f}% volume change")
    
    if abs(proper_change) < 30:
        print("✓ Proper smoothing: Reasonable volume change")
    else:
        print("⚠️ Proper smoothing: Large volume change")
        
    if abs(conservative_change) < 5:
        print("✓ Conservative smoothing: Minimal volume change")
    else:
        print("⚠️ Conservative smoothing: Unexpected volume change")

if __name__ == "__main__":
    main() 