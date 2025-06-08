#!/usr/bin/env python3
"""
Fast Parallel Batch Processing for Kissing Artifact Removal

This script processes multiple vessel segmentation files in parallel,
using simple but effective morphological operations for speed.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import glob
from scipy.ndimage import (
    binary_erosion, binary_dilation, binary_closing,
    label, distance_transform_edt, gaussian_filter
)
from skimage.morphology import skeletonize, remove_small_objects
import argparse
from tqdm import tqdm


def setup_logger(log_file: Optional[str] = None):
    """Setup logging"""
    logger = logging.getLogger('FastKissingRemoval')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def fast_kissing_removal(mask: np.ndarray, voxel_size: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Fast kissing artifact removal using morphological operations
    
    Strategy:
    1. Identify thin bridges using erosion/dilation
    2. Find narrow connections in skeleton
    3. Remove identified bridges
    4. Clean up result
    """
    start_time = time.time()
    stats = {
        'original_voxels': int(mask.sum()),
        'thin_regions': 0,
        'bridges_removed': 0,
        'final_voxels': 0
    }
    
    # Step 1: Find thin regions (potential bridges)
    # These are regions that disappear with minimal erosion
    eroded1 = binary_erosion(mask, iterations=1)
    eroded2 = binary_erosion(mask, iterations=2)
    
    # Thin regions are those that exist in original but not after 2 erosions
    thin_regions = mask & ~eroded2
    stats['thin_regions'] = int(thin_regions.sum())
    
    # Step 2: Skeleton analysis for bridge detection
    skeleton = skeletonize(mask)
    
    # Find skeleton points in thin regions (likely bridges)
    bridge_skeleton = skeleton & thin_regions
    
    # Step 3: Distance-based bridge detection
    # Calculate distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Find narrow points (distance < threshold)
    narrow_threshold = 2.0 * np.mean(voxel_size)  # ~2mm
    narrow_mask = (dist_transform < narrow_threshold) & (dist_transform > 0)
    
    # Combine with skeleton to find bridge points
    bridge_candidates = bridge_skeleton & narrow_mask
    
    # Step 4: Remove bridges by breaking at narrow points
    if bridge_candidates.any():
        # Dilate bridge points to ensure disconnection
        bridges_to_remove = binary_dilation(bridge_candidates, iterations=2)
        
        # Remove bridges
        cleaned_mask = mask & ~bridges_to_remove
        stats['bridges_removed'] = int(bridges_to_remove.sum())
        
        # Step 5: Keep only largest connected component
        labeled, num_features = label(cleaned_mask)
        if num_features > 1:
            # Find largest component
            component_sizes = np.array([
                (labeled == i).sum() for i in range(1, num_features + 1)
            ])
            largest_component = np.argmax(component_sizes) + 1
            cleaned_mask = (labeled == largest_component)
    else:
        cleaned_mask = mask
    
    # Step 6: Morphological closing to smooth result
    cleaned_mask = binary_closing(cleaned_mask, iterations=1)
    
    # Remove small isolated components
    cleaned_mask = remove_small_objects(cleaned_mask, min_size=100)
    
    stats['final_voxels'] = int(cleaned_mask.sum())
    stats['processing_time'] = time.time() - start_time
    stats['voxels_removed'] = stats['original_voxels'] - stats['final_voxels']
    stats['percent_removed'] = 100 * stats['voxels_removed'] / max(1, stats['original_voxels'])
    
    return cleaned_mask, stats


def process_single_file(args: Tuple[str, str, Dict]) -> Dict:
    """
    Process a single file (for parallel processing)
    
    Parameters:
    -----------
    args : tuple
        (input_file, output_file, options)
    
    Returns:
    --------
    dict : Processing results
    """
    input_file, output_file, options = args
    
    result = {
        'input': input_file,
        'output': output_file,
        'success': False,
        'error': None,
        'stats': {}
    }
    
    try:
        # Load NIfTI
        nifti = nib.load(input_file)
        data = nifti.get_fdata()
        affine = nifti.affine
        voxel_size = np.abs(np.diag(affine)[:3])
        
        # Convert to binary mask
        mask = data > 0.5
        
        # Apply fast kissing removal
        cleaned_mask, stats = fast_kissing_removal(mask, voxel_size)
        
        # Save result
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        cleaned_nifti = nib.Nifti1Image(
            cleaned_mask.astype(np.float32), affine
        )
        nib.save(cleaned_nifti, output_file)
        
        result['success'] = True
        result['stats'] = stats
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_process_parallel(
    input_files: List[str],
    output_dir: str,
    num_workers: Optional[int] = None,
    output_suffix: str = '_no_kissing'
) -> List[Dict]:
    """
    Process multiple files in parallel
    
    Parameters:
    -----------
    input_files : list
        List of input NIfTI files
    output_dir : str
        Output directory
    num_workers : int, optional
        Number of parallel workers (default: CPU count)
    output_suffix : str
        Suffix to add to output files
    
    Returns:
    --------
    list : Results for each file
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    logger = logging.getLogger('FastKissingRemoval')
    logger.info(f"Processing {len(input_files)} files with {num_workers} workers")
    
    # Prepare arguments for parallel processing
    process_args = []
    for input_file in input_files:
        # Generate output filename
        base_name = os.path.basename(input_file)
        name_parts = base_name.split('.')
        output_name = name_parts[0] + output_suffix + '.' + '.'.join(name_parts[1:])
        output_file = os.path.join(output_dir, output_name)
        
        process_args.append((input_file, output_file, {}))
    
    # Process in parallel with progress bar
    results = []
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_single_file, process_args):
                results.append(result)
                pbar.update(1)
                
                # Log result
                if result['success']:
                    stats = result['stats']
                    logger.info(
                        f"✓ {os.path.basename(result['input'])}: "
                        f"removed {stats['voxels_removed']} voxels "
                        f"({stats['percent_removed']:.1f}%) in {stats['processing_time']:.1f}s"
                    )
                else:
                    logger.error(
                        f"✗ {os.path.basename(result['input'])}: {result['error']}"
                    )
    
    return results


def find_input_files(input_pattern: str) -> List[str]:
    """Find all input files matching pattern"""
    # Handle different input types
    if os.path.isdir(input_pattern):
        # Directory: find all NIfTI files
        patterns = [
            os.path.join(input_pattern, "*.nii.gz"),
            os.path.join(input_pattern, "*.nii"),
            os.path.join(input_pattern, "*_seg.nii.gz"),
            os.path.join(input_pattern, "*MRA*seg*.nii.gz")
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        return sorted(list(set(files)))  # Remove duplicates
    elif '*' in input_pattern or '?' in input_pattern:
        # Glob pattern
        return sorted(glob.glob(input_pattern))
    else:
        # Single file
        return [input_pattern] if os.path.exists(input_pattern) else []


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fast parallel batch processing for kissing artifact removal"
    )
    parser.add_argument(
        'input',
        help='Input pattern: directory, glob pattern, or single file'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        help='Number of parallel workers (default: all CPUs)'
    )
    parser.add_argument(
        '--suffix',
        default='_no_kissing',
        help='Suffix for output files (default: _no_kissing)'
    )
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.log_file)
    
    # Find input files
    input_files = find_input_files(args.input)
    
    if not input_files:
        logger.error(f"No input files found matching: {args.input}")
        return 1
    
    logger.info(f"Found {len(input_files)} files to process")
    
    if args.dry_run:
        logger.info("Dry run - files that would be processed:")
        for f in input_files:
            logger.info(f"  {f}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start processing
    start_time = time.time()
    
    results = batch_process_parallel(
        input_files,
        args.output_dir,
        num_workers=args.workers,
        output_suffix=args.suffix
    )
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info("\n" + "="*60)
    logger.info(f"Processing complete in {total_time:.1f} seconds")
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if successful > 0:
        # Calculate average statistics
        total_removed = sum(
            r['stats']['voxels_removed'] 
            for r in results if r['success']
        )
        avg_percent = np.mean([
            r['stats']['percent_removed'] 
            for r in results if r['success']
        ])
        avg_time = np.mean([
            r['stats']['processing_time'] 
            for r in results if r['success']
        ])
        
        logger.info(f"\nAverage statistics:")
        logger.info(f"  Voxels removed per file: {total_removed/successful:.0f}")
        logger.info(f"  Percent removed: {avg_percent:.2f}%")
        logger.info(f"  Processing time per file: {avg_time:.2f}s")
        logger.info(f"  Total processing rate: {successful/total_time:.1f} files/second")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main()) 