#!/usr/bin/env python3
"""
Taubin Smoothing for STL Files

This script applies Taubin smoothing to STL files in parallel.
Taubin smoothing preserves volume better than simple Laplacian smoothing.
"""

import os
import glob
import numpy as np
import trimesh
import pymeshlab
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def setup_logger(verbose: bool = True) -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger('TaubinSmoothing')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def taubin_smooth_mesh(mesh_path: str, 
                      output_path: str,
                      iterations: int = 10,
                      lambda_factor: float = 0.5,
                      mu_factor: float = -0.53,
                      method: str = 'pymeshlab') -> Dict:
    """
    Apply Taubin smoothing to a mesh
    
    Parameters:
    -----------
    mesh_path : str
        Input STL file path
    output_path : str
        Output STL file path
    iterations : int
        Number of smoothing iterations
    lambda_factor : float
        Lambda parameter for Taubin smoothing
    mu_factor : float
        Mu parameter for Taubin smoothing (should be negative)
    method : str
        Method to use: 'pymeshlab' or 'trimesh'
    
    Returns:
    --------
    dict : Processing results
    """
    result = {
        'input': mesh_path,
        'output': output_path,
        'success': False,
        'error': None,
        'stats': {}
    }
    
    try:
        start_time = time.time()
        
        if method == 'pymeshlab':
            # Use PyMeshLab for Taubin smoothing
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_path)
            
            # Get original stats
            original_vertices = ms.current_mesh().vertex_number()
            original_faces = ms.current_mesh().face_number()
            
            # Apply Taubin smoothing - use correct method
            for _ in range(iterations):
                ms.apply_coord_taubin_smoothing()
            
            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ms.save_current_mesh(output_path)
            
            result['stats'] = {
                'original_vertices': original_vertices,
                'original_faces': original_faces,
                'final_vertices': ms.current_mesh().vertex_number(),
                'final_faces': ms.current_mesh().face_number(),
                'iterations': iterations,
                'lambda': lambda_factor,
                'mu': mu_factor
            }
            
        else:  # trimesh method
            # Load mesh with trimesh
            mesh = trimesh.load(mesh_path)
            
            # Get original stats
            original_vertices = len(mesh.vertices)
            original_faces = len(mesh.faces)
            
            # Apply Taubin smoothing using trimesh
            # Note: trimesh doesn't have built-in Taubin, so we'll implement it
            smoothed_mesh = apply_taubin_smoothing_trimesh(
                mesh, iterations, lambda_factor, mu_factor
            )
            
            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            smoothed_mesh.export(output_path)
            
            result['stats'] = {
                'original_vertices': original_vertices,
                'original_faces': original_faces,
                'final_vertices': len(smoothed_mesh.vertices),
                'final_faces': len(smoothed_mesh.faces),
                'iterations': iterations,
                'lambda': lambda_factor,
                'mu': mu_factor
            }
        
        result['stats']['processing_time'] = time.time() - start_time
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def apply_taubin_smoothing_trimesh(mesh: trimesh.Trimesh,
                                  iterations: int,
                                  lambda_factor: float,
                                  mu_factor: float) -> trimesh.Trimesh:
    """
    Apply Taubin smoothing using trimesh
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    iterations : int
        Number of smoothing iterations
    lambda_factor : float
        Lambda parameter for Taubin smoothing
    mu_factor : float
        Mu parameter for Taubin smoothing (should be negative)
    
    Returns:
    --------
    trimesh.Trimesh : Smoothed mesh
    """
    smoothed_mesh = mesh.copy()
    
    for i in range(iterations):
        # Step 1: Laplacian smoothing with lambda
        smoothed_mesh = smoothed_mesh.smoothed(
            algorithm='laplacian',
            iterations=1,
            lamb=lambda_factor
        )
        
        # Step 2: Laplacian smoothing with mu (shrinkage correction)
        smoothed_mesh = smoothed_mesh.smoothed(
            algorithm='laplacian',
            iterations=1,
            lamb=mu_factor
        )
    
    return smoothed_mesh


def process_single_file(args: Tuple[str, str, Dict]) -> Dict:
    """
    Process a single STL file (for parallel processing)
    
    Parameters:
    -----------
    args : tuple
        (input_file, output_file, options)
    
    Returns:
    --------
    dict : Processing results
    """
    input_file, output_file, options = args
    
    return taubin_smooth_mesh(
        input_file,
        output_file,
        iterations=options.get('iterations', 10),
        lambda_factor=options.get('lambda_factor', 0.5),
        mu_factor=options.get('mu_factor', -0.53),
        method=options.get('method', 'pymeshlab')
    )


def batch_taubin_smoothing(input_dir: str,
                          output_dir: str,
                          pattern: str = "*.stl",
                          num_workers: Optional[int] = None,
                          **smooth_options) -> List[Dict]:
    """
    Apply Taubin smoothing to multiple STL files in parallel
    
    Parameters:
    -----------
    input_dir : str
        Input directory containing STL files
    output_dir : str
        Output directory for smoothed STL files
    pattern : str
        File pattern to match
    num_workers : int, optional
        Number of parallel workers
    **smooth_options : dict
        Additional options for smoothing
    
    Returns:
    --------
    list : Results for each file
    """
    logger = logging.getLogger('TaubinSmoothing')
    
    # Find input files
    input_pattern = os.path.join(input_dir, pattern)
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        logger.error(f"No files found matching: {input_pattern}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = []
    for input_file in input_files:
        filename = os.path.basename(input_file)
        # Keep same filename but in output directory
        output_file = os.path.join(output_dir, filename)
        process_args.append((input_file, output_file, smooth_options))
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(input_files))
    
    logger.info(f"Processing {len(input_files)} files with {num_workers} workers")
    
    # Process in parallel
    results = []
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Smoothing meshes") as pbar:
            for result in pool.imap_unordered(process_single_file, process_args):
                results.append(result)
                pbar.update(1)
                
                # Log result
                if result['success']:
                    stats = result['stats']
                    logger.info(
                        f"✓ {os.path.basename(result['input'])}: "
                        f"{stats['final_vertices']} vertices, "
                        f"{stats['final_faces']} faces "
                        f"({stats['processing_time']:.1f}s)"
                    )
                else:
                    logger.error(
                        f"✗ {os.path.basename(result['input'])}: {result['error']}"
                    )
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Apply Taubin smoothing to STL files"
    )
    parser.add_argument(
        '--input-dir',
        default=os.path.expanduser("~/urp/data/uan/original"),
        help='Input directory containing STL files'
    )
    parser.add_argument(
        '--output-dir', 
        default=os.path.expanduser("~/urp/data/uan/original_taubin_smoothed"),
        help='Output directory for smoothed STL files'
    )
    parser.add_argument(
        '--pattern',
        default="*.stl",
        help='File pattern to match (default: *.stl)'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=10,
        help='Number of smoothing iterations (default: 10)'
    )
    parser.add_argument(
        '--lambda-factor',
        type=float,
        default=0.5,
        help='Lambda factor for Taubin smoothing (default: 0.5)'
    )
    parser.add_argument(
        '--mu-factor',
        type=float,
        default=-0.53,
        help='Mu factor for Taubin smoothing (default: -0.53)'
    )
    parser.add_argument(
        '--method',
        choices=['pymeshlab', 'trimesh'],
        default='pymeshlab',
        help='Method to use for smoothing (default: pymeshlab)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        help='Number of parallel workers (default: auto)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.verbose)
    
    # Show configuration
    logger.info("Taubin Smoothing Configuration:")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  File pattern: {args.pattern}")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  Lambda factor: {args.lambda_factor}")
    logger.info(f"  Mu factor: {args.mu_factor}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Workers: {args.workers or 'auto'}")
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Find files
    input_pattern = os.path.join(args.input_dir, args.pattern)
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        logger.error(f"No files found matching: {input_pattern}")
        return 1
    
    logger.info(f"Found {len(input_files)} files to process")
    
    if args.dry_run:
        logger.info("Dry run - files that would be processed:")
        for f in input_files[:10]:  # Show first 10
            logger.info(f"  {f}")
        if len(input_files) > 10:
            logger.info(f"  ... and {len(input_files) - 10} more")
        return 0
    
    # Start processing
    start_time = time.time()
    
    results = batch_taubin_smoothing(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        num_workers=args.workers,
        iterations=args.iterations,
        lambda_factor=args.lambda_factor,
        mu_factor=args.mu_factor,
        method=args.method
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
        avg_time = np.mean([r['stats']['processing_time'] for r in results if r['success']])
        logger.info(f"Average processing time: {avg_time:.2f}s per file")
        logger.info(f"Processing rate: {successful/total_time:.1f} files/second")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main()) 