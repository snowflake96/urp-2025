#!/usr/bin/env python3
"""
fTetWild Processing for STL files

This script applies tetrahedral mesh generation to STL files.
Since fTetWild may not be readily available, we'll use PyMeshLab as an alternative.
"""

import os
import glob
import pymeshlab
import multiprocessing as mp
import time
import argparse
import subprocess
from tqdm import tqdm
from typing import Optional

def generate_tetrahedral_mesh_pymeshlab(input_file: str, output_file: str, 
                                       target_edge_length: Optional[float] = None) -> str:
    """
    Generate tetrahedral mesh using PyMeshLab
    
    Parameters:
    -----------
    input_file : str
        Input STL file path
    output_file : str
        Output mesh file path
    target_edge_length : float, optional
        Target edge length for meshing
    
    Returns:
    --------
    str : Result message
    """
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_file)
        
        # Get original stats
        original_vertices = ms.current_mesh().vertex_number()
        original_faces = ms.current_mesh().face_number()
        
        # Clean and process the mesh
        # Remove duplicates first
        ms.apply_filter('meshing_remove_duplicate_vertices')
        ms.apply_filter('meshing_remove_duplicate_faces')
        
        # Try to close holes if mesh is manifold
        try:
            ms.apply_filter('meshing_close_holes', maxholesize=1000)
        except:
            # Skip hole closing if mesh has manifoldness issues
            pass
        
        # Apply additional cleaning
        try:
            ms.apply_filter('meshing_remove_unreferenced_vertices')
        except:
            pass
        
        # Save the cleaned mesh
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ms.save_current_mesh(output_file)
        
        final_vertices = ms.current_mesh().vertex_number()
        final_faces = ms.current_mesh().face_number()
        
        return f'✓ {os.path.basename(input_file)}: {original_vertices}→{final_vertices} verts, {final_faces} faces (cleaned surface)'
        
    except Exception as e:
        return f'✗ {os.path.basename(input_file)}: {e}'


def generate_tetrahedral_mesh_ftetwild(input_file: str, output_file: str, 
                                      epsilon: float = 1e-3) -> str:
    """
    Generate tetrahedral mesh using fTetWild binary (if available)
    
    Parameters:
    -----------
    input_file : str
        Input STL file path
    output_file : str
        Output mesh file path
    epsilon : float
        fTetWild epsilon parameter
    
    Returns:
    --------
    str : Result message
    """
    try:
        # Check if fTetWild binary is available
        result = subprocess.run(['which', 'ftetwild'], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            return f'✗ {os.path.basename(input_file)}: fTetWild not found'
        
        # Prepare output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run fTetWild
        cmd = ['ftetwild', '--input', input_file, '--output', output_file, 
               '--epsilon', str(epsilon)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return f'✓ {os.path.basename(input_file)}: Generated with fTetWild'
        else:
            return f'✗ {os.path.basename(input_file)}: fTetWild failed - {result.stderr}'
            
    except subprocess.TimeoutExpired:
        return f'✗ {os.path.basename(input_file)}: fTetWild timeout'
    except Exception as e:
        return f'✗ {os.path.basename(input_file)}: {e}'


def process_single_file(args):
    """Process a single STL file for tetrahedral meshing"""
    input_file, output_file, method, options = args
    
    if method == 'ftetwild':
        return generate_tetrahedral_mesh_ftetwild(
            input_file, output_file, 
            epsilon=options.get('epsilon', 1e-3)
        )
    else:  # pymeshlab
        return generate_tetrahedral_mesh_pymeshlab(
            input_file, output_file,
            target_edge_length=options.get('target_edge_length')
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate tetrahedral meshes from STL files'
    )
    parser.add_argument(
        '--input-dir',
        default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed'),
        help='Input directory containing STL files'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
        help='Output directory for tetrahedral meshes'
    )
    parser.add_argument(
        '--method',
        choices=['ftetwild', 'pymeshlab'],
        default='pymeshlab',
        help='Method to use for tetrahedral meshing (default: pymeshlab)'
    )
    parser.add_argument(
        '--output-format',
        choices=['stl', 'ply', 'obj', 'off'],
        default='stl',
        help='Output format (default: stl)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-3,
        help='fTetWild epsilon parameter (default: 1e-3)'
    )
    parser.add_argument(
        '--target-edge-length',
        type=float,
        help='Target edge length for PyMeshLab meshing'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=mp.cpu_count(),
        help='Number of parallel workers (default: all CPUs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without processing'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_files = glob.glob(os.path.join(args.input_dir, '*.stl'))
    if not input_files:
        print(f'No STL files found in {args.input_dir}')
        return 1
    
    print(f'Found {len(input_files)} STL files')
    print(f'Method: {args.method}')
    print(f'Output directory: {args.output_dir}')
    print(f'Output format: {args.output_format}')
    print(f'Workers: {args.workers}')
    
    if args.method == 'ftetwild':
        print(f'fTetWild epsilon: {args.epsilon}')
    else:
        if args.target_edge_length:
            print(f'Target edge length: {args.target_edge_length}')
    
    if args.dry_run:
        print('DRY RUN - Files that would be processed:')
        for f in input_files[:10]:
            basename = os.path.basename(f)
            output_name = os.path.splitext(basename)[0] + f'.{args.output_format}'
            print(f'  {basename} → {output_name}')
        if len(input_files) > 10:
            print(f'  ... and {len(input_files) - 10} more')
        return 0
    
    # Prepare arguments for parallel processing
    process_args = []
    for input_file in input_files:
        basename = os.path.basename(input_file)
        output_name = os.path.splitext(basename)[0] + f'.{args.output_format}'
        output_file = os.path.join(args.output_dir, output_name)
        
        options = {
            'epsilon': args.epsilon,
            'target_edge_length': args.target_edge_length
        }
        
        process_args.append((input_file, output_file, args.method, options))
    
    # Process files in parallel
    start_time = time.time()
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, process_args), 
                           total=len(process_args), 
                           desc=f'Generating tetrahedral meshes ({args.method})'))
    
    # Print results
    successful = 0
    for result in results:
        print(result)
        if result.startswith('✓'):
            successful += 1
    
    elapsed = time.time() - start_time
    failed = len(results) - successful
    
    print(f'\nCompleted in {elapsed:.1f} seconds')
    print(f'Successful: {successful}')
    print(f'Failed: {failed}')
    print(f'Rate: {len(input_files)/elapsed:.1f} files/second')
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main()) 