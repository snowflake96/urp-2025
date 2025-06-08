#!/usr/bin/env python3
"""
Validate aneurysm locations by creating spheres at the aneurysm coordinates
and combining them with the original STL meshes for visual inspection.
"""

import json
import os
import trimesh
import numpy as np
from pathlib import Path
import argparse
from joblib import Parallel, delayed
import time

def create_sphere_at_location(center, radius=2.0, subdivisions=2):
    """Create a sphere at the specified location."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.vertices += center
    return sphere

def process_patient(patient_data, input_dir, output_dir):
    """Process a single patient: load STL, add aneurysm sphere, save result."""
    patient_id = patient_data['patient_id']
    
    try:
        # Load the STL mesh
        stl_filename = f"{patient_id}.stl"
        stl_path = os.path.join(input_dir, stl_filename)
        
        if not os.path.exists(stl_path):
            print(f"  ❌ STL file not found: {stl_filename}")
            return False
            
        mesh = trimesh.load_mesh(stl_path)
        
        # Create spheres for each aneurysm
        spheres = []
        for aneurysm in patient_data['aneurysms']:
            # Use the mesh vertex coordinates (closest point on mesh to aneurysm)
            aneurysm_location = np.array(aneurysm['mesh_vertex_coords'])
            
            # Create a sphere at the aneurysm location
            sphere = create_sphere_at_location(aneurysm_location, radius=3.0)
            spheres.append(sphere)
        
        # Combine mesh with spheres
        if spheres:
            combined_meshes = [mesh] + spheres
            combined_mesh = trimesh.util.concatenate(combined_meshes)
        else:
            combined_mesh = mesh
        
        # Save the result
        output_path = os.path.join(output_dir, stl_filename)
        combined_mesh.export(output_path)
        
        print(f"  ✓ {patient_id}: Added {len(spheres)} sphere(s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {patient_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate aneurysm locations by adding spheres')
    parser.add_argument('--aneurysm-file', default='../all_patients_aneurysms_for_stl.json',
                        help='Path to aneurysm coordinates file')
    parser.add_argument('--input-dir', default='~/urp/data/uan/original_taubin_smoothed_ftetwild',
                        help='Input directory with STL files')
    parser.add_argument('--output-dir', default='~/urp/data/uan/original_stl_location_test',
                        help='Output directory for validation files')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--sphere-radius', type=float, default=3.0,
                        help='Radius of validation spheres')
    
    args = parser.parse_args()
    
    # Expand paths
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    aneurysm_file = args.aneurysm_file
    
    print(f"Loading aneurysm coordinates from: {aneurysm_file}")
    
    # Load aneurysm data
    with open(aneurysm_file, 'r') as f:
        aneurysm_data = json.load(f)
    
    print(f"Found {len(aneurysm_data)} patients")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process patients in parallel
    start_time = time.time()
    
    def process_wrapper(patient_data):
        return process_patient(patient_data, input_dir, output_dir)
    
    print(f"Processing {len(aneurysm_data)} patients with {args.workers} workers...")
    results = Parallel(n_jobs=args.workers)(
        delayed(process_wrapper)(patient_data) 
        for patient_data in aneurysm_data
    )
    
    end_time = time.time()
    
    # Summary
    successful = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Validation file generation complete in {end_time - start_time:.1f} seconds")
    print(f"Successful patients: {successful}/{total}")
    
    if successful < total:
        print(f"Failed patients: {total - successful}")
    
    print(f"Validation files saved to: {output_dir}")
    print(f"\nTo validate locations:")
    print(f"1. Open the STL files in a 3D viewer (e.g., MeshLab, ParaView)")
    print(f"2. Look for small spheres - they should be located inside or near aneurysms")
    print(f"3. If spheres are in wrong locations, the coordinate conversion needs adjustment")

if __name__ == "__main__":
    main() 