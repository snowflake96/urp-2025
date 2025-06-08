#!/usr/bin/env python3
"""
Convert aneurysm coordinates from NIfTI voxel space to STL mesh space using normalized coordinates.
This approach normalizes the voxel coordinates within NIfTI dimensions and maps them to STL mesh bounds.
"""

import json
import os
import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import argparse
from sklearn.neighbors import NearestNeighbors
import time

def normalize_coordinates(voxel_coords, nifti_shape):
    """Convert voxel coordinates to normalized coordinates [0,1]."""
    normalized = np.array(voxel_coords) / (np.array(nifti_shape) - 1)
    return normalized

def denormalize_to_mesh(normalized_coords, mesh_bounds):
    """Convert normalized coordinates to mesh coordinate space."""
    min_bounds = np.array(mesh_bounds[0])
    max_bounds = np.array(mesh_bounds[1])
    
    # Map normalized coordinates to mesh bounds
    mesh_coords = min_bounds + normalized_coords * (max_bounds - min_bounds)
    return mesh_coords

def find_closest_vertex(target_point, mesh_vertices):
    """Find the closest vertex in the mesh to the target point."""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(mesh_vertices)
    distances, indices = nbrs.kneighbors([target_point])
    
    closest_vertex_idx = indices[0][0]
    closest_vertex_coords = mesh_vertices[closest_vertex_idx]
    distance = distances[0][0]
    
    return closest_vertex_idx, closest_vertex_coords, distance

def process_patient(patient_id, aneurysm_data, nifti_dir, stl_dir):
    """Process a single patient to convert aneurysm coordinates."""
    try:
        # Load NIfTI file
        nifti_filename = f"{patient_id}.nii.gz"
        nifti_path = os.path.join(nifti_dir, nifti_filename)
        
        if not os.path.exists(nifti_path):
            print(f"  ❌ NIfTI file not found: {nifti_filename}")
            return None
            
        nifti_img = nib.load(nifti_path)
        nifti_shape = nifti_img.shape
        
        # Load STL mesh
        stl_filename = f"{patient_id}.stl"
        stl_path = os.path.join(stl_dir, stl_filename)
        
        if not os.path.exists(stl_path):
            print(f"  ❌ STL file not found: {stl_filename}")
            return None
            
        mesh = trimesh.load_mesh(stl_path)
        mesh_bounds = [mesh.bounds[0].tolist(), mesh.bounds[1].tolist()]
        
        # Process each aneurysm
        converted_aneurysms = []
        for aneurysm in aneurysm_data:
            # Get original voxel coordinates
            voxel_coords = aneurysm['coords']
            
            # Normalize coordinates within NIfTI space
            normalized_coords = normalize_coordinates(voxel_coords, nifti_shape)
            
            # Map to mesh coordinate space
            mesh_space_coords = denormalize_to_mesh(normalized_coords, mesh_bounds)
            
            # Find closest mesh vertex
            closest_vertex_idx, closest_vertex_coords, distance = find_closest_vertex(
                mesh_space_coords, mesh.vertices
            )
            
            converted_aneurysm = {
                "aneurysm_id": aneurysm.get('aneurysm_id', 'aneurysm_1'),
                "description": aneurysm.get('description', ''),
                "original_voxel_coords": voxel_coords,
                "normalized_coords": normalized_coords.tolist(),
                "mesh_space_coords": mesh_space_coords.tolist(),
                "mesh_vertex_index": int(closest_vertex_idx),
                "mesh_vertex_coords": closest_vertex_coords.tolist(),
                "distance_to_mesh": float(distance)
            }
            converted_aneurysms.append(converted_aneurysm)
        
        # Compile patient data
        patient_data = {
            "aneurysms": converted_aneurysms,
            "nifti_info": {
                "shape": list(nifti_shape),
                "affine": nifti_img.affine.tolist(),
                "voxel_size": nifti_img.header.get_zooms()[:3]
            },
            "mesh_info": {
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "bounds": mesh_bounds
            }
        }
        
        print(f"  ✓ {patient_id}: {len(converted_aneurysms)} aneurysm(s) converted")
        return patient_data
        
    except Exception as e:
        print(f"  ❌ Error processing {patient_id}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert aneurysm coordinates using normalized approach')
    parser.add_argument('--aneurysm-file', default='../all_patients_aneurysms.json',
                        help='Path to original aneurysm coordinates file')
    parser.add_argument('--nifti-dir', default='~/urp/data/uan/original',
                        help='Directory containing NIfTI files')
    parser.add_argument('--stl-dir', default='~/urp/data/uan/original_taubin_smoothed_ftetwild',
                        help='Directory containing STL files')
    parser.add_argument('--output-file', default='../all_patients_aneurysms_for_stl_normalized.json',
                        help='Output file for converted coordinates')
    
    args = parser.parse_args()
    
    # Expand paths
    nifti_dir = os.path.expanduser(args.nifti_dir)
    stl_dir = os.path.expanduser(args.stl_dir)
    aneurysm_file = args.aneurysm_file
    output_file = args.output_file
    
    print(f"Loading aneurysm coordinates from: {aneurysm_file}")
    
    # Load original aneurysm data
    with open(aneurysm_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"Found {len(original_data)} patients")
    print(f"NIfTI directory: {nifti_dir}")
    print(f"STL directory: {stl_dir}")
    
    # Process each patient
    start_time = time.time()
    converted_data = {}
    successful = 0
    
    for patient_id, aneurysm_data in original_data.items():
        print(f"Processing {patient_id}...")
        result = process_patient(patient_id, aneurysm_data, nifti_dir, stl_dir)
        
        if result is not None:
            converted_data[patient_id] = result
            successful += 1
    
    end_time = time.time()
    
    # Save converted data
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    # Summary
    total = len(original_data)
    print(f"\n{'='*60}")
    print(f"Coordinate conversion complete in {end_time - start_time:.1f} seconds")
    print(f"Successful patients: {successful}/{total}")
    
    if successful < total:
        print(f"Failed patients: {total - successful}")
    
    total_aneurysms = sum(len(data['aneurysms']) for data in converted_data.values())
    print(f"Total aneurysms converted: {total_aneurysms}")
    print(f"Converted coordinates saved to: {output_file}")
    
    print(f"\nNormalized coordinate approach:")
    print(f"1. Voxel coords normalized to [0,1] within NIfTI dimensions")
    print(f"2. Normalized coords mapped to STL mesh bounds")
    print(f"3. Closest mesh vertex found for each mapped coordinate")

if __name__ == "__main__":
    main() 