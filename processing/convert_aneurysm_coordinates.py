#!/usr/bin/env python3
"""
Convert Aneurysm Coordinates from NIfTI to STL Space

This script converts aneurysm internal point coordinates from NIfTI voxel space
to STL mesh coordinate space for downstream random walk analysis.
"""

import json
import os
import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional


def load_nifti_with_transform(nifti_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NIfTI file and extract transformation information
    
    Returns:
    --------
    data : np.ndarray
        Image data
    affine : np.ndarray  
        Affine transformation matrix
    header : nibabel header
        NIfTI header with spacing information
    """
    nii = nib.load(nifti_path)
    return nii.get_fdata(), nii.affine, nii.header


def voxel_to_world_coordinates(voxel_coords: List[int], affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to world coordinates using affine transformation
    
    Parameters:
    -----------
    voxel_coords : List[int]
        Voxel coordinates [x, y, z]
    affine : np.ndarray
        4x4 affine transformation matrix
    
    Returns:
    --------
    np.ndarray : World coordinates [x, y, z]
    """
    # Convert to homogeneous coordinates
    voxel_homogeneous = np.array([voxel_coords[0], voxel_coords[1], voxel_coords[2], 1])
    
    # Apply affine transformation
    world_homogeneous = affine @ voxel_homogeneous
    
    # Return world coordinates (drop homogeneous coordinate)
    return world_homogeneous[:3]


def find_closest_mesh_vertex(world_coord: np.ndarray, mesh: trimesh.Trimesh) -> Tuple[int, np.ndarray, float]:
    """
    Find the closest vertex in the mesh to the given world coordinate
    
    Parameters:
    -----------
    world_coord : np.ndarray
        World coordinate [x, y, z]
    mesh : trimesh.Trimesh
        Mesh object
    
    Returns:
    --------
    tuple : (vertex_index, vertex_coordinate, distance)
    """
    # Calculate distances to all vertices
    distances = np.linalg.norm(mesh.vertices - world_coord, axis=1)
    
    # Find closest vertex
    closest_idx = np.argmin(distances)
    closest_vertex = mesh.vertices[closest_idx]
    closest_distance = distances[closest_idx]
    
    return closest_idx, closest_vertex, closest_distance


def convert_single_patient(patient_id: str, 
                          aneurysm_data: Dict,
                          nifti_dir: str,
                          stl_dir: str) -> Dict:
    """
    Convert aneurysm coordinates for a single patient
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier (e.g., "06_MRA1_seg")
    aneurysm_data : Dict
        Aneurysm data for this patient
    nifti_dir : str
        Directory containing NIfTI files
    stl_dir : str
        Directory containing STL files
    
    Returns:
    --------
    Dict : Converted aneurysm data with mesh coordinates
    """
    # Find corresponding files
    nifti_path = os.path.join(nifti_dir, f"{patient_id}.nii.gz")
    stl_path = os.path.join(stl_dir, f"{patient_id}.stl")
    
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    print(f"Processing {patient_id}...")
    
    # Load NIfTI file and get transformation
    nifti_data, affine, header = load_nifti_with_transform(nifti_path)
    
    # Load STL mesh
    mesh = trimesh.load(stl_path)
    
    # Convert each aneurysm's coordinates
    converted_aneurysms = []
    
    for aneurysm in aneurysm_data['aneurysms']:
        # Get original voxel coordinates
        voxel_coords = aneurysm['internal_point']
        
        # Convert to world coordinates
        world_coords = voxel_to_world_coordinates(voxel_coords, affine)
        
        # Find closest mesh vertex
        vertex_idx, vertex_coords, distance = find_closest_mesh_vertex(world_coords, mesh)
        
        # Create converted aneurysm entry
        converted_aneurysm = {
            'aneurysm_id': aneurysm['aneurysm_id'],
            'description': aneurysm['description'],
            'original_voxel_coords': voxel_coords,
            'world_coords': world_coords.tolist(),
            'mesh_vertex_index': int(vertex_idx),
            'mesh_vertex_coords': vertex_coords.tolist(),
            'distance_to_mesh': float(distance)
        }
        
        converted_aneurysms.append(converted_aneurysm)
        
        print(f"  {aneurysm['aneurysm_id']}: voxel {voxel_coords} -> world {world_coords} -> vertex {vertex_idx} (dist: {distance:.2f})")
    
    return {
        'aneurysms': converted_aneurysms,
        'nifti_info': {
            'shape': list(nifti_data.shape),
            'affine': affine.tolist(),
            'voxel_size': [float(header.get_zooms()[i]) for i in range(3)]
        },
        'mesh_info': {
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'bounds': mesh.bounds.tolist()
        }
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert aneurysm coordinates from NIfTI to STL space'
    )
    parser.add_argument(
        '--input-json',
        default='../all_patients_aneurysms.json',
        help='Input JSON file with NIfTI coordinates'
    )
    parser.add_argument(
        '--output-json',
        default='../all_patients_aneurysms_for_stl.json',
        help='Output JSON file with STL coordinates'
    )
    parser.add_argument(
        '--nifti-dir',
        default=os.path.expanduser('~/urp/data/uan/original'),
        help='Directory containing NIfTI files'
    )
    parser.add_argument(
        '--stl-dir',
        default=os.path.expanduser('~/urp/data/uan/original'),
        help='Directory containing STL files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Load input JSON
    print(f"Loading input JSON: {args.input_json}")
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
    
    print(f"Found {len(input_data)} patients")
    
    # Convert coordinates for each patient
    converted_data = {}
    successful = 0
    failed = 0
    
    for patient_id, aneurysm_data in input_data.items():
        try:
            converted_data[patient_id] = convert_single_patient(
                patient_id, aneurysm_data, args.nifti_dir, args.stl_dir
            )
            successful += 1
        except Exception as e:
            print(f"ERROR processing {patient_id}: {e}")
            failed += 1
            continue
    
    # Save converted data
    print(f"\nSaving converted coordinates to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    # Summary
    print(f"\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total aneurysms converted: {sum(len(data['aneurysms']) for data in converted_data.values())}")
    
    # Show some example conversions
    if converted_data and args.verbose:
        print(f"\nExample conversions:")
        for patient_id, data in list(converted_data.items())[:3]:
            print(f"\n{patient_id}:")
            for aneurysm in data['aneurysms']:
                print(f"  {aneurysm['aneurysm_id']} ({aneurysm['description']}):")
                print(f"    Voxel: {aneurysm['original_voxel_coords']}")
                print(f"    World: [{aneurysm['world_coords'][0]:.1f}, {aneurysm['world_coords'][1]:.1f}, {aneurysm['world_coords'][2]:.1f}]")
                print(f"    Mesh vertex: {aneurysm['mesh_vertex_index']} at [{aneurysm['mesh_vertex_coords'][0]:.1f}, {aneurysm['mesh_vertex_coords'][1]:.1f}, {aneurysm['mesh_vertex_coords'][2]:.1f}]")
                print(f"    Distance: {aneurysm['distance_to_mesh']:.2f}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main()) 