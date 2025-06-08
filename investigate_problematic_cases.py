#!/usr/bin/env python3
"""
Investigate problematic aneurysm location cases
"""

import json
import numpy as np
import nibabel as nib
import trimesh
import os

def investigate_case(patient_id, data, nifti_dir, stl_dir):
    """Investigate a specific patient case in detail."""
    if patient_id not in data:
        print(f"Patient {patient_id} not found in data")
        return
    
    patient_data = data[patient_id]
    aneurysm = patient_data['aneurysms'][0]
    
    print(f"=== {patient_id} ===")
    print(f"NIfTI shape: {patient_data['nifti_info']['shape']}")
    print(f"Original voxel coords: {aneurysm['original_voxel_coords']}")
    print(f"Normalized coords: {aneurysm['normalized_coords']}")
    print(f"Mesh bounds: {patient_data['mesh_info']['bounds']}")
    print(f"Calculated mesh coords: {aneurysm['mesh_space_coords']}")
    print(f"Final mesh vertex coords: {aneurysm['mesh_vertex_coords']}")
    print(f"Distance to closest vertex: {aneurysm['distance_to_mesh']:.2f}")
    
    # Load actual NIfTI to check data bounds
    nifti_path = os.path.join(nifti_dir, f"{patient_id}.nii.gz")
    if os.path.exists(nifti_path):
        nifti_img = nib.load(nifti_path)
        data_img = nifti_img.get_fdata()
        
        # Find actual data bounds (non-zero voxels)
        nonzero_coords = np.where(data_img > 0)
        if len(nonzero_coords[0]) > 0:
            actual_bounds = {
                'x': [int(nonzero_coords[0].min()), int(nonzero_coords[0].max())],
                'y': [int(nonzero_coords[1].min()), int(nonzero_coords[1].max())],
                'z': [int(nonzero_coords[2].min()), int(nonzero_coords[2].max())]
            }
            print(f"Actual data bounds in NIfTI: {actual_bounds}")
            
            # Check if aneurysm coords are within actual data
            voxel_coords = aneurysm['original_voxel_coords']
            is_within = (
                actual_bounds['x'][0] <= voxel_coords[0] <= actual_bounds['x'][1] and
                actual_bounds['y'][0] <= voxel_coords[1] <= actual_bounds['y'][1] and
                actual_bounds['z'][0] <= voxel_coords[2] <= actual_bounds['z'][1]
            )
            print(f"Aneurysm coords within actual data bounds: {is_within}")
            
            # Calculate normalized coords using actual data bounds
            normalized_with_data_bounds = [
                (voxel_coords[0] - actual_bounds['x'][0]) / (actual_bounds['x'][1] - actual_bounds['x'][0]),
                (voxel_coords[1] - actual_bounds['y'][0]) / (actual_bounds['y'][1] - actual_bounds['y'][0]),
                (voxel_coords[2] - actual_bounds['z'][0]) / (actual_bounds['z'][1] - actual_bounds['z'][0])
            ]
            print(f"Alternative normalized coords (using data bounds): {normalized_with_data_bounds}")
        
    print()

def main():
    # Load data
    with open('../all_patients_aneurysms_for_stl.json', 'r') as f:
        data = json.load(f)
    
    nifti_dir = os.path.expanduser('~/urp/data/uan/original')
    stl_dir = os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild')
    
    # Investigate problematic cases
    problematic_cases = ['26_MRA1_seg', '26_MRA2_seg']
    
    for case in problematic_cases:
        investigate_case(case, data, nifti_dir, stl_dir)

if __name__ == "__main__":
    main() 