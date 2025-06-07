#!/usr/bin/env python3
"""
Coordinate Transformer for Aneurysm Locations

Transforms aneurysm coordinates between different coordinate systems:
- NIfTI voxel space (i,j,k indices)
- World/RAS space (x,y,z in mm)
- Mesh vertex space (for STL/PLY files)
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
import trimesh
from typing import Dict, List, Tuple, Optional


class CoordinateTransformer:
    """Handle coordinate transformations for aneurysm locations"""
    
    def __init__(self, nifti_file: str):
        """
        Initialize with reference NIfTI file
        
        Parameters:
        -----------
        nifti_file : str
            Path to NIfTI file that defines the coordinate system
        """
        self.nifti = nib.load(nifti_file)
        self.affine = self.nifti.affine
        self.inv_affine = np.linalg.inv(self.affine)
        self.shape = self.nifti.shape
        
    def voxel_to_world(self, voxel_coords: List[float]) -> np.ndarray:
        """
        Transform voxel coordinates to world/RAS coordinates
        
        Parameters:
        -----------
        voxel_coords : list
            [i, j, k] voxel indices
            
        Returns:
        --------
        np.ndarray : [x, y, z] world coordinates in mm
        """
        voxel_homogeneous = np.append(voxel_coords, 1)
        world_coords = np.dot(self.affine, voxel_homogeneous)[:3]
        return world_coords
    
    def world_to_voxel(self, world_coords: List[float]) -> np.ndarray:
        """
        Transform world/RAS coordinates to voxel coordinates
        
        Parameters:
        -----------
        world_coords : list
            [x, y, z] world coordinates in mm
            
        Returns:
        --------
        np.ndarray : [i, j, k] voxel indices
        """
        world_homogeneous = np.append(world_coords, 1)
        voxel_coords = np.dot(self.inv_affine, world_homogeneous)[:3]
        return voxel_coords
    
    def find_closest_mesh_vertex(self, world_point: np.ndarray, 
                               mesh: trimesh.Trimesh,
                               max_distance: float = 10.0) -> Optional[int]:
        """
        Find closest vertex in mesh to given world coordinate
        
        Parameters:
        -----------
        world_point : np.ndarray
            [x, y, z] world coordinates
        mesh : trimesh.Trimesh
            Target mesh
        max_distance : float
            Maximum allowed distance (mm)
            
        Returns:
        --------
        int : Vertex index or None if too far
        """
        distances = np.linalg.norm(mesh.vertices - world_point, axis=1)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance <= max_distance:
            return min_idx
        return None
    
    def transform_aneurysm_data(self, aneurysm_json: str, 
                              output_json: str,
                              mesh_file: Optional[str] = None) -> Dict:
        """
        Transform all aneurysm coordinates and save to new JSON
        
        Parameters:
        -----------
        aneurysm_json : str
            Input JSON file with voxel coordinates
        output_json : str
            Output JSON file with transformed coordinates
        mesh_file : str, optional
            If provided, also find closest mesh vertices
            
        Returns:
        --------
        dict : Transformed aneurysm data
        """
        # Load original data
        with open(aneurysm_json, 'r') as f:
            data = json.load(f)
        
        # Load mesh if provided
        mesh = None
        if mesh_file:
            mesh = trimesh.load(mesh_file)
        
        # Transform each patient's data
        transformed_data = {}
        
        for patient_id, patient_data in data.items():
            if 'aneurysms' not in patient_data:
                continue
            
            transformed_aneurysms = []
            
            for aneurysm in patient_data['aneurysms']:
                # Get voxel coordinates
                voxel_coords = aneurysm['internal_point']
                
                # Transform to world coordinates
                world_coords = self.voxel_to_world(voxel_coords)
                
                # Create transformed entry
                transformed = {
                    'aneurysm_id': aneurysm['aneurysm_id'],
                    'description': aneurysm['description'],
                    'voxel_coordinates': voxel_coords,
                    'world_coordinates': world_coords.tolist(),
                    'world_coordinates_mm': {
                        'x': float(world_coords[0]),
                        'y': float(world_coords[1]), 
                        'z': float(world_coords[2])
                    }
                }
                
                # Find closest mesh vertex if mesh provided
                if mesh:
                    vertex_idx = self.find_closest_mesh_vertex(world_coords, mesh)
                    if vertex_idx is not None:
                        transformed['mesh_vertex_index'] = int(vertex_idx)
                        transformed['mesh_vertex_coords'] = mesh.vertices[vertex_idx].tolist()
                        transformed['distance_to_mesh'] = float(
                            np.linalg.norm(mesh.vertices[vertex_idx] - world_coords)
                        )
                
                transformed_aneurysms.append(transformed)
            
            transformed_data[patient_id] = {
                'aneurysms': transformed_aneurysms,
                'coordinate_system': {
                    'nifti_shape': self.shape,
                    'affine_matrix': self.affine.tolist(),
                    'mesh_file': mesh_file if mesh_file else None
                }
            }
        
        # Save transformed data
        with open(output_json, 'w') as f:
            json.dump(transformed_data, f, indent=2)
        
        print(f"Transformed coordinates saved to: {output_json}")
        return transformed_data
    
    def create_visualization_markers(self, aneurysm_data: Dict,
                                   output_dir: str,
                                   marker_radius: float = 3.0) -> None:
        """
        Create spherical mesh markers at aneurysm locations for visualization
        
        Parameters:
        -----------
        aneurysm_data : dict
            Transformed aneurysm data with world coordinates
        output_dir : str
            Directory to save marker meshes
        marker_radius : float
            Radius of marker spheres in mm
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_markers = []
        
        for patient_id, patient_data in aneurysm_data.items():
            if 'aneurysms' not in patient_data:
                continue
            
            patient_markers = []
            
            for aneurysm in patient_data['aneurysms']:
                # Get world coordinates
                world_coords = np.array(aneurysm['world_coordinates'])
                
                # Create sphere marker
                sphere = trimesh.creation.uv_sphere(radius=marker_radius)
                sphere.apply_translation(world_coords)
                
                # Color based on location
                if aneurysm['description'] == 'ICA':
                    sphere.visual.vertex_colors = [255, 0, 0, 255]  # Red
                elif aneurysm['description'] == 'MCA':
                    sphere.visual.vertex_colors = [0, 255, 0, 255]  # Green
                elif aneurysm['description'] == 'Acom':
                    sphere.visual.vertex_colors = [0, 0, 255, 255]  # Blue
                else:
                    sphere.visual.vertex_colors = [255, 255, 0, 255]  # Yellow
                
                patient_markers.append(sphere)
                all_markers.append(sphere)
            
            # Save patient-specific markers
            if patient_markers:
                combined = trimesh.util.concatenate(patient_markers)
                combined.export(output_path / f"{patient_id}_aneurysm_markers.stl")
        
        # Save all markers combined
        if all_markers:
            all_combined = trimesh.util.concatenate(all_markers)
            all_combined.export(output_path / "all_aneurysm_markers.stl")
            print(f"Created {len(all_markers)} aneurysm markers in {output_dir}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transform aneurysm coordinates between coordinate systems"
    )
    parser.add_argument('nifti_file', help='Reference NIfTI file')
    parser.add_argument('aneurysm_json', help='Input aneurysm JSON file')
    parser.add_argument('output_json', help='Output transformed JSON file')
    parser.add_argument('--mesh-file', help='Optional mesh file for vertex mapping')
    parser.add_argument('--create-markers', action='store_true',
                       help='Create visualization markers')
    parser.add_argument('--marker-dir', default='aneurysm_markers',
                       help='Directory for marker meshes')
    
    args = parser.parse_args()
    
    # Initialize transformer
    transformer = CoordinateTransformer(args.nifti_file)
    
    # Transform coordinates
    transformed_data = transformer.transform_aneurysm_data(
        args.aneurysm_json,
        args.output_json,
        mesh_file=args.mesh_file
    )
    
    # Create visualization markers if requested
    if args.create_markers:
        transformer.create_visualization_markers(
            transformed_data,
            args.marker_dir
        )
    
    # Print summary
    total_aneurysms = sum(
        len(p.get('aneurysms', [])) 
        for p in transformed_data.values()
    )
    print(f"\nTransformed {total_aneurysms} aneurysm locations")


if __name__ == "__main__":
    main() 