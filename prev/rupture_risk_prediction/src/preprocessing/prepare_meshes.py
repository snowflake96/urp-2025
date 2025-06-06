"""
Mesh preprocessing module for cerebral aneurysm models.
Prepares STL/VTK files for FEA analysis.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import distance
from sklearn.decomposition import PCA
import pymeshlab as ml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AneurysmMeshPreprocessor:
    """Preprocess aneurysm meshes for FEA analysis."""
    
    def __init__(self, target_edge_length: float = 0.5):
        """
        Initialize preprocessor.
        
        Args:
            target_edge_length: Target edge length for remeshing (mm)
        """
        self.target_edge_length = target_edge_length
        self.ms = ml.MeshSet()
        
    def load_mesh(self, file_path: str) -> trimesh.Trimesh:
        """Load mesh from file."""
        logger.info(f"Loading mesh from {file_path}")
        mesh = trimesh.load(file_path)
        
        if isinstance(mesh, trimesh.Scene):
            # If multiple meshes, combine them
            mesh = mesh.dump(concatenate=True)
            
        return mesh
        
    def clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Clean and repair mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Cleaned mesh
        """
        logger.info("Cleaning mesh...")
        
        # Remove duplicate vertices and faces
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Fill holes
        mesh.fill_holes()
        
        # Fix normals
        mesh.fix_normals()
        
        # Ensure watertight
        if not mesh.is_watertight:
            logger.warning("Mesh is not watertight, attempting repair...")
            mesh = self._make_watertight(mesh)
            
        return mesh
        
    def _make_watertight(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attempt to make mesh watertight using pymeshlab."""
        # Save to temporary file
        temp_file = "temp_mesh.stl"
        mesh.export(temp_file)
        
        # Load in pymeshlab
        self.ms.clear()
        self.ms.load_new_mesh(temp_file)
        
        # Apply filters to close holes
        self.ms.meshing_close_holes(maxholesize=30)
        self.ms.meshing_repair_non_manifold_edges()
        self.ms.meshing_repair_non_manifold_vertices()
        
        # Save and reload
        self.ms.save_current_mesh(temp_file)
        mesh = trimesh.load(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        return mesh
        
    def remesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Remesh to achieve uniform element size.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Remeshed mesh
        """
        logger.info(f"Remeshing with target edge length: {self.target_edge_length}mm")
        
        # Save to temporary file
        temp_file = "temp_mesh.stl"
        mesh.export(temp_file)
        
        # Load in pymeshlab
        self.ms.clear()
        self.ms.load_new_mesh(temp_file)
        
        # Compute target metrics
        current_mesh = self.ms.current_mesh()
        bbox_diag = current_mesh.bounding_box().diagonal()
        target_edge_percent = (self.target_edge_length / bbox_diag) * 100
        
        # Apply isotropic remeshing
        self.ms.meshing_isotropic_explicit_remeshing(
            targetlen=ml.AbsoluteValue(self.target_edge_length),
            iterations=3
        )
        
        # Save and reload
        self.ms.save_current_mesh(temp_file)
        mesh = trimesh.load(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        return mesh
        
    def extract_centerline(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Extract vessel centerline (simplified version).
        
        Args:
            mesh: Aneurysm mesh
            
        Returns:
            Centerline points
        """
        # Use PCA to find principal axis
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        
        # Project vertices onto principal axis
        principal_axis = pca.components_[0]
        projections = mesh.vertices.dot(principal_axis)
        
        # Sample points along the axis
        num_points = 50
        t = np.linspace(projections.min(), projections.max(), num_points)
        centerline = np.array([mesh.centroid + ti * principal_axis for ti in t])
        
        return centerline
        
    def identify_neck_region(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Identify aneurysm neck region (simplified).
        
        Args:
            mesh: Aneurysm mesh
            
        Returns:
            Vertex indices of neck region
        """
        # Compute curvature
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=1.0
        )
        
        # Find high curvature regions (potential neck)
        threshold = np.percentile(np.abs(curvature), 90)
        neck_candidates = np.where(np.abs(curvature) > threshold)[0]
        
        # Cluster candidates and find the most compact cluster
        # This is a simplified approach
        if len(neck_candidates) > 0:
            # Use the bottom 10% of vertices as neck (simplified)
            z_coords = mesh.vertices[:, 2]
            threshold = np.percentile(z_coords, 10)
            neck_indices = np.where(z_coords < threshold)[0]
        else:
            neck_indices = np.array([])
            
        return neck_indices
        
    def compute_morphological_features(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute morphological features of the aneurysm.
        
        Args:
            mesh: Aneurysm mesh
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Basic geometric features
        features['volume'] = float(mesh.volume)
        features['surface_area'] = float(mesh.area)
        features['max_diameter'] = float(mesh.bounding_box.primitive.extents.max())
        
        # Compute neck diameter (simplified)
        neck_indices = self.identify_neck_region(mesh)
        if len(neck_indices) > 0:
            neck_vertices = mesh.vertices[neck_indices]
            neck_diameter = distance.pdist(neck_vertices).max()
            features['neck_diameter'] = float(neck_diameter)
        else:
            features['neck_diameter'] = features['max_diameter'] * 0.3  # Default estimate
            
        # Aspect ratio
        features['aspect_ratio'] = features['max_diameter'] / features['neck_diameter']
        
        # Size ratio
        parent_diameter = features['neck_diameter'] * 1.5  # Estimate
        features['size_ratio'] = features['max_diameter'] / parent_diameter
        
        # Sphericity index
        sphere_volume = (4/3) * np.pi * (features['max_diameter']/2)**3
        features['sphericity_index'] = features['volume'] / sphere_volume
        
        # Surface irregularity (based on curvature variation)
        curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=0.5
        )
        features['surface_irregularity'] = float(np.std(curvature))
        
        return features
        
    def save_preprocessed_mesh(self, mesh: trimesh.Trimesh, 
                             output_path: str,
                             features: Dict[str, float],
                             metadata: Dict = None):
        """
        Save preprocessed mesh and associated data.
        
        Args:
            mesh: Preprocessed mesh
            output_path: Output file path
            features: Morphological features
            metadata: Additional metadata
        """
        # Save mesh
        mesh.export(output_path)
        logger.info(f"Saved preprocessed mesh to {output_path}")
        
        # Save features and metadata
        data = {
            'morphological_features': features,
            'preprocessing_info': {
                'target_edge_length': self.target_edge_length,
                'num_vertices': len(mesh.vertices),
                'num_faces': len(mesh.faces),
                'is_watertight': mesh.is_watertight
            }
        }
        
        if metadata:
            data['metadata'] = metadata
            
        # Save as JSON
        json_path = output_path.replace('.stl', '_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def process_mesh(self, input_path: str, output_path: str,
                    metadata_path: Optional[str] = None) -> Dict[str, float]:
        """
        Complete preprocessing pipeline for a single mesh.
        
        Args:
            input_path: Input mesh file path
            output_path: Output mesh file path
            metadata_path: Optional metadata JSON file
            
        Returns:
            Morphological features
        """
        # Load mesh
        mesh = self.load_mesh(input_path)
        
        # Clean mesh
        mesh = self.clean_mesh(mesh)
        
        # Remesh
        mesh = self.remesh(mesh)
        
        # Compute features
        features = self.compute_morphological_features(mesh)
        
        # Load metadata if provided
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        # Save results
        self.save_preprocessed_mesh(mesh, output_path, features, metadata)
        
        return features


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess aneurysm meshes')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing mesh files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed meshes')
    parser.add_argument('--edge_length', type=float, default=0.5,
                       help='Target edge length for remeshing (mm)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AneurysmMeshPreprocessor(target_edge_length=args.edge_length)
    
    # Process all mesh files
    input_path = Path(args.input_dir)
    mesh_files = list(input_path.glob('*.stl')) + list(input_path.glob('*.vtk'))
    
    all_features = []
    
    for mesh_file in mesh_files:
        logger.info(f"Processing {mesh_file.name}")
        
        # Check for metadata
        metadata_file = mesh_file.with_suffix('.json')
        metadata_path = str(metadata_file) if metadata_file.exists() else None
        
        # Output path
        output_path = os.path.join(args.output_dir, mesh_file.name)
        
        try:
            # Process mesh
            features = preprocessor.process_mesh(
                str(mesh_file), 
                output_path,
                metadata_path
            )
            
            features['filename'] = mesh_file.name
            all_features.append(features)
            
        except Exception as e:
            logger.error(f"Error processing {mesh_file.name}: {e}")
            continue
            
    # Save summary of all features
    import pandas as pd
    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(args.output_dir, 'morphological_features.csv'), index=False)
    logger.info(f"Processed {len(all_features)} meshes successfully")


if __name__ == "__main__":
    main() 