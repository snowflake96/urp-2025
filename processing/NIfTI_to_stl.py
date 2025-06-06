"""
NIfTI to STL Converter
Converts NIfTI (.nii, .nii.gz) files to STL format for 3D visualization and processing.
"""

import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
import os
from pathlib import Path
import argparse


def load_nifti(nifti_path):
    """
    Load NIfTI file and return image data and affine matrix.
    
    Args:
        nifti_path (str): Path to the NIfTI file
        
    Returns:
        tuple: (image_data, affine_matrix)
    """
    try:
        nii_img = nib.load(nifti_path)
        image_data = nii_img.get_fdata()
        affine = nii_img.affine
        return image_data, affine
    except Exception as e:
        raise ValueError(f"Error loading NIfTI file {nifti_path}: {str(e)}")


def create_mesh_from_volume(volume_data, threshold=0.5, step_size=1):
    """
    Create 3D mesh from volume data using marching cubes algorithm.
    
    Args:
        volume_data (np.ndarray): 3D volume data
        threshold (float): Threshold for surface extraction
        step_size (int): Step size for marching cubes
        
    Returns:
        tuple: (vertices, faces, normals, values)
    """
    try:
        # Use marching cubes to extract surface
        vertices, faces, normals, values = measure.marching_cubes(
            volume_data, 
            level=threshold,
            step_size=step_size,
            allow_degenerate=True
        )
        return vertices, faces, normals, values
    except Exception as e:
        raise ValueError(f"Error creating mesh: {str(e)}")


def save_stl(vertices, faces, output_path, binary=True):
    """
    Save mesh as STL file.
    
    Args:
        vertices (np.ndarray): Mesh vertices
        faces (np.ndarray): Mesh faces
        output_path (str): Output STL file path
        binary (bool): Save as binary STL (default: True)
    """
    try:
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Clean up the mesh
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        # Save as STL
        mesh.export(output_path, file_type='stl')
        print(f"STL file saved: {output_path}")
        
    except Exception as e:
        raise ValueError(f"Error saving STL file: {str(e)}")


def nifti_to_stl(nifti_path, output_path=None, threshold=0.5, step_size=1, 
                 target_label=None, binary_stl=True):
    """
    Convert NIfTI file to STL format.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_path (str): Path for output STL file (optional)
        threshold (float): Threshold for surface extraction
        step_size (int): Step size for marching cubes
        target_label (int): Specific label to extract (optional)
        binary_stl (bool): Save as binary STL
        
    Returns:
        str: Path to created STL file
    """
    # Set default output path if not provided
    if output_path is None:
        nifti_name = Path(nifti_path).stem
        if nifti_name.endswith('.nii'):
            nifti_name = nifti_name[:-4]
        output_path = f"{nifti_name}.stl"
    
    print(f"Converting {nifti_path} to {output_path}")
    
    # Load NIfTI file
    volume_data, affine = load_nifti(nifti_path)
    
    # Filter by specific label if requested
    if target_label is not None:
        print(f"Extracting label {target_label}")
        volume_data = (volume_data == target_label).astype(float)
        threshold = 0.5
    
    # Check if volume has any data above threshold
    if np.max(volume_data) <= threshold:
        raise ValueError(f"No data above threshold {threshold} found in volume")
    
    print(f"Volume shape: {volume_data.shape}")
    print(f"Volume range: [{np.min(volume_data):.3f}, {np.max(volume_data):.3f}]")
    
    # Create mesh
    vertices, faces, normals, values = create_mesh_from_volume(
        volume_data, threshold, step_size
    )
    
    print(f"Mesh created: {len(vertices)} vertices, {len(faces)} faces")
    
    # Apply affine transformation to vertices if needed
    if affine is not None:
        # Convert to homogeneous coordinates
        vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
        # Apply transformation
        vertices_transformed = vertices_homo.dot(affine.T)[:, :3]
        vertices = vertices_transformed
    
    # Save STL
    save_stl(vertices, faces, output_path, binary_stl)
    
    return output_path


def main():
    """Command line interface for NIfTI to STL conversion."""
    parser = argparse.ArgumentParser(description='Convert NIfTI files to STL format')
    parser.add_argument('input', help='Input NIfTI file path')
    parser.add_argument('-o', '--output', help='Output STL file path')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Threshold for surface extraction (default: 0.5)')
    parser.add_argument('-s', '--step-size', type=int, default=1,
                       help='Step size for marching cubes (default: 1)')
    parser.add_argument('-l', '--label', type=int,
                       help='Extract specific label value')
    parser.add_argument('--ascii', action='store_true',
                       help='Save as ASCII STL instead of binary')
    
    args = parser.parse_args()
    
    try:
        output_path = nifti_to_stl(
            nifti_path=args.input,
            output_path=args.output,
            threshold=args.threshold,
            step_size=args.step_size,
            target_label=args.label,
            binary_stl=not args.ascii
        )
        print(f"Conversion completed successfully: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 