#!/usr/bin/env python3
"""
STL to Volume Mesh Converter using Gmsh
Author: Jiwoo Lee

This script converts STL files to volume meshes compatible with Fluent CFD.
The process involves:
1. Load STL surface mesh
2. Generate 3D volume mesh
3. Export to Fluent-compatible format (.msh)
"""

import gmsh
import sys
import os

def convert_stl_to_volume_mesh(stl_file, output_file, element_size=None):
    """
    Convert STL file to volume mesh using Gmsh.
    
    Args:
        stl_file (str): Path to input STL file
        output_file (str): Path to output mesh file (.msh format)
        element_size (float): Maximum element size for meshing
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize Gmsh
        gmsh.initialize()
        
        # Set verbosity level (0: silent, 1: errors, 2: warnings, 3: info, 4: debug)
        gmsh.option.setNumber("General.Verbosity", 3)
        
        # Create a new model
        gmsh.model.add("volume_mesh")
        
        print(f"Loading STL file: {stl_file}")
        
        # Merge the STL file (surface mesh)
        gmsh.merge(stl_file)
        
        # Create surface from triangular mesh
        # Get all triangular elements
        triangle_tags, triangle_nodes = gmsh.model.mesh.getElementsByType(2)  # Type 2 = triangles
        
        if len(triangle_tags) == 0:
            print("No triangles found in STL file")
            return False
        
        print(f"Found {len(triangle_tags)} triangles in STL")
        
        # Create surface loops and surfaces
        # First, we need to get all edges and create a watertight surface
        gmsh.model.mesh.createTopology()
        
        # Get all surfaces created from the STL
        surfaces = gmsh.model.getEntities(2)  # dimension 2 = surfaces
        print(f"Created {len(surfaces)} surfaces")
        
        if len(surfaces) == 0:
            print("No surfaces created from STL")
            return False
        
        # Create surface loop from all surfaces
        surface_tags = [surf[1] for surf in surfaces]
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        
        # Create volume from surface loop
        volume = gmsh.model.geo.addVolume([surface_loop])
        
        # Synchronize CAD representation with mesh representation
        gmsh.model.geo.synchronize()
        
        # Set mesh size if specified
        if element_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size * 0.1)
            print(f"Set element size: {element_size}")
        else:
            # Set a reasonable element size for blood flow simulation
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
            print("Set automatic element size for blood flow")
        
        # Set meshing algorithm
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal algorithm
        gmsh.option.setNumber("Mesh.Optimize", 1)      # Optimize mesh quality
        
        # Generate 3D mesh
        print("Generating volume mesh...")
        gmsh.model.mesh.generate(3)  # 3D mesh
        
        # Get mesh statistics
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)  # Type 4 = tetrahedra
        
        print(f"Generated mesh with {len(node_tags)} nodes and {len(tet_tags)} tetrahedra")
        
        # Set mesh format to be compatible with Fluent
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # Use older format compatible with Fluent
        gmsh.option.setNumber("Mesh.Binary", 0)  # ASCII format
        
        # Write mesh to file
        print(f"Writing mesh to: {output_file}")
        gmsh.write(output_file)
        
        # Also create Fluent case file format
        cas_file = output_file.replace('.msh', '.cas')
        print(f"Writing Fluent case file to: {cas_file}")
        
        # Set up for Fluent export
        gmsh.option.setNumber("Mesh.FluentFormat", 1)
        gmsh.write(cas_file)
        
        # Finalize Gmsh
        gmsh.finalize()
        
        return True
        
    except Exception as e:
        print(f"Error during mesh conversion: {str(e)}")
        gmsh.finalize()
        return False

def main():
    # File paths
    stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
    output_file = "../meshes/78_MRA1_seg_aneurysm_volume.msh"
    
    # Check if STL file exists
    if not os.path.exists(stl_file):
        print(f"STL file not found: {stl_file}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("=== STL to Volume Mesh Conversion ===")
    print(f"Input STL: {stl_file}")
    print(f"Output mesh: {output_file}")
    
    # Convert with reasonable element sizing for blood flow
    success = convert_stl_to_volume_mesh(
        stl_file=stl_file,
        output_file=output_file,
        element_size=0.3  # 0.3mm element size for detailed blood flow
    )
    
    if success:
        print("\n✅ SUCCESS: Volume mesh created successfully!")
        print(f"Output files:")
        print(f"  - MSH format: {output_file}")
        
        cas_file = output_file.replace('.msh', '.cas')
        if os.path.exists(cas_file):
            print(f"  - CAS format: {cas_file}")
        
        # Check file sizes
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"MSH file size: {size_mb:.2f} MB")
            
        if os.path.exists(cas_file):
            size_mb = os.path.getsize(cas_file) / (1024 * 1024)
            print(f"CAS file size: {size_mb:.2f} MB")
            
        return 0
    else:
        print("\n❌ FAILED: Volume mesh conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 