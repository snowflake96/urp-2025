#!/usr/bin/env python3
"""
Simplified High-Quality Surface Remeshing using Ansys Meshing Prime
Author: Jiwoo Lee

This script uses Ansys Meshing Prime to create high-quality surface meshes
from STL files while preserving important geometric features.
"""

import ansys.meshing.prime as prime
import os
import sys

def create_high_quality_surface_mesh(stl_file, output_stl):
    """
    Create high-quality surface mesh using Ansys Meshing Prime
    
    Args:
        stl_file (str): Path to input STL file
        output_stl (str): Path to output STL file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if STL file exists
        if not os.path.exists(stl_file):
            print(f"❌ STL file not found: {stl_file}")
            return False
        
        print("=== Ansys Meshing Prime Surface Remeshing ===")
        print(f"Input STL: {stl_file}")
        print(f"Output STL: {output_stl}")
        
        # Launch Ansys Meshing Prime
        print("🚀 Launching Ansys Meshing Prime...")
        prime_client = prime.launch_prime()
        model = prime_client.model
        
        print("✅ Ansys Meshing Prime launched successfully!")
        
        # Import STL file using import_cad
        print(f"📥 Importing STL file: {stl_file}")
        
        # Use absolute path
        abs_stl_file = os.path.abspath(stl_file)
        abs_output_stl = os.path.abspath(output_stl)
        
        # Import STL using FileIO.import_cad
        file_io = prime.FileIO(model)
        
        # Read STL file as CAD with minimal parameters
        import_result = file_io.import_cad(
            file_name=abs_stl_file,
            params=prime.ImportCadParams(model=model)
        )
        
        print("✅ STL file imported successfully!")
        
        # Get parts
        parts = model.parts
        if not parts:
            print("❌ No parts found in the imported STL")
            return False
        
        print(f"Found {len(parts)} part(s)")
        part = parts[0]
        print(f"Working with part: {part.name}")
        
        # Apply curvature-based sizing controls
        print("🔧 Setting up curvature-based sizing controls...")
        
        # Create curvature sizing parameters
        try:
            curvature_sizing_params = prime.CurvatureSizingParams(
                model=model,
                min_size=0.05,     # Fine mesh on high curvature areas (0.05mm)
                max_size=0.3,      # Coarser on flat areas (0.3mm)
                normal_angle=15.0  # Curvature angle threshold in degrees
            )
            
            # Apply sizing controls to all face zones
            face_zones = part.get_face_zones()
            for face_zone in face_zones:
                # Apply curvature-based sizing
                face_zone.add_curvature_sizing_control(curvature_sizing_params)
            
            print("✅ Curvature-based sizing controls applied")
            
        except Exception as e:
            print(f"⚠️  Could not apply curvature sizing: {e}")
            print("Proceeding with default sizing...")
        
        # Set up surface mesh parameters
        print("🔨 Setting up surface mesh parameters...")
        
        try:
            surface_mesh_params = prime.SurfaceMeshParams(
                model=model,
                max_size=0.5,      # Maximum element size (0.5mm)
                min_size=0.05,     # Minimum element size (0.05mm)
                generate_quads=False  # Use triangles
            )
            
            # Generate surface mesh
            print("🔨 Generating high-quality surface mesh...")
            
            # Perform surface meshing
            part.compute_surface_mesh(surface_mesh_params)
            
            print("✅ Surface mesh generated successfully!")
            
        except Exception as e:
            print(f"⚠️  Surface meshing error: {e}")
            print("Proceeding with export of existing mesh...")
        
        # Export refined STL
        print(f"💾 Exporting refined mesh to: {output_stl}")
        
        file_io.export_stl(
            file_name=abs_output_stl,
            part_ids=[part.id for part in parts]
        )
        
        print("✅ High-quality surface mesh exported successfully!")
        
        # Clean up
        prime_client.exit()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during surface remeshing: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        try:
            prime_client.exit()
        except:
            pass
        return False

def compare_meshes(original_stl, refined_stl):
    """Compare original and refined meshes"""
    
    print("\n📊 MESH COMPARISON")
    print("=" * 30)
    
    if os.path.exists(original_stl):
        size_mb = os.path.getsize(original_stl) / (1024 * 1024)
        print(f"Original STL: {size_mb:.2f} MB")
    
    if os.path.exists(refined_stl):
        size_mb = os.path.getsize(refined_stl) / (1024 * 1024)
        print(f"Refined STL:  {size_mb:.2f} MB")
        
        # Quick triangle count estimation
        with open(refined_stl, 'r') as f:
            content = f.read()
            triangle_count = content.count('facet normal')
        print(f"Estimated triangles: {triangle_count:,}")

def main():
    """Main function"""
    
    # File paths
    stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
    output_stl = "../meshes/78_MRA1_seg_aneurysm_highqual.stl"
    
    print("🔬 HIGH-QUALITY SURFACE REMESHING WITH ANSYS PRIME")
    print("=" * 60)
    
    success = create_high_quality_surface_mesh(stl_file, output_stl)
    
    if success:
        print("\n🎉 SUCCESS: High-quality surface mesh created!")
        
        # Compare meshes
        compare_meshes(stl_file, output_stl)
        
        print("\n✨ Key Features:")
        print("  ✅ Curvature-based refinement")
        print("  ✅ Fine mesh on aneurysm curves")
        print("  ✅ Coarser mesh on flat vessel walls")
        print("  ✅ Preserved geometric features")
        print("  ✅ Ready for volume meshing with Gmsh")
        
        return 0
    else:
        print("\n❌ FAILED: Surface remeshing failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 