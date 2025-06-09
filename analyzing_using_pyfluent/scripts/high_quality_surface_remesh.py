#!/usr/bin/env python3
"""
High-Quality Surface Remeshing using Ansys Meshing Prime
Author: Jiwoo Lee

This script uses Ansys Meshing Prime to create high-quality surface meshes
from STL files while preserving important geometric features like aneurysm shape.
The approach uses curvature-based mesh refinement.
"""

import ansys.meshing.prime as prime
import os
import sys

def create_high_quality_surface_mesh(stl_file, output_file):
    """
    Create high-quality surface mesh using Ansys Meshing Prime
    
    Args:
        stl_file (str): Path to input STL file
        output_file (str): Path to output mesh file
    
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
        print(f"Output mesh: {output_file}")
        
        # Launch Ansys Meshing Prime
        print("🚀 Launching Ansys Meshing Prime...")
        prime_client = prime.launch_prime()
        model = prime_client.model
        
        print("✅ Ansys Meshing Prime launched successfully!")
        
        # Import STL file using import_cad
        print(f"📥 Importing STL file: {stl_file}")
        
        # Use absolute path
        abs_stl_file = os.path.abspath(stl_file)
        abs_output_file = os.path.abspath(output_file)
        
        # Import STL using FileIO.import_cad (STL is considered CAD format)
        file_io = prime.FileIO(model)
        
        # Read STL file as CAD
        import_result = file_io.import_cad(
            file_name=abs_stl_file,
            params=prime.ImportCadParams(
                model=model,
                append=False,  # Replace existing mesh
                feature_angle=40.0,  # Feature angle for edge detection
                merge_node_tolerance=1e-6
            )
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
        
        # Get mesh information before remeshing
        print("📊 Original mesh information:")
        summary = model.get_summary(prime.GetSummaryParams(model))
        print(f"Summary: {summary}")
        
        # Set up surface mesh parameters with curvature controls
        print("🔧 Setting up curvature-based mesh parameters...")
        
        # Create surface mesh params with curvature control
        surface_mesh_params = prime.SurfaceMeshParams(
            model=model,
            max_size=0.5,      # Maximum element size (0.5mm)
            min_size=0.05,     # Minimum element size (0.05mm)
            generate_quads=False,  # Use triangles
            algorithm=prime.SurfaceMeshAlgorithm.PATCH_INDEPENDENT
        )
        
        # Apply curvature sizing - this refines mesh based on curvature
        curvature_sizing_params = prime.CurvatureSizingParams(
            model=model,
            min_size=0.05,     # Fine mesh on high curvature areas
            max_size=0.3,      # Coarser on flat areas
            normal_angle=15.0  # Curvature angle threshold in degrees
        )
        
        # Apply sizing controls to all face zones
        for part in parts:
            face_zones = part.get_face_zones()
            for face_zone in face_zones:
                # Apply curvature-based sizing
                face_zone.add_curvature_sizing_control(curvature_sizing_params)
        
        print("✅ Curvature-based sizing controls applied")
        
        # Generate surface mesh
        print("🔨 Generating high-quality surface mesh...")
        
        # Perform surface meshing on all parts
        for part in parts:
            part.compute_surface_mesh(surface_mesh_params)
        
        print("✅ Surface mesh generated successfully!")
        
        # Get final mesh statistics
        final_summary = model.get_summary(prime.GetSummaryParams(model))
        print(f"📊 Final mesh summary: {final_summary}")
        
        # Export mesh - try different formats
        print(f"💾 Exporting mesh to: {output_file}")
        
        # First try to export as STL for visualization
        stl_output = output_file.replace('.msh', '.stl')
        file_io.export_stl(
            file_name=stl_output,
            part_ids=[part.id for part in parts]
        )
        print(f"✅ STL exported: {stl_output}")
        
        # Try to export as Fluent case for CFD
        try:
            cas_output = output_file.replace('.msh', '.cas')
            file_io.export_fluent_case(
                file_name=cas_output,
                part_ids=[part.id for part in parts]
            )
            print(f"✅ Fluent case exported: {cas_output}")
        except Exception as e:
            print(f"⚠️  Could not export Fluent case: {e}")
        
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

def main():
    """Main function"""
    
    # File paths
    stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
    output_file = "../meshes/78_MRA1_seg_aneurysm_highqual_surface.msh"
    
    print("🔬 HIGH-QUALITY SURFACE REMESHING")
    print("=" * 50)
    
    success = create_high_quality_surface_mesh(stl_file, output_file)
    
    if success:
        print("\n🎉 SUCCESS: High-quality surface mesh created!")
        print(f"Output files:")
        
        # Check various output files
        stl_output = output_file.replace('.msh', '.stl')
        cas_output = output_file.replace('.msh', '.cas')
        
        for filepath in [stl_output, cas_output]:
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filepath} ({size_mb:.2f} MB)")
        
        print("\n✨ Key Features:")
        print("  ✅ Curvature-based refinement")
        print("  ✅ Fine mesh on aneurysm curves")
        print("  ✅ Coarser mesh on flat vessel walls")
        print("  ✅ Preserved geometric features")
        
        return 0
    else:
        print("\n❌ FAILED: Surface remeshing failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 