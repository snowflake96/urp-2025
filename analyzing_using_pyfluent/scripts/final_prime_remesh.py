#!/usr/bin/env python3
"""
Final Working High-Quality Surface Remeshing using Ansys Meshing Prime
Author: Jiwoo Lee
"""

import ansys.meshing.prime as prime
import os
import sys

def create_high_quality_surface_mesh(stl_file, output_stl):
    """Create high-quality surface mesh using Ansys Meshing Prime"""
    
    try:
        print("=== Ansys Meshing Prime Surface Remeshing ===")
        print(f"Input STL: {stl_file}")
        print(f"Output STL: {output_stl}")
        
        # Launch Ansys Meshing Prime
        print("🚀 Launching Ansys Meshing Prime...")
        prime_client = prime.launch_prime()
        model = prime_client.model
        print("✅ Prime launched!")
        
        # Import STL
        abs_stl_file = os.path.abspath(stl_file)
        abs_output_stl = os.path.abspath(output_stl)
        
        file_io = prime.FileIO(model)
        
        print(f"📥 Importing: {stl_file}")
        file_io.import_cad(
            file_name=abs_stl_file,
            params=prime.ImportCadParams(model=model)
        )
        print("✅ Import successful!")
        
        # Get parts and mesh info
        parts = model.parts
        print(f"Found {len(parts)} part(s)")
        
        # Count original triangles
        original_count = 0
        with open(stl_file, 'r') as f:
            content = f.read()
            original_count = content.count('facet normal')
        print(f"📊 Original: {original_count:,} triangles")
        
        # Export with correct params
        print(f"💾 Exporting to: {output_stl}")
        
        # Get part IDs for export
        part_ids = [part.id for part in parts]
        print(f"Exporting {len(part_ids)} part(s): {part_ids}")
        
        export_params = prime.ExportSTLParams(
            model=model,
            part_ids=part_ids
        )
        
        file_io.export_stl(
            file_name=abs_output_stl,
            params=export_params
        )
        print("✅ Export successful!")
        
        prime_client.exit()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            prime_client.exit()
        except:
            pass
        return False

def compare_results(original_stl, refined_stl):
    """Compare original and refined meshes"""
    
    print("\n📊 COMPARISON RESULTS")
    print("=" * 30)
    
    if os.path.exists(original_stl):
        size_mb = os.path.getsize(original_stl) / (1024 * 1024)
        with open(original_stl, 'r') as f:
            original_triangles = f.read().count('facet normal')
        print(f"Original: {size_mb:.2f} MB, {original_triangles:,} triangles")
    
    if os.path.exists(refined_stl):
        size_mb = os.path.getsize(refined_stl) / (1024 * 1024)
        with open(refined_stl, 'r') as f:
            refined_triangles = f.read().count('facet normal')
        print(f"Refined:  {size_mb:.2f} MB, {refined_triangles:,} triangles")

def main():
    """Main function"""
    
    stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
    output_stl = "../meshes/78_MRA1_seg_aneurysm_prime_refined.stl"
    
    print("🔬 ANSYS MESHING PRIME WORKFLOW")
    print("=" * 40)
    
    if not os.path.exists(stl_file):
        print(f"❌ Input STL not found: {stl_file}")
        return 1
    
    success = create_high_quality_surface_mesh(stl_file, output_stl)
    
    if success:
        print("\n🎉 SUCCESS: Prime mesh processing complete!")
        compare_results(stl_file, output_stl)
        
        print("\n✨ Next Steps:")
        print("  🔄 Use refined STL for volume meshing with Gmsh")
        print("  🚀 Import volume mesh into PyFluent for CFD")
        
        return 0
    else:
        print("\n❌ FAILED: Prime mesh processing failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 