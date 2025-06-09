#!/usr/bin/env python3
"""
Minimal Ansys Meshing Prime Export Test
Author: Jiwoo Lee

This script just imports STL and exports it to understand the correct API.
"""

import ansys.meshing.prime as prime
import os

def minimal_import_export():
    """Minimal import and export test"""
    
    try:
        print("🚀 Launching Ansys Meshing Prime...")
        prime_client = prime.launch_prime()
        model = prime_client.model
        
        print("✅ Prime launched!")
        
        # Import STL
        stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
        abs_stl_file = os.path.abspath(stl_file)
        
        file_io = prime.FileIO(model)
        
        print(f"📥 Importing: {stl_file}")
        file_io.import_cad(
            file_name=abs_stl_file,
            params=prime.ImportCadParams(model=model)
        )
        
        print("✅ Import successful!")
        
        # Get parts
        parts = model.parts
        print(f"Found {len(parts)} part(s)")
        
        # Try to explore export_stl parameters
        print("\n🔍 Exploring export_stl signature...")
        import inspect
        sig = inspect.signature(file_io.export_stl)
        print(f"export_stl signature: {sig}")
        
        # Try simple export
        output_file = "../meshes/78_MRA1_seg_aneurysm_minimal_export.stl"
        abs_output_file = os.path.abspath(output_file)
        
        print(f"💾 Exporting to: {output_file}")
        
        # Try export with minimal parameters
        try:
            file_io.export_stl(file_name=abs_output_file)
            print("✅ Export successful with file_name only!")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            
            # Try with different parameter combinations
            try:
                file_io.export_stl(abs_output_file)
                print("✅ Export successful with positional argument!")
            except Exception as e2:
                print(f"❌ Both export attempts failed: {e2}")
        
        # Check if file was created
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Output file created: {size_mb:.2f} MB")
            
            # Quick triangle count
            with open(output_file, 'r') as f:
                content = f.read()
                triangle_count = content.count('facet normal')
            print(f"Triangle count: {triangle_count:,}")
        else:
            print("❌ No output file created")
        
        prime_client.exit()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    minimal_import_export() 