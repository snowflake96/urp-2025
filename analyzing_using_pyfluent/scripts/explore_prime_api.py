#!/usr/bin/env python3
"""
Explore Ansys Meshing Prime API
Author: Jiwoo Lee

This script explores the available methods in Ansys Meshing Prime to understand
the correct API syntax for importing and working with STL files.
"""

import ansys.meshing.prime as prime
import os

def explore_prime_api():
    """Explore Prime API methods"""
    
    try:
        print("🔍 EXPLORING ANSYS MESHING PRIME API")
        print("=" * 50)
        
        # Launch Prime
        print("🚀 Launching Ansys Meshing Prime...")
        prime_client = prime.launch_prime()
        model = prime_client.model
        
        print("✅ Prime launched successfully!")
        
        # Explore FileIO methods
        print("\n📂 FileIO methods:")
        file_io = prime.FileIO(model)
        fileio_methods = [method for method in dir(file_io) if not method.startswith('_')]
        for method in sorted(fileio_methods):
            print(f"  - {method}")
        
        # Explore model methods
        print("\n🏗️  Model methods:")
        model_methods = [method for method in dir(model) if not method.startswith('_')]
        for method in sorted(model_methods)[:20]:  # Show first 20
            print(f"  - {method}")
        print("  ... (and more)")
        
        # Check for import methods specifically
        print("\n📥 Import-related methods:")
        all_methods = dir(file_io) + dir(model)
        import_methods = [m for m in all_methods if 'import' in m.lower()]
        for method in sorted(set(import_methods)):
            print(f"  - {method}")
        
        # Check for STL-related methods
        print("\n🗃️  STL-related methods:")
        stl_methods = [m for m in all_methods if 'stl' in m.lower()]
        for method in sorted(set(stl_methods)):
            print(f"  - {method}")
        
        # Try to check what parameters are available
        print("\n⚙️  Available Parameter Classes:")
        prime_attrs = [attr for attr in dir(prime) if 'Param' in attr]
        for attr in sorted(prime_attrs)[:10]:  # Show first 10
            print(f"  - {attr}")
        print("  ... (and more)")
        
        # Look for Import params specifically
        print("\n📋 Import Parameter Classes:")
        import_params = [attr for attr in dir(prime) if 'Import' in attr and 'Param' in attr]
        for attr in sorted(import_params):
            print(f"  - {attr}")
        
        prime_client.exit()
        
        return True
        
    except Exception as e:
        print(f"❌ Error exploring API: {str(e)}")
        return False

if __name__ == "__main__":
    explore_prime_api() 