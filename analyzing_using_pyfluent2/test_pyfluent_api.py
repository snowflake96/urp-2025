#!/usr/bin/env python3
"""
Test PyFluent API Structure
Author: Jiwoo Lee

Simple script to explore the correct PyFluent API structure.
"""

import sys
from pathlib import Path

try:
    import ansys.fluent.core as pyfluent
    print(f"✅ PyFluent {pyfluent.__version__} loaded")
except ImportError as e:
    print(f"❌ PyFluent not available: {e}")
    sys.exit(1)

def explore_session_api():
    """Explore the PyFluent session API structure."""
    
    print("🔍 Exploring PyFluent API structure...")
    
    try:
        # Launch session
        print("🚀 Launching PyFluent session...")
        session = pyfluent.launch_fluent(
            precision="double",
            processor_count=4,
            mode="solver",
            dimension=3,
            ui_mode="no_gui",
            start_timeout=180,
            cleanup_on_exit=True
        )
        
        print("✅ Session launched successfully")
        
        # Explore session attributes
        print("\n📋 Session attributes:")
        session_attrs = [attr for attr in dir(session) if not attr.startswith('_')]
        for attr in sorted(session_attrs):
            print(f"  - session.{attr}")
        
        # Check if we have solver-related attributes
        print("\n🔍 Checking solver access patterns:")
        
        # Try different API patterns
        patterns_to_test = [
            "session.solver",
            "session.setup", 
            "session.solution",
            "session.models",
            "session.boundary_conditions",
            "session.materials"
        ]
        
        for pattern in patterns_to_test:
            try:
                obj = eval(pattern)
                print(f"✅ {pattern} exists: {type(obj)}")
                
                # If it exists, show its attributes
                if hasattr(obj, '__dir__'):
                    attrs = [attr for attr in dir(obj) if not attr.startswith('_')][:5]  # First 5 attrs
                    print(f"   First attributes: {attrs}")
            except AttributeError:
                print(f"❌ {pattern} not found")
            except Exception as e:
                print(f"⚠️  {pattern} error: {e}")
        
        # Test a simple mesh read
        print("\n📁 Testing mesh import...")
        try:
            stl_file = "/home/jiwoo/urp/data/uan/clean_flat_vessels/06_MRA1_seg_clean_flat.stl"
            if Path(stl_file).exists():
                # Try different mesh reading patterns
                read_patterns = [
                    "session.file.read_mesh",
                    "session.mesh.read",
                    "session.read_mesh"
                ]
                
                for pattern in read_patterns:
                    try:
                        read_func = eval(pattern)
                        print(f"✅ Found mesh reader: {pattern}")
                        break
                    except AttributeError:
                        print(f"❌ {pattern} not found")
            else:
                print("⚠️  Test STL file not found")
        except Exception as e:
            print(f"❌ Mesh test error: {e}")
        
        # Close session
        session.exit()
        print("\n✅ Session closed successfully")
        
    except Exception as e:
        print(f"❌ API exploration failed: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🧪 PyFluent API Structure Test")
    print("=" * 40)
    
    success = explore_session_api()
    
    if success:
        print("\n🎉 API exploration completed!")
        print("Use the findings to fix the main analyzer script.")
    else:
        print("\n❌ API exploration failed!")
        print("Check PyFluent documentation for correct API usage.")

if __name__ == "__main__":
    main() 