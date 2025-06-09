#!/usr/bin/env python3
"""
Test PyFluent Import and Basic Functionality
"""

print("🧪 Testing PyFluent Import...")

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    print(f"✅ PyFluent imported successfully!")
    print(f"   Version: {pyfluent.__version__}")
    
    # Test if we can access basic functionality
    print(f"   Available modules: {dir(pyfluent)[:5]}...")
    
    # Check if Fluent is available on the system
    print("\n🔍 Checking Fluent availability...")
    
    try:
        # This will fail if Fluent is not installed, but that's expected
        print("   Attempting to launch Fluent (this may fail if Fluent not installed)...")
        session = launch_fluent(
            version="3d",
            precision="double", 
            processor_count=2,
            mode="solver",
            show_gui=False,
            cleanup_on_exit=True
        )
        print("   ✅ Fluent launched successfully!")
        session.exit()
        print("   ✅ Fluent session closed")
        
    except Exception as e:
        print(f"   ⚠️  Fluent launch failed (expected if Fluent not installed): {e}")
        print("   This is normal if Ansys Fluent is not installed on the system")
    
except ImportError as e:
    print(f"❌ PyFluent import failed: {e}")

print("\n🎯 PyFluent test complete!") 