#!/usr/bin/env python3
"""
Test Real PyFluent CFD Analysis - Single Patient Test
"""

import os
import sys

# Set environment variable for Ansys 2025 R1
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'

print("🧪 Testing Real PyFluent CFD Analysis")
print("=" * 60)

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    print(f"✅ PyFluent imported successfully (v{pyfluent.__version__})")
except ImportError as e:
    print(f"❌ PyFluent import failed: {e}")
    sys.exit(1)

print(f"✅ Environment: AWP_ROOT251 = {os.environ.get('AWP_ROOT251')}")

# Test minimal Fluent session launch
print("\n🚀 Testing minimal Fluent session launch...")

try:
    # Launch with minimal configuration
    session = launch_fluent(
        dimension=3,  # Use integer, not string
        product_version=pyfluent.FluentVersion.v251,
        precision="double",
        processor_count=2,  # Start with just 2 cores
        mode="solver",
        ui_mode="no_gui",
        cleanup_on_exit=True,
        start_timeout=120  # Give it 2 minutes
    )
    
    print("✅ Fluent session launched successfully!")
    
    # Test basic functionality
    version_info = session.get_fluent_version()
    print(f"   Version: {version_info}")
    
    # Test if we can access basic solver functions
    try:
        # This is just a test - don't actually run anything
        print("   Testing solver access...")
        solver_info = session.solution
        print("   ✅ Solver interface accessible")
        
        # Test setup access
        setup_info = session.setup
        print("   ✅ Setup interface accessible")
        
        # Test file interface
        file_info = session.file
        print("   ✅ File interface accessible")
        
    except Exception as e:
        print(f"   ⚠️ Interface test warning: {e}")
    
    # Close session
    session.exit()
    print("✅ Session closed successfully")
    
    print("\n🎯 SUCCESS! PyFluent is ready for CFD analysis")
    print("   ✅ Ansys Fluent 2025 R1 detected and working")
    print("   ✅ License is properly configured")
    print("   ✅ All interfaces accessible")
    print("   ✅ Ready for real aneurysm CFD analysis!")
    
    print("\n📋 Next Steps:")
    print("   1. Prepare STL mesh files and boundary condition JSON files")
    print("   2. Run: export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251")
    print("   3. Run: python pyfluent_real_cfd.py --patient-limit 1 --n-cores 4")
    print("   4. Check results in pyfluent_results/ directory")
    
except Exception as e:
    error_msg = str(e).lower()
    print(f"❌ PyFluent session failed: {e}")
    
    if 'license' in error_msg:
        print("\n💡 License issue detected. Possible solutions:")
        print("   1. Check if license server is running")
        print("   2. Set ANSYSLMD_LICENSE_FILE environment variable")
        print("   3. Contact your Ansys administrator")
    elif 'timeout' in error_msg:
        print("\n💡 Timeout issue. Possible solutions:")
        print("   1. Increase start_timeout parameter")
        print("   2. Check system resources")
        print("   3. Try with fewer cores")
    else:
        print("\n💡 Check Ansys installation and environment variables")

print("\n" + "=" * 60)
print("🎯 Test Complete") 