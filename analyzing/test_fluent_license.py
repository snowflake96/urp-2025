#!/usr/bin/env python3
"""
Test Fluent Installation and License
"""

import os
import sys
import subprocess

print("🧪 Testing Ansys Fluent Installation and License")
print("=" * 60)

# Set environment variable
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
print(f"✅ Set AWP_ROOT251: {os.environ['AWP_ROOT251']}")

# Check if Fluent executable exists
fluent_exe = '/opt/cvbml/softwares/ansys_inc/v251/fluent/bin/fluent'
if os.path.exists(fluent_exe):
    print(f"✅ Fluent executable found: {fluent_exe}")
else:
    print(f"❌ Fluent executable not found: {fluent_exe}")
    sys.exit(1)

# Test PyFluent import
try:
    import ansys.fluent.core as pyfluent
    print(f"✅ PyFluent imported successfully (v{pyfluent.__version__})")
except ImportError as e:
    print(f"❌ PyFluent import failed: {e}")
    sys.exit(1)

# Check current license environment variables
print("\n🔍 Checking License Environment Variables:")
license_vars = ['ANSYSLMD_LICENSE_FILE', 'LM_LICENSE_FILE', 'ANSYS_LICENSE_FILE']
for var in license_vars:
    value = os.environ.get(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"⚠️  {var}: Not set")

# Test license by running fluent -help
print("\n🧪 Testing Fluent License (running 'fluent -help'):")
try:
    result = subprocess.run(
        [fluent_exe, '-help'], 
        capture_output=True, 
        text=True, 
        timeout=30,
        env=os.environ.copy()
    )
    
    if result.returncode == 0:
        print("✅ Fluent help command executed successfully")
        print("✅ License appears to be working")
    else:
        print(f"⚠️  Fluent help command failed with return code: {result.returncode}")
        if 'license' in result.stderr.lower():
            print("❌ License error detected:")
            print(result.stderr)
        else:
            print("ℹ️  Error output:")
            print(result.stderr)
            
except subprocess.TimeoutExpired:
    print("⚠️  Fluent command timed out")
except Exception as e:
    print(f"❌ Error running Fluent: {e}")

# Test PyFluent connection (this might fail due to license)
print("\n🚀 Testing PyFluent Launch (expect license check):")
try:
    print("   Attempting to launch Fluent session...")
    
    # Try to launch with minimal options
    session = pyfluent.launch_fluent(
        version="3d",
        precision="double", 
        processor_count=1,
        mode="solver",
        show_gui=False,
        cleanup_on_exit=True,
        start_timeout=60  # Give it more time
    )
    
    print("✅ PyFluent session launched successfully!")
    print("✅ License is working properly!")
    
    # Test basic functionality
    version_info = session.get_fluent_version()
    print(f"   Fluent version: {version_info}")
    
    # Close session
    session.exit()
    print("✅ Session closed successfully")
    
except Exception as e:
    error_msg = str(e).lower()
    if 'license' in error_msg:
        print("❌ License Error:")
        print(f"   {e}")
        print("\n💡 License Setup Required:")
        print("   You need to configure Ansys licensing. Options:")
        print("   1. Set ANSYSLMD_LICENSE_FILE to license server:")
        print("      export ANSYSLMD_LICENSE_FILE=port@server")
        print("   2. Set to license file path:")
        print("      export ANSYSLMD_LICENSE_FILE=/path/to/license.lic")
        print("   3. Contact your Ansys administrator for license server details")
        
    elif 'awp_root' in error_msg:
        print("❌ Installation Error:")
        print(f"   {e}")
        
    else:
        print(f"❌ Unknown Error:")
        print(f"   {e}")

print("\n" + "=" * 60)
print("🎯 Installation Summary:")
print(f"   Ansys Installation: ✅ Found at {os.environ['AWP_ROOT251']}")
print(f"   Fluent Executable: ✅ Available")
print(f"   PyFluent Library: ✅ Installed")
print("   License Status: Check output above")

print("\n💡 Next Steps:")
print("   1. If license error: Configure ANSYSLMD_LICENSE_FILE")
print("   2. If working: Ready for CFD analysis!")
print("   3. Run: python pyfluent_real_cfd.py --help") 