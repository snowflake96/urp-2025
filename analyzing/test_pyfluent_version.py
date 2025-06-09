#!/usr/bin/env python3
"""
Test PyFluent with Explicit Version Specification
"""

import os
import sys

print("🧪 Testing PyFluent with Ansys 2025 R1 (v251)")
print("=" * 60)

# Set environment variable for v251
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
print(f"✅ Set AWP_ROOT251: {os.environ['AWP_ROOT251']}")

# Import PyFluent
try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import FluentVersion
    print(f"✅ PyFluent imported (v{pyfluent.__version__})")
except ImportError as e:
    print(f"❌ PyFluent import failed: {e}")
    sys.exit(1)

# Check available Fluent versions
print("\n🔍 Available Fluent Versions:")
try:
    for version in FluentVersion:
        print(f"   {version.name}: {version.value}")
except Exception as e:
    print(f"   Could not list versions: {e}")

# Test different ways to specify version 25.1
print("\n🚀 Testing PyFluent Launch with Version 25.1:")

version_specs = [
    FluentVersion.v251,
    "25.1.0", 
    "25.1",
    25.1,
    251
]

for i, version_spec in enumerate(version_specs):
    print(f"\n   Test {i+1}: Using version = {version_spec}")
    try:
        session = pyfluent.launch_fluent(
            version="3d",
            product_version=version_spec,  # Explicitly specify version
            precision="double", 
            processor_count=1,
            mode="solver",
            show_gui=False,
            cleanup_on_exit=True,
            start_timeout=60
        )
        
        print(f"   ✅ SUCCESS! Session launched with version {version_spec}")
        
        # Test basic functionality
        try:
            version_info = session.get_fluent_version()
            print(f"   ✅ Fluent version info: {version_info}")
        except:
            print(f"   ℹ️  Could not get version info")
        
        # Close session
        session.exit()
        print(f"   ✅ Session closed successfully")
        
        print(f"\n🎯 WORKING CONFIGURATION:")
        print(f"   export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251")
        print(f"   product_version={version_spec}")
        break
        
    except Exception as e:
        error_msg = str(e)
        if 'license' in error_msg.lower():
            print(f"   ❌ License Error: {e}")
        elif 'awp_root' in error_msg.lower():
            print(f"   ❌ Version Error: {e}")
        else:
            print(f"   ❌ Other Error: {e}")

else:
    print(f"\n❌ None of the version specifications worked")
    print(f"💡 Try setting additional environment variables:")
    print(f"   export FLUENT_ROOT=/opt/cvbml/softwares/ansys_inc/v251/fluent")

print("\n" + "=" * 60)
print("🎯 Summary:")
print("   If a version worked above, use that configuration")
print("   If license errors, you need to configure Ansys licensing")
print("   If no version worked, check Ansys installation paths") 