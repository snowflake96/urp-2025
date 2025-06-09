#!/usr/bin/env python3
"""
Setup Script for PyFluent CFD Analysis System
Author: Jiwoo Lee

This script sets up the environment and validates the installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def setup_virtual_environment():
    """Set up virtual environment."""
    venv_path = Path("./venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    success = run_command("python -m venv venv", "Creating virtual environment")
    return success

def install_dependencies():
    """Install Python dependencies."""
    # Activate virtual environment and install packages
    if os.name == 'nt':  # Windows
        pip_cmd = "./venv/Scripts/pip"
    else:  # Linux/Mac
        pip_cmd = "./venv/bin/pip"
    
    # Upgrade pip first
    success = run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    if not success:
        return False
    
    # Install dependencies
    success = run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")
    return success

def check_ansys_installation():
    """Check if Ansys Fluent is available."""
    print("🔍 Checking Ansys Fluent installation...")
    
    # Common Ansys installation paths
    possible_paths = [
        "/ansys_inc/v251/fluent/bin/fluent",
        "/opt/ansys_inc/v251/fluent/bin/fluent", 
        "C:/Program Files/ANSYS Inc/v251/fluent/ntbin/win64/fluent.exe"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found Ansys Fluent: {path}")
            return True
    
    print("⚠️  Ansys Fluent not found in standard locations")
    print("   Please ensure Ansys 2025 R1 is installed and licensed")
    return False

def test_pyfluent_import():
    """Test PyFluent import."""
    try:
        if os.name == 'nt':  # Windows
            python_cmd = "./venv/Scripts/python"
        else:  # Linux/Mac
            python_cmd = "./venv/bin/python"
        
        test_cmd = f'{python_cmd} -c "import ansys.fluent.core as pyfluent; print(f\'PyFluent version: {{pyfluent.__version__}}\')"'
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
        else:
            print(f"❌ PyFluent import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ PyFluent test failed: {e}")
        return False

def validate_data_directory():
    """Validate the data directory."""
    data_dir = Path("/home/jiwoo/urp/data/uan/clean_flat_vessels")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Count STL and JSON files
    stl_files = list(data_dir.glob("*_clean_flat.stl"))
    json_files = list(data_dir.glob("*_boundary_conditions.json"))
    
    print(f"✅ Data directory found: {data_dir}")
    print(f"   STL files: {len(stl_files)}")
    print(f"   BC files: {len(json_files)}")
    
    if len(stl_files) > 0 and len(json_files) > 0:
        return True
    else:
        print("❌ No data files found")
        return False

def create_results_directory():
    """Create results directory."""
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Results directory ready: {results_dir.absolute()}")
    return True

def main():
    """Main setup function."""
    print("🚀 PyFluent CFD Analysis System Setup")
    print("=" * 50)
    print("Author: Jiwoo Lee")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("PyFluent Import", test_pyfluent_import),
        ("Ansys Installation", check_ansys_installation),
        ("Data Directory", validate_data_directory),
        ("Results Directory", create_results_directory),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n📋 {name}")
        results[name] = check_func()
    
    # Summary
    print(f"\n📊 SETUP SUMMARY")
    print("=" * 30)
    
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"Next steps:")
        print(f"  1. Activate environment: source venv/bin/activate")
        print(f"  2. Test single case: python test_single_case.py")
        print(f"  3. Run batch analysis: python pyfluent_batch_analyzer.py --max-cases 5")
        return True
    else:
        print(f"\n❌ SETUP INCOMPLETE")
        print(f"Please fix the failed checks before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 