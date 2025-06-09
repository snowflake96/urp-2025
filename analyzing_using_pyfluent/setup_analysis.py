#!/usr/bin/env python3
"""
Setup and Validation Script for PyFluent Aneurysm Analysis

This script checks all prerequisites and prepares the environment for
78_MRA1_seg aneurysm CFD analysis using PyFluent.
"""

import os
import sys
import json
from pathlib import Path

def check_environment():
    """Check system environment and prerequisites"""
    print("🔍 Checking Environment and Prerequisites")
    print("=" * 60)
    
    checks = {
        'ansys_installation': False,
        'pyfluent_available': False,
        'environment_vars': False,
        'mesh_file': False,
        'boundary_conditions': False,
        'output_directories': False
    }
    
    # Check Ansys installation
    ansys_path = '/opt/cvbml/softwares/ansys_inc/v251'
    if os.path.exists(ansys_path):
        fluent_exe = os.path.join(ansys_path, 'fluent', 'bin', 'fluent')
        if os.path.exists(fluent_exe):
            checks['ansys_installation'] = True
            print(f"✅ Ansys Fluent 2025 R1 found at {ansys_path}")
        else:
            print(f"❌ Fluent executable not found in {ansys_path}")
    else:
        print(f"❌ Ansys installation not found at {ansys_path}")
    
    # Check PyFluent
    try:
        import ansys.fluent.core as pyfluent
        checks['pyfluent_available'] = True
        print(f"✅ PyFluent available (v{pyfluent.__version__})")
    except ImportError:
        print(f"❌ PyFluent not available - install with: pip install ansys-fluent-core")
    
    # Check environment variables
    env_var = 'AWP_ROOT251'
    if env_var in os.environ:
        checks['environment_vars'] = True
        print(f"✅ Environment variable {env_var} = {os.environ[env_var]}")
    else:
        print(f"⚠️  Environment variable {env_var} not set")
        print(f"   Run: export AWP_ROOT251={ansys_path}")
    
    # Check project structure
    project_dir = Path(__file__).parent
    
    # Check mesh file
    mesh_file = project_dir / "meshes" / "78_MRA1_seg_aneurysm.stl"
    if mesh_file.exists():
        checks['mesh_file'] = True
        size_mb = mesh_file.stat().st_size / (1024 * 1024)
        print(f"✅ Mesh file found: {mesh_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Mesh file not found: {mesh_file}")
        print(f"💡 Please place the STL mesh file at: {mesh_file}")
    
    # Check boundary conditions
    bc_file = project_dir / "boundary_conditions" / "78_MRA1_seg_pyfluent_bc.json"
    if bc_file.exists():
        checks['boundary_conditions'] = True
        print(f"✅ Boundary conditions found: {bc_file.name}")
        
        # Validate JSON
        try:
            with open(bc_file, 'r') as f:
                bc_data = json.load(f)
            print(f"   Patient: {bc_data['metadata']['patient_id']}")
            print(f"   Reynolds: {bc_data['inlet_conditions']['reynolds_number']}")
            print(f"   Velocity: {bc_data['inlet_conditions']['mean_velocity_ms']} m/s")
        except Exception as e:
            print(f"⚠️  Boundary conditions file invalid: {e}")
    else:
        print(f"❌ Boundary conditions not found: {bc_file}")
    
    # Check/create output directories
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_dir = results_dir / "78_MRA1_seg_pyfluent"
    output_dir.mkdir(exist_ok=True)
    
    checks['output_directories'] = True
    print(f"✅ Output directories ready: {results_dir}")
    
    return checks

def test_pyfluent_connection():
    """Test PyFluent connection to Ansys Fluent"""
    print(f"\n🧪 Testing PyFluent Connection")
    print("=" * 60)
    
    # Set environment
    os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
    
    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core import launch_fluent
        
        print(f"   Attempting to launch Fluent session...")
        print(f"   (This may take 1-2 minutes)")
        
        # Test minimal session
        session = launch_fluent(
            dimension=3,
            product_version=pyfluent.FluentVersion.v251,
            precision="double",
            processor_count=2,  # Use minimal cores for test
            mode="solver",
            ui_mode="no_gui",
            cleanup_on_exit=True,
            start_timeout=120
        )
        
        # Test basic functionality
        version_info = session.get_fluent_version()
        print(f"✅ PyFluent connection successful!")
        print(f"   Version: {version_info}")
        print(f"   Test cores: 2")
        
        # Close session
        session.exit()
        print(f"✅ Session closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ PyFluent connection failed: {e}")
        
        error_msg = str(e).lower()
        if 'license' in error_msg:
            print(f"💡 License issue - contact your Ansys administrator")
        elif 'timeout' in error_msg:
            print(f"💡 Timeout - try increasing start_timeout")
        else:
            print(f"💡 Check Ansys installation and environment")
        
        return False

def show_analysis_summary():
    """Show analysis configuration summary"""
    print(f"\n📋 Analysis Configuration Summary")
    print("=" * 60)
    
    project_dir = Path(__file__).parent
    bc_file = project_dir / "boundary_conditions" / "78_MRA1_seg_pyfluent_bc.json"
    
    if bc_file.exists():
        try:
            with open(bc_file, 'r') as f:
                bc_data = json.load(f)
            
            print(f"🏥 Patient Information:")
            print(f"   ID: {bc_data['metadata']['patient_id']}")
            print(f"   Vessel: {bc_data['metadata']['vessel_type']}")
            print(f"   Aneurysm: {bc_data['metadata']['aneurysm_present']}")
            
            print(f"\n🌊 Flow Conditions:")
            inlet = bc_data['inlet_conditions']
            print(f"   Reynolds: {inlet['reynolds_number']}")
            print(f"   Mean velocity: {inlet['mean_velocity_ms']} m/s")
            print(f"   Peak velocity: {inlet['peak_velocity_ms']} m/s")
            print(f"   Flow direction: {inlet['velocity_direction']}")
            
            print(f"\n⚙️ Simulation Settings:")
            solver = bc_data['solver_settings']
            print(f"   Time step: {solver['time_step_size_s']} s")
            print(f"   Cardiac cycles: {solver['number_of_cycles']}")
            print(f"   Total steps: {int(solver['cardiac_cycle_duration_s'] * solver['number_of_cycles'] / solver['time_step_size_s']):,}")
            print(f"   Simulation time: {solver['cardiac_cycle_duration_s'] * solver['number_of_cycles']:.1f} s")
            
            print(f"\n🎯 Aneurysm Parameters:")
            aneurysm = bc_data['aneurysm_specific']
            print(f"   Size ratio: {aneurysm['rupture_risk_factors']['size_ratio']}")
            print(f"   Aspect ratio: {aneurysm['rupture_risk_factors']['aspect_ratio']}")
            print(f"   Critical WSS: {aneurysm['critical_wss_threshold_pa']} Pa")
            print(f"   High WSS: {aneurysm['high_wss_threshold_pa']} Pa")
            
        except Exception as e:
            print(f"❌ Error reading boundary conditions: {e}")
    else:
        print(f"❌ Boundary conditions file not found")

def show_next_steps(checks):
    """Show next steps based on check results"""
    print(f"\n🎯 Next Steps")
    print("=" * 60)
    
    all_good = all(checks.values())
    
    if all_good:
        print(f"🎉 All prerequisites met! Ready to run analysis.")
        print(f"\n📋 To run the analysis:")
        print(f"   1. cd scripts/")
        print(f"   2. export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251")
        print(f"   3. python analyze_78_MRA1_seg.py")
        
    else:
        print(f"⚠️  Some issues need to be resolved:")
        
        if not checks['ansys_installation']:
            print(f"   ❌ Install Ansys Fluent 2025 R1")
        
        if not checks['pyfluent_available']:
            print(f"   ❌ Install PyFluent: pip install ansys-fluent-core")
        
        if not checks['environment_vars']:
            print(f"   ❌ Set environment: export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251")
        
        if not checks['mesh_file']:
            print(f"   ❌ Add mesh file: meshes/78_MRA1_seg_aneurysm.stl")
        
        if not checks['boundary_conditions']:
            print(f"   ❌ Boundary conditions file missing")
    
    print(f"\n📚 Documentation:")
    print(f"   📖 README.md - Complete setup guide")
    print(f"   📋 PYFLUENT_IMPLEMENTATION_GUIDE.md - Technical details")
    print(f"   🔬 analyze_78_MRA1_seg.py - Main analysis script")

def main():
    """Main setup and validation function"""
    print(f"🚀 PyFluent Aneurysm Analysis Setup")
    print(f"=" * 60)
    print(f"Patient: 78_MRA1_seg")
    print(f"Analysis: Aneurysm hemodynamics CFD")
    print(f"Solver: Ansys Fluent 2025 R1 via PyFluent")
    print(f"=" * 60)
    
    # Check environment
    checks = check_environment()
    
    # Test PyFluent if available
    if checks['ansys_installation'] and checks['pyfluent_available']:
        pyfluent_works = test_pyfluent_connection()
        checks['pyfluent_connection'] = pyfluent_works
    else:
        print(f"\n⚠️  Skipping PyFluent connection test (missing prerequisites)")
    
    # Show configuration
    show_analysis_summary()
    
    # Show next steps
    show_next_steps(checks)
    
    # Summary
    ready_count = sum(checks.values())
    total_count = len(checks)
    
    print(f"\n📊 Setup Status: {ready_count}/{total_count} checks passed")
    
    if ready_count == total_count:
        print(f"🎯 Status: ✅ READY FOR ANALYSIS")
        return 0
    else:
        print(f"🎯 Status: ⚠️  SETUP REQUIRED")
        return 1

if __name__ == "__main__":
    exit(main()) 