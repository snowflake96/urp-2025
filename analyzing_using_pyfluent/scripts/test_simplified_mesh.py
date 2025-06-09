#!/usr/bin/env python3
"""
Test Simplified STL Mesh with PyFluent
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Ansys environment
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
os.environ['PATH'] = f"/opt/cvbml/softwares/ansys_inc/v251/fluent/bin:{os.environ.get('PATH', '')}"

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    logger.info(f"PyFluent version: {pyfluent.__version__}")
except ImportError as e:
    logger.error(f"Failed to import PyFluent: {e}")
    sys.exit(1)

def test_simplified_mesh(mesh_path, mesh_name):
    """Test a simplified mesh with PyFluent"""
    logger.info(f"🧪 Testing {mesh_name} mesh: {mesh_path}")
    
    # Launch Fluent
    logger.info("1. Launching Fluent...")
    session = launch_fluent(dimension=3, precision="double", processor_count=2, ui_mode="no_gui")
    logger.info("✓ Fluent launched successfully")
    
    try:
        # Try to import STL
        logger.info("2. Importing STL mesh...")
        session.file.read_mesh(file_name=str(mesh_path))
        logger.info("✓ STL imported successfully!")
        
        # Check mesh info
        logger.info("3. Checking mesh information...")
        
        # Print mesh statistics using TUI
        logger.info("Mesh statistics:")
        session.tui.mesh.check()
        
        # Set up basic physics
        logger.info("4. Setting up basic physics...")
        session.tui.define.models.viscous.laminar("yes")
        logger.info("✓ Laminar viscous model set")
        
        # Try basic initialization
        logger.info("5. Initializing flow...")
        session.tui.solve.initialize.compute_defaults.all_zones()
        session.tui.solve.initialize.initialize_flow()
        logger.info("✓ Flow initialized")
        
        # Run a few iterations
        logger.info("6. Running test iterations...")
        session.tui.solve.iterate(5)
        logger.info("✓ 5 iterations completed successfully!")
        
        # Save results
        results_dir = Path(__file__).parent.parent / "results" / f"simplified_{mesh_name}_test"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        case_path = results_dir / f"{mesh_name}_test.cas"
        session.file.write_case(file_name=str(case_path))
        logger.info(f"✓ Case saved: {case_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed for {mesh_name}: {e}")
        return False
        
    finally:
        session.exit()
        logger.info("Fluent session closed.")

def main():
    base_dir = Path(__file__).parent.parent
    simplified_dir = base_dir / "meshes" / "simplified"
    
    logger.info("=" * 80)
    logger.info("🧪 Testing Simplified STL Meshes with PyFluent")
    logger.info("=" * 80)
    
    # List of simplified meshes to test
    test_meshes = [
        "78_MRA1_seg_aneurysm_medium.stl",  # Try medium first
        "78_MRA1_seg_aneurysm_light.stl",   # Then light
        "78_MRA1_seg_aneurysm_heavy.stl",   # Then heavy
        "78_MRA1_seg_aneurysm_extreme.stl"  # Finally extreme
    ]
    
    results = {}
    
    for mesh_file in test_meshes:
        mesh_path = simplified_dir / mesh_file
        mesh_name = mesh_file.replace("78_MRA1_seg_aneurysm_", "").replace(".stl", "")
        
        if not mesh_path.exists():
            logger.warning(f"⚠️  Mesh file not found: {mesh_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING: {mesh_name.upper()} MESH")
        logger.info(f"{'='*60}")
        
        success = test_simplified_mesh(mesh_path, mesh_name)
        results[mesh_name] = success
        
        if success:
            logger.info(f"🎉 {mesh_name.upper()} MESH: SUCCESS!")
            break  # Stop at first successful mesh
        else:
            logger.error(f"❌ {mesh_name.upper()} MESH: FAILED")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    successful_meshes = [name for name, success in results.items() if success]
    failed_meshes = [name for name, success in results.items() if not success]
    
    if successful_meshes:
        logger.info(f"✅ SUCCESSFUL MESHES: {', '.join(successful_meshes)}")
        logger.info("🎯 RECOMMENDATION: Use the first successful mesh for production")
    else:
        logger.info("❌ NO SUCCESSFUL MESHES")
        logger.info("🔧 RECOMMENDATION: Further mesh simplification needed")
    
    if failed_meshes:
        logger.info(f"❌ FAILED MESHES: {', '.join(failed_meshes)}")
    
    logger.info(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    main() 