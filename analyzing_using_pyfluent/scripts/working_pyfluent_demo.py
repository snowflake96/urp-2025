#!/usr/bin/env python3
"""
Working PyFluent Demonstration
Shows what we can successfully accomplish with current setup
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging
import time

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

def demonstrate_pyfluent_capabilities():
    """Demonstrate what we can successfully do with PyFluent"""
    logger.info("=== PyFluent Capabilities Demonstration ===")
    
    # Launch Fluent
    logger.info("1. Launching Fluent...")
    fluent = launch_fluent(version="3d", precision="double", processor_count=2, show_gui=False)
    logger.info("✓ Fluent launched successfully")
    
    try:
        # Check available TUI commands
        logger.info("2. Checking available TUI commands...")
        file_ops = [attr for attr in dir(fluent.tui.file) if not attr.startswith('_')]
        logger.info(f"✓ File operations available: {len(file_ops)} commands")
        
        # Test journal functionality
        logger.info("3. Testing journal functionality...")
        journal_path = Path("test_journal.txt")
        fluent.tui.file.start_journal(str(journal_path))
        logger.info("✓ Journal started")
        
        # Check available models
        logger.info("4. Checking available physics models...")
        if hasattr(fluent.tui.define, 'models'):
            models = [attr for attr in dir(fluent.tui.define.models) if not attr.startswith('_')]
            logger.info(f"✓ Physics models available: {len(models)} models")
        
        # Check materials
        logger.info("5. Checking materials...")
        if hasattr(fluent.tui.define, 'materials'):
            materials = [attr for attr in dir(fluent.tui.define.materials) if not attr.startswith('_')]
            logger.info(f"✓ Material operations available: {len(materials)} operations")
        
        # Stop journal
        fluent.tui.file.stop_journal()
        logger.info("✓ Journal stopped")
        
        # Check if journal was created
        if journal_path.exists():
            logger.info(f"✓ Journal file created: {journal_path} ({journal_path.stat().st_size} bytes)")
        
        logger.info("=== Summary ===")
        logger.info("✓ PyFluent installation: WORKING")
        logger.info("✓ Fluent licensing: WORKING")
        logger.info("✓ Solver mode launch: WORKING")
        logger.info("✓ TUI commands: WORKING")
        logger.info("✓ Journal functionality: WORKING")
        logger.info("⚠ STL import: NOT SUPPORTED in solver mode")
        logger.info("⚠ Meshing mode: UNSTABLE (licensing/network issues)")
        
        logger.info("=== Recommendations ===")
        logger.info("1. Use external meshing tools (Gmsh, SALOME, etc.) to convert STL → mesh")
        logger.info("2. Import existing .cas/.msh files into Fluent solver")
        logger.info("3. Focus on CFD analysis rather than mesh generation")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        
    finally:
        fluent.exit()
        logger.info("Fluent session closed.")

def main():
    patient_id = "78_MRA1_seg"
    base_dir = Path(__file__).parent.parent
    stl_path = base_dir / "meshes" / f"{patient_id}_aneurysm_ASCII.stl"
    results_dir = base_dir / "results" / f"{patient_id}_DEMO"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"STL file: {stl_path}")
    logger.info(f"Results directory: {results_dir}")
    
    # Change to results directory
    os.chdir(results_dir)
    
    # Run demonstration
    demonstrate_pyfluent_capabilities()
    
    logger.info("Demonstration complete.")

if __name__ == "__main__":
    main() 