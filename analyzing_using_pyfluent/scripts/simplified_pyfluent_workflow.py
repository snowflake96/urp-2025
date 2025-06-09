#!/usr/bin/env python3
"""
Simplified PyFluent CFD Workflow
Read existing mesh → PyFluent CFD → VTP/VTU
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging
import time
import numpy as np

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

def run_simplified_cfd(stl_path, results_dir):
    """Run simplified CFD analysis using PyFluent"""
    logger.info(f"Running simplified CFD on STL: {stl_path}")
    
    # Launch Fluent in solver mode with reduced cores
    fluent = launch_fluent(version="3d", precision="double", processor_count=2, show_gui=False)
    
    try:
        # Try to import STL directly in solver mode (newer versions support this)
        logger.info("Attempting to import STL file...")
        
        # Create a simple case from STL if possible
        fluent.tui.file.import_stl(str(stl_path))
        
        # Set up physics with minimal configuration
        logger.info("Setting up physics models...")
        fluent.tui.define.models.viscous.laminar("yes")
        
        # Try to get boundary zones
        logger.info("Checking boundary zones...")
        boundaries = fluent.get_boundary_zone_names()
        logger.info(f"Available boundaries: {boundaries}")
        
        # Save case
        case_path = results_dir / "simple_case.cas"
        fluent.tui.file.write_case(str(case_path))
        logger.info(f"Case saved to: {case_path}")
        
    except Exception as e:
        logger.error(f"Error during CFD setup: {e}")
        
    finally:
        fluent.exit()
        logger.info("Fluent session closed.")

def main():
    patient_id = "78_MRA1_seg"
    base_dir = Path(__file__).parent.parent
    stl_path = base_dir / "meshes" / f"{patient_id}_aneurysm_ASCII.stl"
    results_dir = base_dir / "results" / f"{patient_id}_SIMPLIFIED_CFD"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting simplified workflow for {patient_id}")
    logger.info(f"STL file: {stl_path}")
    logger.info(f"Results directory: {results_dir}")
    
    # Run simplified CFD
    run_simplified_cfd(stl_path, results_dir)
    logger.info("Simplified workflow complete.")

if __name__ == "__main__":
    main() 