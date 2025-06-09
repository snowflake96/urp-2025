#!/usr/bin/env python3
"""
Explore TUI Commands in PyFluent
Author: Jiwoo Lee
"""

import os
import sys
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

def explore_tui():
    """Launch Fluent and explore TUI commands."""
    logger.info("Launching Fluent to explore TUI commands...")
    fluent = launch_fluent(version="3d", precision="double", processor_count=1, show_gui=False)
    
    try:
        # Check file operations
        logger.info("Exploring file operations...")
        file_attrs = [attr for attr in dir(fluent.tui.file) if not attr.startswith('_')]
        logger.info(f"Available file operations: {file_attrs}")
        
        # Check if import_ exists and what it contains
        if hasattr(fluent.tui.file, 'import_'):
            import_attrs = [attr for attr in dir(fluent.tui.file.import_) if not attr.startswith('_')]
            logger.info(f"Available import operations: {import_attrs}")
        
        # Check mesh operations
        logger.info("Exploring mesh operations...")
        if hasattr(fluent.tui, 'mesh'):
            mesh_attrs = [attr for attr in dir(fluent.tui.mesh) if not attr.startswith('_')]
            logger.info(f"Available mesh operations: {mesh_attrs}")
        
        # Check define operations
        logger.info("Exploring define operations...")
        if hasattr(fluent.tui, 'define'):
            define_attrs = [attr for attr in dir(fluent.tui.define) if not attr.startswith('_')]
            logger.info(f"Available define operations: {define_attrs}")
            
    except Exception as e:
        logger.error(f"Error exploring TUI: {e}")
        
    finally:
        fluent.exit()
        logger.info("Fluent session closed.")

if __name__ == "__main__":
    explore_tui() 