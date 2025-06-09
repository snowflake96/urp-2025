#!/usr/bin/env python3
"""
Test PyFluent 0.28.0 APIs
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging
import inspect

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

def test_apis():
    """Test available APIs in PyFluent 0.28.0"""
    logger.info("=== Testing PyFluent 0.28.0 APIs ===")
    
    # Launch Fluent
    logger.info("1. Launching Fluent...")
    session = launch_fluent(version="3d", precision="double", processor_count=2, show_gui=False)
    logger.info("✓ Fluent launched successfully")
    
    try:
        # Check if new file APIs exist
        logger.info("2. Checking file APIs...")
        
        if hasattr(session, 'file'):
            logger.info("✓ session.file exists")
            file_attrs = [attr for attr in dir(session.file) if not attr.startswith('_')]
            logger.info(f"Available file operations: {file_attrs}")
            
            # Check read_geometry
            if hasattr(session.file, 'read_geometry'):
                logger.info("✓ session.file.read_geometry exists")
                try:
                    sig = inspect.signature(session.file.read_geometry)
                    logger.info(f"read_geometry signature: {sig}")
                except Exception as e:
                    logger.warning(f"Could not get signature: {e}")
            else:
                logger.warning("✗ session.file.read_geometry does NOT exist")
            
            # Check read_mesh
            if hasattr(session.file, 'read_mesh'):
                logger.info("✓ session.file.read_mesh exists")
                try:
                    sig = inspect.signature(session.file.read_mesh)
                    logger.info(f"read_mesh signature: {sig}")
                except Exception as e:
                    logger.warning(f"Could not get signature: {e}")
            else:
                logger.warning("✗ session.file.read_mesh does NOT exist")
                
        else:
            logger.warning("✗ session.file does NOT exist")
        
        # Check TUI methods
        logger.info("3. Checking TUI methods...")
        if hasattr(session, 'tui'):
            logger.info("✓ session.tui exists")
            if hasattr(session.tui, 'file'):
                logger.info("✓ session.tui.file exists")
                tui_file_attrs = [attr for attr in dir(session.tui.file) if not attr.startswith('_')]
                logger.info(f"Available TUI file operations: {tui_file_attrs}")
        
        # Check other available methods
        logger.info("4. Checking other session attributes...")
        session_attrs = [attr for attr in dir(session) if not attr.startswith('_')]
        logger.info(f"Available session attributes: {session_attrs}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        
    finally:
        session.exit()
        logger.info("Fluent session closed.")

def main():
    logger.info("Testing PyFluent 0.28.0 APIs")
    test_apis()
    logger.info("Test complete.")

if __name__ == "__main__":
    main() 