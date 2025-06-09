#!/usr/bin/env python3
"""
Test New PyFluent 0.32.1 APIs
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

def test_new_apis():
    """Test the new higher-level Python APIs in PyFluent 0.32.1"""
    logger.info("=== Testing PyFluent 0.32.1 New APIs ===")
    
    # Launch Fluent
    logger.info("1. Launching Fluent...")
    session = launch_fluent(dimension=3, precision="double", processor_count=2, ui_mode="no_gui")
    logger.info("✓ Fluent launched successfully")
    
    try:
        # Check if new file APIs exist
        logger.info("2. Checking new file APIs...")
        
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
        
        # Test STL import with new API
        logger.info("3. Testing STL import with new API...")
        stl_path = Path(__file__).parent.parent / "meshes" / "78_MRA1_seg_aneurysm_ASCII.stl"
        
        if hasattr(session.file, 'read_geometry') and stl_path.exists():
            try:
                logger.info(f"Attempting to read STL as geometry: {stl_path}")
                session.file.read_geometry(file_type="stl", file_name=str(stl_path))
                logger.info("✓ STL imported as geometry successfully!")
            except Exception as e:
                logger.error(f"✗ STL import as geometry failed: {e}")
        
        if hasattr(session.file, 'read_mesh') and stl_path.exists():
            try:
                logger.info(f"Attempting to read STL as mesh: {stl_path}")
                session.file.read_mesh(file_format="stl", mesh_file=str(stl_path))
                logger.info("✓ STL imported as mesh successfully!")
            except Exception as e:
                logger.error(f"✗ STL import as mesh failed: {e}")
        
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
    logger.info("Testing PyFluent 0.32.1 New APIs")
    test_new_apis()
    logger.info("Test complete.")

if __name__ == "__main__":
    main() 