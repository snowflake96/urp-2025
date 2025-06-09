#!/usr/bin/env python3
"""
Minimal PyFluent Test Script
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

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    logger.info(f"PyFluent version: {pyfluent.__version__}")
except ImportError as e:
    logger.error(f"Failed to import PyFluent: {e}")
    sys.exit(1)

def test_fluent():
    """Launch Fluent and run a simple TUI command."""
    logger.info("Launching Fluent...")
    fluent = launch_fluent(version="3d", precision="double", processor_count=1, show_gui=False)
    logger.info("Fluent launched successfully.")
    # Run a simple TUI command
    fluent.tui.file.start_journal("test_journal.txt")
    logger.info("Journal started.")
    fluent.tui.file.stop_journal()
    logger.info("Journal stopped.")
    fluent.exit()
    logger.info("Fluent exited successfully.")

if __name__ == "__main__":
    test_fluent() 