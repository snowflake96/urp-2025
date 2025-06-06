#!/usr/bin/env python3
"""
Aneurysm Cropping Script
Crops local vascular regions around aneurysms using COSTA segmentation results
"""

import pandas as pd
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import json
import logging
from scipy import ndimage
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

# Set up logging
def setup_logging(output_dir):
    """Setup logging configuration"""
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'aneurysm_cropping.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger (will be properly set up when AneurysmCropper is created)
logger = logging.getLogger(__name__)

class AneurysmCropper:
    def __init__(self, excel_file=None):
        # Update paths to use the new data directory structure
        if excel_file is None:
            excel_file = Path.home() / "urp" / "data" / "segmentation" / "aneu" / "SNHU_TAnDB_DICOM.xlsx"
        self.excel_file = excel_file
        self.patient_data = None
        self.seg_base_path = Path.home() / "urp" / "data" / "segmentation" / "aneu" / "UAN_processed" / "Input"
        self.output_base = Path.home() / "urp" / "data" / "aneurysm_cropping"
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Set up logging with the new output directory
        global logger
        logger = setup_logging(self.output_base)
        
        self.load_patient_data() 