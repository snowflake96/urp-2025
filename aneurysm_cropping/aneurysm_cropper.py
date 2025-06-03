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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aneurysm_cropping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AneurysmCropper:
    def __init__(self, excel_file="SNHU_TAnDB_DICOM.xlsx"):
        self.excel_file = excel_file
        self.patient_data = None
        self.seg_base_path = Path.home() / "Downloads" / "segmentation" / "aneu" / "UAN_processed" / "Input"
        self.output_base = Path("output")
        self.output_base.mkdir(exist_ok=True)
        
        self.load_patient_data()
    
    def load_patient_data(self):
        """Load patient information from Excel file"""
        try:
            self.patient_data = pd.read_excel(self.excel_file)
            logger.info(f"Loaded patient data: {len(self.patient_data)} patients")
            
            # Clean up column names for easier access
            self.patient_data.columns = [col.strip() for col in self.patient_data.columns]
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            raise
    
    def get_patient_folders(self):
        """Get all available patient segmentation folders"""
        if not self.seg_base_path.exists():
            logger.error(f"Segmentation path not found: {self.seg_base_path}")
            return []
        
        folders = [f for f in self.seg_base_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(folders)} segmentation folders")
        return sorted(folders)
    
    def extract_patient_id(self, folder_name):
        """Extract patient ID from folder name (e.g., '10_MRA1' -> 10)"""
        try:
            return int(folder_name.split('_')[0])
        except:
            return None
    
    def get_aneurysm_locations(self, patient_id):
        """Get aneurysm locations for a patient from the Excel data"""
        patient_row = self.patient_data[self.patient_data['ID'] == patient_id]
        
        if patient_row.empty:
            return []
        
        patient_row = patient_row.iloc[0]
        locations = []
        
        # Check each anatomical location
        location_columns = ['MCA', 'ACA', 'Acom', 'ICA (total)', 'Pcom', 'BA', 'Other_posterior', 'PCA']
        
        for loc in location_columns:
            if loc in patient_row and pd.notna(patient_row[loc]) and patient_row[loc] > 0:
                locations.append(loc)
        
        return locations
    
    def load_nifti_safely(self, file_path):
        """Safely load a NIfTI file"""
        try:
            img = nib.load(str(file_path))
            return img, img.get_fdata()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def find_aneurysm_centers(self, mask_data, min_volume=10):
        """Find centers of aneurysm regions in the mask"""
        # Label connected components
        labeled_mask = measure.label(mask_data > 0, connectivity=3)
        regions = measure.regionprops(labeled_mask)
        
        centers = []
        for region in regions:
            if region.area >= min_volume:  # Filter small regions
                center = [int(coord) for coord in region.centroid]
                centers.append({
                    'center': center,
                    'volume': region.area,
                    'bbox': region.bbox
                })
        
        return centers
    
    def crop_around_center(self, data, center, crop_size=(64, 64, 32)):
        """Crop a region around the aneurysm center"""
        z, y, x = center
        cz, cy, cx = crop_size
        
        # Calculate crop boundaries
        z_start = max(0, z - cz//2)
        z_end = min(data.shape[0], z + cz//2)
        y_start = max(0, y - cy//2)
        y_end = min(data.shape[1], y + cy//2)
        x_start = max(0, x - cx//2)
        x_end = min(data.shape[2], x + cx//2)
        
        # Crop the data
        cropped = data[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary to maintain consistent size
        if cropped.shape != crop_size:
            padded = np.zeros(crop_size)
            
            # Calculate padding
            pad_z = min(crop_size[0], cropped.shape[0])
            pad_y = min(crop_size[1], cropped.shape[1])
            pad_x = min(crop_size[2], cropped.shape[2])
            
            padded[:pad_z, :pad_y, :pad_x] = cropped[:pad_z, :pad_y, :pad_x]
            cropped = padded
        
        crop_info = {
            'original_center': center,
            'crop_bbox': (z_start, z_end, y_start, y_end, x_start, x_end),
            'crop_size': crop_size
        }
        
        return cropped, crop_info
    
    def process_patient(self, folder_path, crop_size=(64, 64, 32)):
        """Process a single patient folder"""
        folder_name = folder_path.name
        patient_id = self.extract_patient_id(folder_name)
        
        if patient_id is None:
            logger.warning(f"Could not extract patient ID from {folder_name}")
            return
        
        logger.info(f"Processing patient {patient_id} ({folder_name})")
        
        # Get patient clinical information
        locations = self.get_aneurysm_locations(patient_id)
        patient_row = self.patient_data[self.patient_data['ID'] == patient_id]
        
        # File paths
        raw_file = folder_path / "Raw" / f"{folder_name}.nii.gz"
        mask_file = folder_path / "Output" / f"{folder_name}.nii.gz"
        
        if not raw_file.exists() or not mask_file.exists():
            logger.warning(f"Missing files for {folder_name}")
            return
        
        # Load images
        raw_img, raw_data = self.load_nifti_safely(raw_file)
        mask_img, mask_data = self.load_nifti_safely(mask_file)
        
        if raw_data is None or mask_data is None:
            logger.error(f"Failed to load images for {folder_name}")
            return
        
        # Find aneurysm centers
        aneurysm_centers = self.find_aneurysm_centers(mask_data)
        
        if not aneurysm_centers:
            logger.warning(f"No aneurysms found in mask for {folder_name}")
            return
        
        # Create output directory for this patient
        patient_output_dir = self.output_base / f"patient_{patient_id}" / folder_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each aneurysm
        for i, aneurysm_info in enumerate(aneurysm_centers):
            center = aneurysm_info['center']
            
            # Crop raw image and mask
            cropped_raw, crop_info = self.crop_around_center(raw_data, center, crop_size)
            cropped_mask, _ = self.crop_around_center(mask_data, center, crop_size)
            
            # Save cropped images
            raw_filename = patient_output_dir / f"aneurysm_{i+1}_raw.nii.gz"
            mask_filename = patient_output_dir / f"aneurysm_{i+1}_mask.nii.gz"
            
            # Create new NIfTI images with original header info
            raw_cropped_img = nib.Nifti1Image(cropped_raw, raw_img.affine, raw_img.header)
            mask_cropped_img = nib.Nifti1Image(cropped_mask, mask_img.affine, mask_img.header)
            
            nib.save(raw_cropped_img, raw_filename)
            nib.save(mask_cropped_img, mask_filename)
            
            # Create metadata
            metadata = {
                'patient_id': patient_id,
                'folder_name': folder_name,
                'aneurysm_index': i + 1,
                'aneurysm_info': aneurysm_info,
                'crop_info': crop_info,
                'anatomical_locations': locations,
                'crop_size': crop_size
            }
            
            # Add clinical information if available
            if not patient_row.empty:
                patient_info = patient_row.iloc[0].to_dict()
                # Convert numpy types to native Python types for JSON serialization
                patient_info = {k: (v if pd.notna(v) and not isinstance(v, (np.integer, np.floating)) 
                                  else (int(v) if isinstance(v, np.integer) 
                                       else (float(v) if isinstance(v, np.floating) and pd.notna(v)
                                            else None))) 
                               for k, v in patient_info.items()}
                metadata['clinical_info'] = patient_info
            
            # Save metadata
            metadata_filename = patient_output_dir / f"aneurysm_{i+1}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  Saved aneurysm {i+1}: volume={aneurysm_info['volume']}, center={center}")
    
    def process_all_patients(self, max_patients=None, crop_size=(64, 64, 32)):
        """Process all available patients"""
        patient_folders = self.get_patient_folders()
        
        if max_patients:
            patient_folders = patient_folders[:max_patients]
        
        logger.info(f"Processing {len(patient_folders)} patients...")
        
        processed = 0
        failed = 0
        
        for folder in patient_folders:
            try:
                self.process_patient(folder, crop_size)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process {folder.name}: {e}")
                failed += 1
        
        logger.info(f"Processing complete: {processed} successful, {failed} failed")
        
        # Create summary
        self.create_summary()
    
    def create_summary(self):
        """Create a summary of all processed aneurysms"""
        summary_data = []
        
        for patient_dir in self.output_base.glob("patient_*"):
            for scan_dir in patient_dir.iterdir():
                if scan_dir.is_dir():
                    for metadata_file in scan_dir.glob("*_metadata.json"):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            summary_row = {
                                'patient_id': metadata['patient_id'],
                                'folder_name': metadata['folder_name'],
                                'aneurysm_index': metadata['aneurysm_index'],
                                'volume': metadata['aneurysm_info']['volume'],
                                'center_x': metadata['aneurysm_info']['center'][0],
                                'center_y': metadata['aneurysm_info']['center'][1],
                                'center_z': metadata['aneurysm_info']['center'][2],
                                'anatomical_locations': ', '.join(metadata.get('anatomical_locations', [])),
                                'raw_file': str(scan_dir / f"aneurysm_{metadata['aneurysm_index']}_raw.nii.gz"),
                                'mask_file': str(scan_dir / f"aneurysm_{metadata['aneurysm_index']}_mask.nii.gz")
                            }
                            
                            # Add clinical info if available
                            if 'clinical_info' in metadata:
                                clinical = metadata['clinical_info']
                                summary_row.update({
                                    'age': clinical.get('연령'),
                                    'gender': clinical.get('성별'),
                                    'smoking': clinical.get('Smoking_record_combined (non:0, current:1, ex:2)'),
                                    'hypertension': clinical.get('HT'),
                                    'diabetes': clinical.get('DM'),
                                    'aneurysm_size': clinical.get('Max or ruptureA size')
                                })
                            
                            summary_data.append(summary_row)
                            
                        except Exception as e:
                            logger.error(f"Error processing metadata {metadata_file}: {e}")
        
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_base / "aneurysm_cropping_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Summary saved: {len(summary_data)} aneurysms processed")
            
            # Print basic statistics
            print("\n=== PROCESSING SUMMARY ===")
            print(f"Total aneurysms processed: {len(summary_data)}")
            print(f"Unique patients: {summary_df['patient_id'].nunique()}")
            print(f"Average aneurysm volume: {summary_df['volume'].mean():.1f} voxels")
            print(f"Volume range: {summary_df['volume'].min():.0f} - {summary_df['volume'].max():.0f} voxels")
            if 'age' in summary_df.columns:
                print(f"Age range: {summary_df['age'].min():.0f} - {summary_df['age'].max():.0f} years")


def main():
    print("=== Aneurysm Cropping Tool ===\n")
    
    # Initialize cropper
    cropper = AneurysmCropper()
    
    # Process patients (start with a small subset for testing)
    print("Starting with first 5 patients for testing...")
    cropper.process_all_patients(max_patients=5, crop_size=(64, 64, 32))
    
    print("\nTo process all patients, call:")
    print("cropper.process_all_patients(crop_size=(64, 64, 32))")

if __name__ == "__main__":
    main() 