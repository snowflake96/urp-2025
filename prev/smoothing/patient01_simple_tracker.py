#!/usr/bin/env python3
"""
Simple Patient 01 Vessel Tracker
- Extract fixed-size vessel regions around aneurysm seeds
- Simpler approach without complex cross-sectional area calculations
- Generate STL files for each aneurysm region
"""

import numpy as np
import nibabel as nib
import trimesh
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from skimage import measure, morphology
from scipy import ndimage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVesselTracker:
    """Simple vessel tracker that extracts regions around aneurysms"""
    
    def __init__(self, region_size: Tuple[int, int, int] = (50, 50, 30)):
        """
        Initialize simple vessel tracker
        
        Args:
            region_size: Size of region to extract around each aneurysm (x, y, z)
        """
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        self.original_dir = self.uan_base / "original"
        self.test_dir = self.uan_base / "test" / "patient01"
        
        self.region_size = region_size
        self.patient_data = None
        
        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def load_patient_data(self) -> bool:
        """Load patient data from Excel file"""
        try:
            if not self.excel_path.exists():
                logger.error(f"Excel file not found: {self.excel_path}")
                return False
                
            df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded Excel file with {len(df)} rows")
            
            # Find patient 01 (VintID = 1)
            patient_01_data = df[df['VintID'] == 1]
            
            if len(patient_01_data) == 0:
                logger.error("Patient 01 not found in Excel file")
                return False
                
            self.patient_data = patient_01_data.iloc[0]
            logger.info(f"Found patient 01 data: {self.patient_data['VintID']}")
            
            # Check aneurysm locations
            anatomical_regions = ['ACA', 'Acom', 'ICA (total)', 'ICA (noncavernous)', 
                                 'ICA (cavernous)', 'Pcom', 'BA', 'Other_posterior', 'PCA']
            
            aneurysm_locations = {}
            for region in anatomical_regions:
                if region in self.patient_data and pd.notna(self.patient_data[region]):
                    count = int(self.patient_data[region])
                    if count > 0:
                        aneurysm_locations[region] = count
                        logger.info(f"Patient 01 has {count} aneurysm(s) in {region}")
            
            self.aneurysm_locations = aneurysm_locations
            
            if not aneurysm_locations:
                logger.warning("No aneurysms found for patient 01 in doctor's diagnosis")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return False
    
    def load_vessel_data(self, mra_num: int) -> Tuple[Optional[np.ndarray], Optional[nib.Nifti1Image]]:
        """Load vessel data for given MRA number"""
        try:
            # Use original file (not smoothed since they were empty)
            filename = f"01_MRA{mra_num}_seg.nii.gz"
            file_path = self.original_dir / filename
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None, None
                
            img = nib.load(file_path)
            data = img.get_fdata()
            
            logger.info(f"Loaded {filename}: shape {data.shape}, non-zero voxels: {np.sum(data > 0)}")
            return data, img
            
        except Exception as e:
            logger.error(f"Error loading vessel data for MRA {mra_num}: {e}")
            return None, None
    
    def find_aneurysm_centers(self, vessel_data: np.ndarray, region: str) -> List[Tuple[int, int, int]]:
        """Find center points for aneurysms in specific anatomical region"""
        
        # Define approximate anatomical regions based on spatial location
        shape = vessel_data.shape
        region_masks = {
            'ACA': {'z_range': (0.6, 1.0), 'y_range': (0.0, 0.4), 'x_range': (0.3, 0.7)},
            'Acom': {'z_range': (0.5, 0.8), 'y_range': (0.2, 0.6), 'x_range': (0.4, 0.6)},
            'ICA (total)': {'z_range': (0.2, 0.8), 'y_range': (0.3, 0.8), 'x_range': (0.2, 0.8)},
            'ICA (noncavernous)': {'z_range': (0.4, 0.8), 'y_range': (0.3, 0.7), 'x_range': (0.2, 0.8)},
            'ICA (cavernous)': {'z_range': (0.2, 0.6), 'y_range': (0.4, 0.8), 'x_range': (0.2, 0.8)},
            'Pcom': {'z_range': (0.4, 0.7), 'y_range': (0.4, 0.8), 'x_range': (0.3, 0.7)},
            'BA': {'z_range': (0.3, 0.7), 'y_range': (0.6, 1.0), 'x_range': (0.4, 0.6)},
            'Other_posterior': {'z_range': (0.2, 0.6), 'y_range': (0.7, 1.0), 'x_range': (0.2, 0.8)},
            'PCA': {'z_range': (0.3, 0.6), 'y_range': (0.5, 0.9), 'x_range': (0.2, 0.8)}
        }
        
        if region not in region_masks:
            logger.warning(f"Unknown anatomical region: {region}")
            return [(shape[0]//2, shape[1]//2, shape[2]//2)]
        
        # Create region mask
        mask_info = region_masks[region]
        x_range = [int(mask_info['x_range'][0] * shape[0]), int(mask_info['x_range'][1] * shape[0])]
        y_range = [int(mask_info['y_range'][0] * shape[1]), int(mask_info['y_range'][1] * shape[1])]
        z_range = [int(mask_info['z_range'][0] * shape[2]), int(mask_info['z_range'][1] * shape[2])]
        
        # Extract vessel data in this region
        region_vessels = vessel_data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
        
        if np.sum(region_vessels > 0) == 0:
            logger.warning(f"No vessels found in {region} region")
            return []
        
        # Find multiple seed points by dividing region into sub-regions
        centers = []
        
        # For ICA regions, create multiple subdivisions to capture different aneurysms
        if 'ICA' in region:
            # Create a 2x2 grid within the region
            x_mid = (x_range[0] + x_range[1]) // 2
            y_mid = (y_range[0] + y_range[1]) // 2
            
            sub_regions = [
                (x_range[0], x_mid, y_range[0], y_mid),  # Bottom-left
                (x_mid, x_range[1], y_range[0], y_mid),  # Bottom-right
                (x_range[0], x_mid, y_mid, y_range[1]),  # Top-left
                (x_mid, x_range[1], y_mid, y_range[1])   # Top-right
            ]
            
            for x_start, x_end, y_start, y_end in sub_regions:
                sub_region = vessel_data[x_start:x_end, y_start:y_end, z_range[0]:z_range[1]]
                if np.sum(sub_region > 0) > 100:  # Only if significant vessel presence
                    # Find center of mass in this sub-region
                    coords = np.where(sub_region > 0)
                    if len(coords[0]) > 0:
                        center = [
                            int(np.mean(coords[0])) + x_start,
                            int(np.mean(coords[1])) + y_start,
                            int(np.mean(coords[2])) + z_range[0]
                        ]
                        centers.append(tuple(center))
                        logger.info(f"Found aneurysm center in {region} at {center}")
        else:
            # For other regions, just use the center
            vessel_coords = np.where(region_vessels > 0)
            if len(vessel_coords[0]) > 0:
                center = [
                    int(np.mean(vessel_coords[0])) + x_range[0],
                    int(np.mean(vessel_coords[1])) + y_range[0],
                    int(np.mean(vessel_coords[2])) + z_range[0]
                ]
                centers.append(tuple(center))
                logger.info(f"Found aneurysm center in {region} at {center}")
        
        return centers
    
    def extract_region_around_center(self, vessel_data: np.ndarray, center: Tuple[int, int, int]) -> np.ndarray:
        """Extract vessel region around a center point"""
        
        x, y, z = center
        rx, ry, rz = self.region_size
        
        # Calculate extraction bounds
        x_start = max(0, x - rx//2)
        x_end = min(vessel_data.shape[0], x + rx//2)
        y_start = max(0, y - ry//2)
        y_end = min(vessel_data.shape[1], y + ry//2)
        z_start = max(0, z - rz//2)
        z_end = min(vessel_data.shape[2], z + rz//2)
        
        # Extract region
        extracted = vessel_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Ensure region has vessel data
        if np.sum(extracted > 0) == 0:
            logger.warning(f"No vessel data in extracted region around {center}")
            return np.zeros((rx, ry, rz))
        
        # Pad if necessary to get consistent size
        padded = np.zeros((rx, ry, rz))
        actual_size = extracted.shape
        
        padded[:actual_size[0], :actual_size[1], :actual_size[2]] = extracted
        
        logger.info(f"Extracted region: {actual_size} -> {padded.shape}, vessel voxels: {np.sum(padded > 0)}")
        return padded
    
    def create_mesh_from_region(self, region_data: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Optional[trimesh.Trimesh]:
        """Create 3D mesh from vessel region"""
        
        if np.sum(region_data > 0) == 0:
            return None
        
        try:
            # Apply marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                region_data, level=0.5, spacing=voxel_spacing
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Clean up mesh
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            
            if mesh.is_empty:
                return None
                
            logger.info(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return None
    
    def process_patient_01(self):
        """Process patient 01 for simple vessel tracking"""
        
        logger.info("Starting simple vessel tracking for Patient 01")
        
        # Load patient data
        if not self.load_patient_data():
            logger.error("Failed to load patient data")
            return
        
        file_count = 0
        
        # Process both MRA scans
        for mra_num in [1, 2]:
            logger.info(f"\n=== Processing MRA {mra_num} ===")
            
            # Load vessel data
            vessel_data, img = self.load_vessel_data(mra_num)
            if vessel_data is None:
                logger.error(f"Failed to load vessel data for MRA {mra_num}")
                continue
            
            # Get voxel spacing
            voxel_spacing = img.header.get_zooms()[:3]
            logger.info(f"Voxel spacing: {voxel_spacing} mm")
            
            # Process each aneurysm location
            for region, count in self.aneurysm_locations.items():
                logger.info(f"\nProcessing {count} aneurysm(s) in {region}")
                
                # Find centers for aneurysms in this region
                centers = self.find_aneurysm_centers(vessel_data, region)
                
                # Process each center (up to the doctor's count)
                processed_centers = min(len(centers), count)
                for i in range(processed_centers):
                    center = centers[i]
                    file_count += 1
                    
                    logger.info(f"Extracting vessel region {file_count} around {region} at {center}")
                    
                    # Extract region around center
                    region_data = self.extract_region_around_center(vessel_data, center)
                    
                    if np.sum(region_data > 0) > 0:
                        # Create mesh
                        mesh = self.create_mesh_from_region(region_data, voxel_spacing)
                        
                        if mesh:
                            # Save mesh as STL
                            region_safe = region.replace(' ', '_').replace('(', '').replace(')', '')
                            filename = f"01_MRA{mra_num}_{region_safe}_aneurysm_{i+1}_vessels.stl"
                            output_path = self.test_dir / filename
                            
                            mesh.export(str(output_path))
                            logger.info(f"Saved vessel region: {output_path}")
                            
                            # Log mesh stats
                            logger.info(f"Mesh volume: {abs(mesh.volume):.2f} mm³")
                            logger.info(f"Mesh surface area: {mesh.area:.2f} mm²")
                        else:
                            logger.warning(f"Failed to create mesh for region {file_count}")
                    else:
                        logger.warning(f"No vessel data in extracted region {file_count}")
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Results saved in: {self.test_dir}")
        
        # List generated files
        stl_files = list(self.test_dir.glob("*.stl"))
        logger.info(f"Generated {len(stl_files)} STL files:")
        for stl_file in stl_files:
            logger.info(f"  - {stl_file.name}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple vessel tracking for Patient 01")
    parser.add_argument("--region-size", nargs=3, type=int, default=[50, 50, 30], 
                       help="Region size to extract around aneurysms (x y z)")
    
    args = parser.parse_args()
    
    tracker = SimpleVesselTracker(region_size=tuple(args.region_size))
    tracker.process_patient_01()

if __name__ == "__main__":
    main() 