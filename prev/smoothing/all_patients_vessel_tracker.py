#!/usr/bin/env python3
"""
All Patients Vessel Tracker
- Process all patients (01-84) with enhanced random walk
- Double walking size (20,000 voxels per aneurysm)
- Use 16 CPUs for parallel processing
- Create individual patient folders under test/
- Background processing support
"""

import numpy as np
import nibabel as nib
import trimesh
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Set
from skimage import measure, morphology, filters
from scipy import ndimage, spatial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import deque
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AllPatientsVesselTracker:
    """Enhanced vessel tracker for all patients with parallel processing"""
    
    def __init__(self, min_cross_section_area: float = 0.5, max_walking_steps: int = 20000):
        """
        Initialize all patients vessel tracker
        
        Args:
            min_cross_section_area: Minimum cross-sectional area in mm² to continue tracking
            max_walking_steps: Maximum steps for random walk (doubled from 10000 to 20000)
        """
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        self.original_dir = self.uan_base / "original"
        self.test_base_dir = self.uan_base / "test"
        
        self.min_cross_section_area = min_cross_section_area  # mm²
        self.max_walking_steps = max_walking_steps
        self.patient_data_df = None
        
        # Create test base directory
        self.test_base_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_patient_data(self) -> bool:
        """Load all patient data from Excel file"""
        try:
            if not self.excel_path.exists():
                logger.error(f"Excel file not found: {self.excel_path}")
                return False
                
            self.patient_data_df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded Excel file with {len(self.patient_data_df)} rows")
            
            # Check which patients have aneurysms
            anatomical_regions = ['ACA', 'Acom', 'ICA (total)', 'ICA (noncavernous)', 
                                 'ICA (cavernous)', 'Pcom', 'BA', 'Other_posterior', 'PCA']
            
            patients_with_aneurysms = []
            for _, row in self.patient_data_df.iterrows():
                try:
                    patient_id = int(row['VintID'])
                    if patient_id < 1 or patient_id > 84:
                        continue
                        
                    # Check if patient has any aneurysms
                    has_aneurysms = False
                    for region in anatomical_regions:
                        if region in row and pd.notna(row[region]):
                            count = int(row[region])
                            if count > 0:
                                has_aneurysms = True
                                break
                    
                    if has_aneurysms:
                        patients_with_aneurysms.append(patient_id)
                        
                except (ValueError, KeyError):
                    continue
            
            logger.info(f"Found {len(patients_with_aneurysms)} patients with aneurysms: {sorted(patients_with_aneurysms)}")
            self.patients_with_aneurysms = sorted(patients_with_aneurysms)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return False
    
    def get_patient_aneurysm_locations(self, patient_id: int) -> Dict[str, int]:
        """Get aneurysm locations for a specific patient"""
        patient_data = self.patient_data_df[self.patient_data_df['VintID'] == patient_id]
        
        if len(patient_data) == 0:
            return {}
            
        patient_row = patient_data.iloc[0]
        
        anatomical_regions = ['ACA', 'Acom', 'ICA (total)', 'ICA (noncavernous)', 
                             'ICA (cavernous)', 'Pcom', 'BA', 'Other_posterior', 'PCA']
        
        aneurysm_locations = {}
        for region in anatomical_regions:
            if region in patient_row and pd.notna(patient_row[region]):
                count = int(patient_row[region])
                if count > 0:
                    aneurysm_locations[region] = count
        
        return aneurysm_locations
    
    def load_vessel_data(self, patient_id: int, mra_num: int) -> Tuple[Optional[np.ndarray], Optional[nib.Nifti1Image]]:
        """Load vessel data for given patient and MRA number"""
        try:
            # Try smoothed first, then original
            patient_str = f"{patient_id:02d}"
            smoothed_filename = f"{patient_str}_MRA{mra_num}_seg_smoothed.nii.gz"
            smoothed_path = self.uan_base / "original_smoothed" / smoothed_filename
            
            original_filename = f"{patient_str}_MRA{mra_num}_seg.nii.gz"
            original_path = self.original_dir / original_filename
            
            # Check smoothed file first
            if smoothed_path.exists():
                img = nib.load(smoothed_path)
                data = img.get_fdata()
                
                if np.sum(data > 0) > 100:  # If not nearly empty, use smoothed
                    logger.debug(f"Loaded smoothed {smoothed_filename}: {np.sum(data > 0)} voxels")
                    return data, img
                else:
                    logger.debug(f"Smoothed file {smoothed_filename} too sparse, trying original")
            
            # Use original file
            if original_path.exists():
                img = nib.load(original_path)
                data = img.get_fdata()
                logger.debug(f"Loaded original {original_filename}: {np.sum(data > 0)} voxels")
                return data, img
            else:
                logger.warning(f"No files found for patient {patient_id} MRA {mra_num}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading vessel data for patient {patient_id} MRA {mra_num}: {e}")
            return None, None
    
    def find_aneurysm_seeds(self, vessel_data: np.ndarray, region: str) -> List[Tuple[int, int, int]]:
        """Find vessel center seed points for aneurysms in specific anatomical region"""
        
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
            return [(shape[0]//2, shape[1]//2, shape[2]//2)]
        
        # Create region mask
        mask_info = region_masks[region]
        x_range = [int(mask_info['x_range'][0] * shape[0]), int(mask_info['x_range'][1] * shape[0])]
        y_range = [int(mask_info['y_range'][0] * shape[1]), int(mask_info['y_range'][1] * shape[1])]
        z_range = [int(mask_info['z_range'][0] * shape[2]), int(mask_info['z_range'][1] * shape[2])]
        
        # Extract vessel data in this region
        region_vessels = vessel_data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
        
        if np.sum(region_vessels > 0) == 0:
            return []
        
        seed_points = []
        
        # Find vessel centers
        if 'ICA' in region:
            # Create subdivisions for ICA regions
            x_mid = (x_range[0] + x_range[1]) // 2
            y_mid = (y_range[0] + y_range[1]) // 2
            
            sub_regions = [
                (x_range[0], x_mid, y_range[0], y_mid),
                (x_mid, x_range[1], y_range[0], y_mid),
                (x_range[0], x_mid, y_mid, y_range[1]),
                (x_mid, x_range[1], y_mid, y_range[1])
            ]
            
            for x_start, x_end, y_start, y_end in sub_regions:
                sub_region = vessel_data[x_start:x_end, y_start:y_end, z_range[0]:z_range[1]]
                if np.sum(sub_region > 0) > 500:
                    vessel_coords = np.where(sub_region > 0)
                    if len(vessel_coords[0]) > 0:
                        center_global = [
                            int(np.mean(vessel_coords[0])) + x_start,
                            int(np.mean(vessel_coords[1])) + y_start,
                            int(np.mean(vessel_coords[2])) + z_range[0]
                        ]
                        
                        # Verify vessel voxel
                        gx, gy, gz = center_global
                        if 0 <= gx < vessel_data.shape[0] and 0 <= gy < vessel_data.shape[1] and 0 <= gz < vessel_data.shape[2]:
                            if vessel_data[gx, gy, gz] > 0:
                                seed_points.append(tuple(center_global))
        else:
            # Single center for other regions
            vessel_coords = np.where(region_vessels > 0)
            if len(vessel_coords[0]) > 0:
                center_global = [
                    int(np.mean(vessel_coords[0])) + x_range[0],
                    int(np.mean(vessel_coords[1])) + y_range[0],
                    int(np.mean(vessel_coords[2])) + z_range[0]
                ]
                
                gx, gy, gz = center_global
                if 0 <= gx < vessel_data.shape[0] and 0 <= gy < vessel_data.shape[1] and 0 <= gz < vessel_data.shape[2]:
                    if vessel_data[gx, gy, gz] > 0:
                        seed_points.append(tuple(center_global))
        
        return seed_points
    
    def calculate_cross_sectional_area(self, vessel_data: np.ndarray, point: Tuple[int, int, int], 
                                     voxel_spacing: Tuple[float, float, float]) -> float:
        """Calculate cross-sectional area at a point"""
        x, y, z = point
        region_size = 3
        
        x_start, x_end = max(0, x-region_size), min(vessel_data.shape[0], x+region_size+1)
        y_start, y_end = max(0, y-region_size), min(vessel_data.shape[1], y+region_size+1)
        z_start, z_end = max(0, z-region_size), min(vessel_data.shape[2], z+region_size+1)
        
        local_vessel = vessel_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        if np.sum(local_vessel > 0) == 0:
            return 0.0
        
        vessel_voxels = np.sum(local_vessel > 0)
        voxel_area = voxel_spacing[0] * voxel_spacing[1]  # mm²
        cross_sectional_area = vessel_voxels * voxel_area  # mm²
        
        return cross_sectional_area
    
    def random_walk_from_seed(self, vessel_data: np.ndarray, seed: Tuple[int, int, int], 
                            voxel_spacing: Tuple[float, float, float]) -> Set[Tuple[int, int, int]]:
        """Enhanced random walk with doubled walking size"""
        
        visited = set()
        vessel_region = set()
        to_visit = deque([seed])
        
        # 26-connectivity neighbors
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        neighbors.append((dx, dy, dz))
        
        step = 0
        consecutive_small_areas = 0
        max_consecutive_small = 50
        
        while to_visit and step < self.max_walking_steps:
            if len(to_visit) == 0:
                break
                
            current = to_visit.popleft()
            
            if current in visited:
                continue
                
            x, y, z = current
            
            # Check bounds
            if (x < 0 or x >= vessel_data.shape[0] or 
                y < 0 or y >= vessel_data.shape[1] or 
                z < 0 or z >= vessel_data.shape[2]):
                continue
            
            # Check if this is a vessel voxel
            if vessel_data[x, y, z] == 0:
                continue
            
            visited.add(current)
            
            # Calculate cross-sectional area
            area = self.calculate_cross_sectional_area(vessel_data, current, voxel_spacing)
            
            # Lenient area threshold
            if area < self.min_cross_section_area:
                consecutive_small_areas += 1
                if consecutive_small_areas > max_consecutive_small:
                    continue
            else:
                consecutive_small_areas = 0
            
            # Add to vessel region
            vessel_region.add(current)
            
            # Add vessel neighbors
            for dx, dy, dz in neighbors:
                neighbor = (x + dx, y + dy, z + dz)
                
                if (neighbor not in visited and 
                    0 <= neighbor[0] < vessel_data.shape[0] and
                    0 <= neighbor[1] < vessel_data.shape[1] and
                    0 <= neighbor[2] < vessel_data.shape[2]):
                    
                    nx, ny, nz = neighbor
                    if vessel_data[nx, ny, nz] > 0:
                        to_visit.append(neighbor)
            
            step += 1
        
        return vessel_region
    
    def extract_vessel_region_mesh(self, vessel_data: np.ndarray, vessel_region: Set[Tuple[int, int, int]], 
                                 voxel_spacing: Tuple[float, float, float]) -> Optional[trimesh.Trimesh]:
        """Extract vessel region as a 3D mesh"""
        
        if not vessel_region:
            return None
        
        # Create binary mask for the vessel region
        mask = np.zeros_like(vessel_data)
        for x, y, z in vessel_region:
            mask[x, y, z] = 1
        
        if np.sum(mask) == 0:
            return None
        
        try:
            # Apply marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=voxel_spacing
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Clean up mesh
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            
            if mesh.is_empty:
                return None
                
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return None
    
    def process_single_patient(self, patient_id: int) -> Tuple[str, bool, int]:
        """Process a single patient for vessel tracking"""
        
        try:
            patient_str = f"{patient_id:02d}"
            patient_dir = self.test_base_dir / f"patient{patient_str}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Get aneurysm locations for this patient
            aneurysm_locations = self.get_patient_aneurysm_locations(patient_id)
            
            if not aneurysm_locations:
                return f"Patient {patient_str}: No aneurysms found", False, 0
            
            file_count = 0
            
            # Process both MRA scans
            for mra_num in [1, 2]:
                # Load vessel data
                vessel_data, img = self.load_vessel_data(patient_id, mra_num)
                if vessel_data is None:
                    continue
                
                # Get voxel spacing
                voxel_spacing = img.header.get_zooms()[:3]
                
                # Process each aneurysm location
                for region, count in aneurysm_locations.items():
                    # Find centers for aneurysms in this region
                    centers = self.find_aneurysm_seeds(vessel_data, region)
                    
                    # Process each center (up to the doctor's count)
                    processed_centers = min(len(centers), count)
                    for i in range(processed_centers):
                        center = centers[i]
                        
                        # Perform random walk
                        vessel_region = self.random_walk_from_seed(vessel_data, center, voxel_spacing)
                        
                        if vessel_region and len(vessel_region) > 100:  # Minimum size check
                            # Extract mesh
                            mesh = self.extract_vessel_region_mesh(vessel_data, vessel_region, voxel_spacing)
                            
                            if mesh:
                                # Save mesh as STL
                                region_safe = region.replace(' ', '_').replace('(', '').replace(')', '')
                                filename = f"{patient_str}_MRA{mra_num}_{region_safe}_aneurysm_{i+1}_vessels.stl"
                                output_path = patient_dir / filename
                                
                                mesh.export(str(output_path))
                                file_count += 1
            
            return f"Patient {patient_str}: {file_count} vessel regions", True, file_count
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            return f"Patient {patient_id:02d}: Error - {str(e)}", False, 0
    
    def run_all_patients(self, max_workers: int = 16):
        """Run vessel tracking for all patients with parallel processing"""
        
        logger.info("Starting vessel tracking for all patients")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        logger.info(f"Enhanced walking size: {self.max_walking_steps} steps (doubled)")
        
        # Load patient data
        if not self.load_all_patient_data():
            logger.error("Failed to load patient data")
            return
        
        start_time = time.time()
        
        # Process patients in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_patient = {
                executor.submit(self.process_single_patient, patient_id): patient_id 
                for patient_id in self.patients_with_aneurysms
            }
            
            results = []
            completed = 0
            total = len(self.patients_with_aneurysms)
            total_files = 0
            
            for future in as_completed(future_to_patient):
                result, success, file_count = future.result()
                results.append((result, success, file_count))
                completed += 1
                total_files += file_count
                
                if success:
                    logger.info(f"✓ ({completed}/{total}) {result}")
                else:
                    logger.warning(f"✗ ({completed}/{total}) {result}")
        
        # Final summary
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n=== All Patients Processing Complete ===")
        logger.info(f"Total patients processed: {len(self.patients_with_aneurysms)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(self.patients_with_aneurysms)*100:.1f}%")
        logger.info(f"Total STL files generated: {total_files}")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Average time per patient: {elapsed_time/len(self.patients_with_aneurysms):.1f} seconds")
        
        # Check output directories
        patient_dirs = list(self.test_base_dir.glob("patient*"))
        logger.info(f"Patient directories created: {len(patient_dirs)}")
        
        return successful, failed, total_files

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vessel tracking for all patients with enhanced random walk")
    parser.add_argument("--min-area", type=float, default=0.5, 
                       help="Minimum cross-sectional area in mm² to continue tracking (default: 0.5)")
    parser.add_argument("--max-workers", type=int, default=16,
                       help="Number of CPU cores to use (default: 16)")
    
    args = parser.parse_args()
    
    tracker = AllPatientsVesselTracker(
        min_cross_section_area=args.min_area,
        max_walking_steps=20000  # Doubled walking size
    )
    tracker.run_all_patients(max_workers=args.max_workers)

if __name__ == "__main__":
    main() 