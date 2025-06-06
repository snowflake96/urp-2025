#!/usr/bin/env python3
"""
Patient 01 Vessel Tracker
- Load patient diagnosis data from Excel
- Find aneurysm locations based on doctor's diagnosis
- Perform random walk through vessels starting from aneurysms
- Calculate cross-sectional areas along vessel path
- Stop when area < threshold (parameter: default 4mm²)
- Extract nearby vessel regions and save as STL
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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VesselTracker:
    """Track vessels from aneurysm locations using random walk"""
    
    def __init__(self, min_cross_section_area: float = 4.0):
        """
        Initialize vessel tracker
        
        Args:
            min_cross_section_area: Minimum cross-sectional area in mm² to continue tracking
        """
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        self.original_smoothed_dir = self.uan_base / "original_smoothed"
        self.test_dir = self.uan_base / "test" / "patient01"
        
        self.min_cross_section_area = min_cross_section_area  # mm²
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
        """Load vessel data for given MRA number (use original if smoothed is empty)"""
        try:
            # Try smoothed first
            smoothed_filename = f"01_MRA{mra_num}_seg_smoothed.nii.gz"
            smoothed_path = self.original_smoothed_dir / smoothed_filename
            
            # Try original as backup
            original_filename = f"01_MRA{mra_num}_seg.nii.gz"
            original_path = self.uan_base / "original" / original_filename
            
            # Check smoothed file first
            if smoothed_path.exists():
                img = nib.load(smoothed_path)
                data = img.get_fdata()
                
                if np.sum(data > 0) > 0:  # If not empty, use smoothed
                    logger.info(f"Loaded smoothed {smoothed_filename}: shape {data.shape}, non-zero voxels: {np.sum(data > 0)}")
                    return data, img
                else:
                    logger.warning(f"Smoothed file {smoothed_filename} is empty, trying original")
            
            # Use original file as backup
            if original_path.exists():
                img = nib.load(original_path)
                data = img.get_fdata()
                logger.info(f"Loaded original {original_filename}: shape {data.shape}, non-zero voxels: {np.sum(data > 0)}")
                return data, img
            else:
                logger.error(f"Neither smoothed nor original file found for MRA {mra_num}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading vessel data for MRA {mra_num}: {e}")
            return None, None
    
    def find_aneurysm_seeds(self, vessel_data: np.ndarray, region: str) -> List[Tuple[int, int, int]]:
        """Find seed points for aneurysms in specific anatomical region"""
        
        # Define approximate anatomical regions based on spatial location
        # This is simplified - in practice, you'd use more sophisticated anatomical atlases
        shape = vessel_data.shape
        region_masks = {
            'ACA': {'z_range': (0.6, 1.0), 'y_range': (0.0, 0.4), 'x_range': (0.3, 0.7)},  # Superior, anterior
            'Acom': {'z_range': (0.5, 0.8), 'y_range': (0.2, 0.6), 'x_range': (0.4, 0.6)},  # Central
            'ICA (total)': {'z_range': (0.2, 0.8), 'y_range': (0.3, 0.8), 'x_range': (0.2, 0.8)},  # Wide range
            'ICA (noncavernous)': {'z_range': (0.4, 0.8), 'y_range': (0.3, 0.7), 'x_range': (0.2, 0.8)},
            'ICA (cavernous)': {'z_range': (0.2, 0.6), 'y_range': (0.4, 0.8), 'x_range': (0.2, 0.8)},
            'Pcom': {'z_range': (0.4, 0.7), 'y_range': (0.4, 0.8), 'x_range': (0.3, 0.7)},
            'BA': {'z_range': (0.3, 0.7), 'y_range': (0.6, 1.0), 'x_range': (0.4, 0.6)},  # Posterior midline
            'Other_posterior': {'z_range': (0.2, 0.6), 'y_range': (0.7, 1.0), 'x_range': (0.2, 0.8)},
            'PCA': {'z_range': (0.3, 0.6), 'y_range': (0.5, 0.9), 'x_range': (0.2, 0.8)}
        }
        
        if region not in region_masks:
            logger.warning(f"Unknown anatomical region: {region}")
            # Use middle of volume as fallback
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
        
        # New approach: Find vessel centers rather than edge detection
        seed_points = []
        
        # For ICA regions, create multiple subdivisions to find vessel centers
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
                if np.sum(sub_region > 0) > 500:  # Require significant vessel presence
                    
                    # Find the densest vessel area (center of vessel mass)
                    vessel_coords = np.where(sub_region > 0)
                    if len(vessel_coords[0]) > 0:
                        # Find center of mass
                        center_local = [
                            int(np.mean(vessel_coords[0])),
                            int(np.mean(vessel_coords[1])),
                            int(np.mean(vessel_coords[2]))
                        ]
                        
                        # Convert to global coordinates
                        center_global = [
                            center_local[0] + x_start,
                            center_local[1] + y_start,
                            center_local[2] + z_range[0]
                        ]
                        
                        # Verify this is actually a vessel voxel
                        gx, gy, gz = center_global
                        if vessel_data[gx, gy, gz] > 0:
                            seed_points.append(tuple(center_global))
                            logger.info(f"Found vessel center seed in {region} at {center_global}")
                        else:
                            # Find nearest vessel voxel
                            best_distance = float('inf')
                            best_point = None
                            for dx in range(-5, 6):
                                for dy in range(-5, 6):
                                    for dz in range(-5, 6):
                                        check_x, check_y, check_z = gx + dx, gy + dy, gz + dz
                                        if (0 <= check_x < vessel_data.shape[0] and
                                            0 <= check_y < vessel_data.shape[1] and
                                            0 <= check_z < vessel_data.shape[2]):
                                            if vessel_data[check_x, check_y, check_z] > 0:
                                                distance = dx*dx + dy*dy + dz*dz
                                                if distance < best_distance:
                                                    best_distance = distance
                                                    best_point = (check_x, check_y, check_z)
                            
                            if best_point:
                                seed_points.append(best_point)
                                logger.info(f"Found nearest vessel seed in {region} at {best_point}")
        else:
            # For other regions, just use the center of vessel mass
            vessel_coords = np.where(region_vessels > 0)
            if len(vessel_coords[0]) > 0:
                center_local = [
                    int(np.mean(vessel_coords[0])),
                    int(np.mean(vessel_coords[1])),
                    int(np.mean(vessel_coords[2]))
                ]
                
                center_global = [
                    center_local[0] + x_range[0],
                    center_local[1] + y_range[0],
                    center_local[2] + z_range[0]
                ]
                
                # Verify and find nearest vessel voxel if needed
                gx, gy, gz = center_global
                if vessel_data[gx, gy, gz] > 0:
                    seed_points.append(tuple(center_global))
                    logger.info(f"Found vessel center seed in {region} at {center_global}")
                else:
                    # Find nearest vessel voxel
                    vessel_coords_global = np.where(vessel_data > 0)
                    if len(vessel_coords_global[0]) > 0:
                        distances = np.sqrt(
                            (vessel_coords_global[0] - gx)**2 + 
                            (vessel_coords_global[1] - gy)**2 + 
                            (vessel_coords_global[2] - gz)**2
                        )
                        nearest_idx = np.argmin(distances)
                        nearest_point = (
                            vessel_coords_global[0][nearest_idx], 
                            vessel_coords_global[1][nearest_idx], 
                            vessel_coords_global[2][nearest_idx]
                        )
                        seed_points.append(nearest_point)
                        logger.info(f"Found nearest vessel seed in {region} at {nearest_point}")
        
        return seed_points
    
    def calculate_cross_sectional_area(self, vessel_data: np.ndarray, point: Tuple[int, int, int], 
                                     direction: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> float:
        """Calculate cross-sectional area at a point along a direction"""
        
        x, y, z = point
        
        # Create a small local region around the point
        region_size = 3  # Smaller region for more sensitive area calculation
        x_start, x_end = max(0, x-region_size), min(vessel_data.shape[0], x+region_size+1)
        y_start, y_end = max(0, y-region_size), min(vessel_data.shape[1], y+region_size+1)
        z_start, z_end = max(0, z-region_size), min(vessel_data.shape[2], z+region_size+1)
        
        local_vessel = vessel_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        if np.sum(local_vessel > 0) == 0:
            return 0.0
        
        # More accurate cross-sectional area calculation
        vessel_voxels = np.sum(local_vessel > 0)
        
        # Convert to physical area using voxel spacing
        voxel_area = voxel_spacing[0] * voxel_spacing[1]  # mm²
        cross_sectional_area = vessel_voxels * voxel_area  # mm²
        
        return cross_sectional_area
    
    def random_walk_from_seed(self, vessel_data: np.ndarray, seed: Tuple[int, int, int], 
                            voxel_spacing: Tuple[float, float, float], max_steps: int = 10000) -> Set[Tuple[int, int, int]]:
        """Enhanced random walk that follows vessel paths from seed point"""
        
        visited = set()
        vessel_region = set()
        
        # Use a priority queue to prioritize vessel-like paths
        from collections import deque
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
        max_consecutive_small = 50  # Much more lenient - allow many small areas
        
        while to_visit and step < max_steps:
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
            
            # Calculate cross-sectional area at this point
            direction = np.array([1, 0, 0])  # Simple direction
            area = self.calculate_cross_sectional_area(vessel_data, current, direction, voxel_spacing)
            
            # Very lenient area threshold - only stop if very small for many consecutive steps
            if area < self.min_cross_section_area:
                consecutive_small_areas += 1
                if consecutive_small_areas > max_consecutive_small:
                    logger.debug(f"Stopping walk at {current}: {consecutive_small_areas} consecutive small areas")
                    continue
            else:
                consecutive_small_areas = 0  # Reset counter
            
            # Add to vessel region
            vessel_region.add(current)
            
            # Simple neighbor addition - just follow any vessel voxels
            for dx, dy, dz in neighbors:
                neighbor = (x + dx, y + dy, z + dz)
                
                if (neighbor not in visited and 
                    0 <= neighbor[0] < vessel_data.shape[0] and
                    0 <= neighbor[1] < vessel_data.shape[1] and
                    0 <= neighbor[2] < vessel_data.shape[2]):
                    
                    # Check if neighbor is a vessel voxel
                    nx, ny, nz = neighbor
                    if vessel_data[nx, ny, nz] > 0:
                        to_visit.append(neighbor)
            
            step += 1
        
        logger.info(f"Enhanced random walk completed: {len(vessel_region)} voxels in vessel region")
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
            # Apply marching cubes to create surface mesh
            vertices, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=voxel_spacing
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Clean up mesh
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            mesh.remove_duplicate_faces()
            
            if mesh.is_empty:
                return None
                
            logger.info(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            return None
    
    def process_patient_01(self):
        """Process patient 01 for vessel tracking"""
        
        logger.info("Starting vessel tracking for Patient 01")
        
        # Load patient data
        if not self.load_patient_data():
            logger.error("Failed to load patient data")
            return
        
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
            
            aneurysm_count = 0
            
            # Process each aneurysm location
            for region, count in self.aneurysm_locations.items():
                logger.info(f"\nProcessing {count} aneurysm(s) in {region}")
                
                # Find seed points for aneurysms in this region
                seed_points = self.find_aneurysm_seeds(vessel_data, region)
                
                # Process each seed point (up to the doctor's count)
                processed_seeds = min(len(seed_points), count)
                for i in range(processed_seeds):
                    seed = seed_points[i]
                    aneurysm_count += 1
                    
                    logger.info(f"Tracking vessels from aneurysm {aneurysm_count} in {region} at {seed}")
                    
                    # Perform random walk
                    vessel_region = self.random_walk_from_seed(vessel_data, seed, voxel_spacing)
                    
                    if vessel_region:
                        # Extract mesh
                        mesh = self.extract_vessel_region_mesh(vessel_data, vessel_region, voxel_spacing)
                        
                        if mesh:
                            # Save mesh as STL
                            region_safe = region.replace(' ', '_').replace('(', '').replace(')', '')
                            filename = f"01_MRA{mra_num}_{region_safe}_aneurysm_{i+1}_vessels.stl"
                            output_path = self.test_dir / filename
                            
                            mesh.export(str(output_path))
                            logger.info(f"Saved vessel region: {output_path}")
                            
                            # Log mesh stats
                            logger.info(f"Mesh volume: {mesh.volume:.2f} mm³")
                            logger.info(f"Mesh surface area: {mesh.area:.2f} mm²")
                        else:
                            logger.warning(f"Failed to create mesh for aneurysm {aneurysm_count}")
                    else:
                        logger.warning(f"No vessel region found for aneurysm {aneurysm_count}")
        
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
    
    parser = argparse.ArgumentParser(description="Track vessels from aneurysm locations for Patient 01")
    parser.add_argument("--min-area", type=float, default=4.0, 
                       help="Minimum cross-sectional area in mm² to continue tracking (default: 4.0)")
    
    args = parser.parse_args()
    
    tracker = VesselTracker(min_cross_section_area=args.min_area)
    tracker.process_patient_01()

if __name__ == "__main__":
    main() 