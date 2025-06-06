#!/usr/bin/env python3
"""
Comprehensive UAN (Unruptured ANeurysm) Processing Pipeline
Processes segmentation data through the complete pipeline:
1. Largest island extraction
2. Area separation 
3. Aneurysm detection with doctor annotation correlation
4. Cropping for stress analysis
"""

import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from scipy import ndimage, spatial
from skimage import measure, morphology, segmentation
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import trimesh

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UANProcessor:
    """Comprehensive processor for unruptured aneurysm analysis"""
    
    def __init__(self):
        # Data paths
        self.segmentation_path = Path("/home/jiwoo/urp/data/segmentation/aneu/UAN_processed/Output")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        self.output_base = Path("/home/jiwoo/urp/data/uan")
        
        # Output directories
        self.largest_island_dir = self.output_base / "largest_island"
        self.area_separation_dir = self.output_base / "area_separation"
        self.aneurysm_detection_dir = self.output_base / "aneurysm_detection"
        self.cropped_dir = self.output_base / "cropped"
        
        # Create output directories
        for output_dir in [self.largest_island_dir, self.area_separation_dir, 
                          self.aneurysm_detection_dir, self.cropped_dir]:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Anatomical regions of interest
        self.regions = ['ACA', 'Acom', 'ICA (total)', 'Pcom', 'BA', 'Other_posterior', 'PCA']
        
        # Load doctor annotations
        self.load_doctor_annotations()
        
        # Get list of segmentation files
        self.get_segmentation_files()
        
    def load_doctor_annotations(self):
        """Load doctor annotations from Excel file"""
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded Excel file with {len(df)} rows")
        
            # Create patient annotations dictionary
            self.patient_annotations = {}
        
            for _, row in df.iterrows():
                try:
                    patient_id = int(row['VintID'])  # Convert to int to match filename parsing
        
                    # Extract aneurysm counts for each region
            annotations = {}
            for region in self.regions:
                        if region in row and pd.notna(row[region]):
                            annotations[region] = int(row[region])
                else:
                    annotations[region] = 0
                    
                    self.patient_annotations[patient_id] = annotations
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping row due to invalid data: {e}")
                    continue
            
            logger.info(f"Loaded annotations for {len(self.patient_annotations)} patients")
            
        except Exception as e:
            logger.error(f"Error loading doctor annotations: {e}")
            self.patient_annotations = {}
    
    def get_segmentation_files(self):
        """Get list of all segmentation files"""
        seg_files = list(self.segmentation_path.glob("*_seg.nii.gz"))
        
        self.segmentation_files = []
        for seg_file in seg_files:
            # Parse filename: e.g., "10_MRA1_seg.nii.gz"
            filename = seg_file.stem.replace('_seg', '')  # Remove _seg
            if filename.endswith('.nii'):
                filename = filename[:-4]  # Remove .nii
                
            parts = filename.split('_')
            if len(parts) == 2 and parts[1].startswith('MRA'):
                try:
                    patient_num = int(parts[0])
                    mra_num = int(parts[1][-1])  # Extract number from MRA1, MRA2
                    self.segmentation_files.append({
                        'file_path': seg_file,
                        'patient_num': patient_num,
                        'mra_num': mra_num,
                        'filename': filename
                    })
                except ValueError:
                    continue
        
        logger.info(f"Found {len(self.segmentation_files)} segmentation files")
    
    def nifti_to_stl(self, mask_data: np.ndarray, output_path: Path, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Convert binary mask to STL file using marching cubes"""
        try:
            if np.sum(mask_data) == 0:
                logger.warning(f"Empty mask, skipping STL generation for {output_path}")
                return False
            
            # Apply marching cubes to extract surface
            vertices, faces, normals, values = measure.marching_cubes(
                mask_data, level=0.5, spacing=spacing
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Clean up mesh
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Export to STL
            mesh.export(str(output_path))
            logger.info(f"Created STL file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating STL file {output_path}: {e}")
            return False
    
    def extract_largest_island(self, seg_data: np.ndarray) -> np.ndarray:
        """Extract the largest connected component (island) from segmentation"""
        # Binarize the segmentation
        binary_mask = seg_data > 0
        
        # Find connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        if num_labels == 0:
            logger.warning("No connected components found")
            return np.zeros_like(seg_data)
        
        # Find the largest component
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))
        largest_component_label = np.argmax(component_sizes) + 1
        
        # Create mask for largest component
        largest_island = (labeled_mask == largest_component_label).astype(np.uint8)
        
        logger.info(f"Extracted largest island with {np.sum(largest_island)} voxels from {num_labels} components")
        return largest_island
    
    def process_largest_island(self, file_info: dict) -> Tuple[str, bool]:
        """Process a single file for largest island extraction"""
        try:
            patient_num = file_info['patient_num']
            mra_num = file_info['mra_num']
            seg_file = file_info['file_path']
            
            logger.info(f"Processing largest island for patient {patient_num}, MRA {mra_num}")
            
            # Load segmentation
            seg_img = nib.load(seg_file)
            seg_data = seg_img.get_fdata()
            
            # Extract largest island
            largest_island = self.extract_largest_island(seg_data)
            
            # Save largest island NIfTI
            output_filename = f"{patient_num}_{mra_num}_largest_island.nii.gz"
            output_path = self.largest_island_dir / output_filename
            
            largest_island_img = nib.Nifti1Image(largest_island, seg_img.affine, seg_img.header)
            nib.save(largest_island_img, output_path)
            
            # Create STL file for preview
            stl_filename = f"{patient_num}_{mra_num}_largest_island.stl"
            stl_path = self.largest_island_dir / stl_filename
            spacing = seg_img.header.get_zooms()[:3]
            self.nifti_to_stl(largest_island, stl_path, spacing)
            
            return f"Patient {patient_num} MRA {mra_num}", True
            
        except Exception as e:
            logger.error(f"Error processing {file_info}: {e}")
            return f"Patient {file_info.get('patient_num', 'unknown')} MRA {file_info.get('mra_num', 'unknown')}", False
    
    def analyze_anatomical_regions(self, mask_data: np.ndarray, patient_num: int) -> Dict[str, Any]:
        """Analyze anatomical regions in the vessel mask"""
        
        # Get patient annotations
        annotations = self.patient_annotations.get(patient_num, {})
        
        # Simple region analysis based on spatial location
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        
        shape = mask_data.shape
        regions_analysis = {}
        
        # Define rough anatomical locations (simplified)
        region_locations = {
            'ACA': {'z_range': (0.6, 1.0), 'y_range': (0.0, 0.4)},  # Superior, anterior
            'Acom': {'z_range': (0.5, 0.8), 'y_range': (0.2, 0.6)},  # Mid-superior, central
            'ICA (total)': {'z_range': (0.2, 0.8), 'y_range': (0.3, 0.8)},  # Wide range
            'Pcom': {'z_range': (0.4, 0.7), 'y_range': (0.4, 0.8)},  # Mid-level, posterior
            'BA': {'z_range': (0.3, 0.7), 'y_range': (0.6, 1.0)},  # Mid-level, posterior
            'Other_posterior': {'z_range': (0.2, 0.6), 'y_range': (0.7, 1.0)},  # Lower, posterior
            'PCA': {'z_range': (0.3, 0.6), 'y_range': (0.5, 0.9)}  # Mid-lower, posterior
        }
        
        # Analyze each region
        for region, location in region_locations.items():
            # Get region mask based on spatial location
            z_min, z_max = int(location['z_range'][0] * shape[2]), int(location['z_range'][1] * shape[2])
            y_min, y_max = int(location['y_range'][0] * shape[1]), int(location['y_range'][1] * shape[1])
            
            region_mask = np.zeros_like(mask_data)
            region_mask[:, y_min:y_max, z_min:z_max] = mask_data[:, y_min:y_max, z_min:z_max]
            
            # Calculate region properties
            region_volume = np.sum(region_mask > 0)
            
            # Get doctor annotation for this region
            doctor_count = annotations.get(region, 0)
            
            regions_analysis[region] = {
                'volume_voxels': int(region_volume),
                'doctor_annotation': int(doctor_count),
                'has_aneurysm': doctor_count > 0,
                'spatial_bounds': {
                    'z_range': (z_min, z_max),
                    'y_range': (y_min, y_max)
                }
            }
        
        return regions_analysis
    
    def process_area_separation(self, file_info: dict) -> Tuple[str, bool]:
        """Process area separation for a single patient scan"""
        try:
            patient_num = file_info['patient_num']
            mra_num = file_info['mra_num']
            
            logger.info(f"Processing area separation for patient {patient_num}, MRA {mra_num}")
            
            # Load largest island
            island_filename = f"{patient_num}_{mra_num}_largest_island.nii.gz"
            island_path = self.largest_island_dir / island_filename
            
            if not island_path.exists():
                logger.warning(f"Largest island not found: {island_path}")
                return f"Patient {patient_num} MRA {mra_num}", False
            
            island_img = nib.load(island_path)
            island_data = island_img.get_fdata()
            
            # Analyze anatomical regions
            regions_analysis = self.analyze_anatomical_regions(island_data, patient_num)
            
            # Save area separation data as pickle (best for complex nested data)
            output_filename = f"{patient_num}_{mra_num}_area_separation.pkl"
            output_path = self.area_separation_dir / output_filename
            
            separation_data = {
                'patient_num': patient_num,
                'mra_num': mra_num,
                'regions_analysis': regions_analysis,
                'total_volume': int(np.sum(island_data > 0)),
                'image_shape': island_data.shape,
                'voxel_spacing': island_img.header.get_zooms()[:3]
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(separation_data, f)
            
            return f"Patient {patient_num} MRA {mra_num}", True
            
        except Exception as e:
            logger.error(f"Error processing area separation {file_info}: {e}")
            return f"Patient {file_info.get('patient_num', 'unknown')} MRA {file_info.get('mra_num', 'unknown')}", False
    
    def detect_aneurysms_in_region(self, mask_data: np.ndarray, region_mask: np.ndarray) -> List[Dict]:
        """Detect aneurysms in a specific anatomical region"""
        
        # Apply region mask
        region_vessels = mask_data * region_mask
        
        if np.sum(region_vessels) == 0:
            return []
        
        # Simple aneurysm detection based on local dilation/expansion
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        
        # Apply morphological operations to find dilated regions
        structure = morphology.ball(3)
        dilated = morphology.binary_dilation(region_vessels > 0, structure)
        eroded = morphology.binary_erosion(region_vessels > 0, structure)
        
        # Find regions that are present in dilated but reduced in eroded (potential aneurysms)
        aneurysm_candidates = dilated & ~eroded
        
        # Find connected components of candidates
        labeled_candidates, num_candidates = ndimage.label(aneurysm_candidates)
        
        detected_aneurysms = []
        
        for i in range(1, num_candidates + 1):
            candidate_mask = labeled_candidates == i
            candidate_volume = np.sum(candidate_mask)
            
            # Filter by size (aneurysms should have minimum volume)
            if candidate_volume < 10:  # Minimum volume threshold
                continue
            
            # Get centroid
            coords = np.where(candidate_mask)
            centroid = [float(np.mean(c)) for c in coords]
            
            # Get bounding box
            min_coords = [int(np.min(c)) for c in coords]
            max_coords = [int(np.max(c)) for c in coords]
            
            detected_aneurysms.append({
                'centroid': centroid,
                'volume_voxels': int(candidate_volume),
                'bounding_box': {
                    'min': min_coords,
                    'max': max_coords
                }
            })
        
        return detected_aneurysms
    
    def process_aneurysm_detection(self, file_info: dict) -> Tuple[str, bool]:
        """Process aneurysm detection for a single patient scan"""
        try:
            patient_num = file_info['patient_num']
            mra_num = file_info['mra_num']
            
            logger.info(f"Processing aneurysm detection for patient {patient_num}, MRA {mra_num}")
            
            # Load area separation data
            separation_filename = f"{patient_num}_{mra_num}_area_separation.pkl"
            separation_path = self.area_separation_dir / separation_filename
            
            if not separation_path.exists():
                logger.warning(f"Area separation not found: {separation_path}")
                return f"Patient {patient_num} MRA {mra_num}", False
            
            with open(separation_path, 'rb') as f:
                separation_data = pickle.load(f)
            
            # Load largest island
            island_filename = f"{patient_num}_{mra_num}_largest_island.nii.gz"
            island_path = self.largest_island_dir / island_filename
            island_img = nib.load(island_path)
            island_data = island_img.get_fdata()
            
            # Detect aneurysms in each region
            detection_results = {
                'patient_num': patient_num,
                'mra_num': mra_num,
                'regions': {}
            }
            
            for region, region_info in separation_data['regions_analysis'].items():
                # Create region mask
                bounds = region_info['spatial_bounds']
                region_mask = np.zeros_like(island_data)
                region_mask[:, bounds['y_range'][0]:bounds['y_range'][1], 
                           bounds['z_range'][0]:bounds['z_range'][1]] = 1
                
                # Detect aneurysms in this region
                detected = self.detect_aneurysms_in_region(island_data, region_mask)
                
                detection_results['regions'][region] = {
                    'doctor_annotation': region_info['doctor_annotation'],
                    'has_doctor_confirmed_aneurysm': region_info['has_aneurysm'],
                    'detected_aneurysms': detected,
                    'detection_count': len(detected),
                    'confirmed_aneurysms': []  # Will be filled based on doctor annotations
                }
                
                # Mark confirmed aneurysms based on doctor annotations
                if region_info['has_aneurysm'] and detected:
                    # If doctor says there are aneurysms and we detected some,
                    # mark the largest detected ones as confirmed
                    num_confirmed = min(region_info['doctor_annotation'], len(detected))
                    # Sort by volume and take the largest ones
                    sorted_detected = sorted(detected, key=lambda x: x['volume_voxels'], reverse=True)
                    detection_results['regions'][region]['confirmed_aneurysms'] = sorted_detected[:num_confirmed]
            
            # Save detection results
            output_filename = f"{patient_num}_{mra_num}_aneurysm_detection.json"
            output_path = self.aneurysm_detection_dir / output_filename
            
            with open(output_path, 'w') as f:
                json.dump(detection_results, f, indent=2)
            
            return f"Patient {patient_num} MRA {mra_num}", True
            
        except Exception as e:
            logger.error(f"Error processing aneurysm detection {file_info}: {e}")
            return f"Patient {file_info.get('patient_num', 'unknown')} MRA {file_info.get('mra_num', 'unknown')}", False
    
    def crop_aneurysm_region(self, mask_data: np.ndarray, aneurysm_info: Dict, 
                           crop_size: Tuple[int, int, int] = (80, 80, 60)) -> np.ndarray:
        """Crop region around aneurysm with sufficient boundary for analysis"""
        
        centroid = aneurysm_info['centroid']
        x_center, y_center, z_center = [int(c) for c in centroid]
        
        crop_x, crop_y, crop_z = crop_size
        
        # Calculate crop bounds
        x_start = max(0, x_center - crop_x // 2)
        x_end = min(mask_data.shape[0], x_start + crop_x)
        x_start = max(0, x_end - crop_x)
        
        y_start = max(0, y_center - crop_y // 2)
        y_end = min(mask_data.shape[1], y_start + crop_y)
        y_start = max(0, y_end - crop_y)
        
        z_start = max(0, z_center - crop_z // 2)
        z_end = min(mask_data.shape[2], z_start + crop_z)
        z_start = max(0, z_end - crop_z)
        
        # Crop the data
        cropped = mask_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Ensure only the largest island containing the aneurysm remains
        cropped_largest = self.extract_largest_island(cropped)
        
        return cropped_largest
    
    def process_cropping(self, file_info: dict) -> Tuple[str, bool]:
        """Process cropping for stress analysis for a single patient scan"""
        try:
            patient_num = file_info['patient_num']
            mra_num = file_info['mra_num']
            
            logger.info(f"Processing cropping for patient {patient_num}, MRA {mra_num}")
            
            # Load detection results
            detection_filename = f"{patient_num}_{mra_num}_aneurysm_detection.json"
            detection_path = self.aneurysm_detection_dir / detection_filename
            
            if not detection_path.exists():
                logger.warning(f"Detection results not found: {detection_path}")
                return f"Patient {patient_num} MRA {mra_num}", False
            
            with open(detection_path, 'r') as f:
                detection_results = json.load(f)
            
            # Load largest island
            island_filename = f"{patient_num}_{mra_num}_largest_island.nii.gz"
            island_path = self.largest_island_dir / island_filename
            island_img = nib.load(island_path)
            island_data = island_img.get_fdata()
            
            # Create output directory for this patient/scan
            patient_dir = self.cropped_dir / str(patient_num)
            scan_dir = patient_dir / str(mra_num)
            scan_dir.mkdir(parents=True, exist_ok=True)
            
            crop_count = 0
            
            # Process each region with confirmed aneurysms
            for region, region_data in detection_results['regions'].items():
                confirmed_aneurysms = region_data.get('confirmed_aneurysms', [])
                
                if not confirmed_aneurysms:
                    continue
                
                # Crop each confirmed aneurysm
                for i, aneurysm in enumerate(confirmed_aneurysms):
                    cropped_data = self.crop_aneurysm_region(island_data, aneurysm)
                    
                    # Save cropped region NIfTI
                    region_safe = region.replace(' ', '_').replace('(', '').replace(')', '')
                    crop_filename = f"{region_safe}_aneurysm_{i+1}.nii.gz"
                    crop_path = scan_dir / crop_filename
                    
                    cropped_img = nib.Nifti1Image(cropped_data, island_img.affine, island_img.header)
                    nib.save(cropped_img, crop_path)
                    
                    # Create STL file for preview
                    stl_filename = f"{region_safe}_aneurysm_{i+1}.stl"
                    stl_path = scan_dir / stl_filename
                    spacing = island_img.header.get_zooms()[:3]
                    self.nifti_to_stl(cropped_data, stl_path, spacing)
                    
                    crop_count += 1
                    logger.info(f"Saved cropped aneurysm: {crop_filename}")
            
            return f"Patient {patient_num} MRA {mra_num} ({crop_count} crops)", True
            
        except Exception as e:
            logger.error(f"Error processing cropping {file_info}: {e}")
            return f"Patient {file_info.get('patient_num', 'unknown')} MRA {file_info.get('mra_num', 'unknown')}", False
    
    def run_pipeline(self, max_workers: int = 4, limit: Optional[int] = None):
        """Run the complete UAN processing pipeline"""
        
        files_to_process = self.segmentation_files
        if limit:
            files_to_process = files_to_process[:limit]
            logger.info(f"Limited processing to first {limit} files for testing")
        
        logger.info(f"Starting UAN processing pipeline for {len(files_to_process)} files")
        
        # Step 1: Extract largest islands
        logger.info("=== Step 1: Extracting largest islands ===")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_largest_island, file_info): file_info 
                             for file_info in files_to_process}
            
            island_results = []
            for future in as_completed(future_to_file):
                result, success = future.result()
                island_results.append((result, success))
                if success:
                    logger.info(f"✓ {result}")
                else:
                    logger.error(f"✗ {result}")
        
        step1_time = time.time() - start_time
        successful_islands = sum(1 for _, success in island_results if success)
        logger.info(f"Step 1 completed: {successful_islands}/{len(files_to_process)} successful in {step1_time:.1f}s")
        
        # Step 2: Area separation
        logger.info("=== Step 2: Area separation analysis ===")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_area_separation, file_info): file_info 
                             for file_info in files_to_process}
            
            separation_results = []
            for future in as_completed(future_to_file):
                result, success = future.result()
                separation_results.append((result, success))
                if success:
                    logger.info(f"✓ {result}")
                else:
                    logger.error(f"✗ {result}")
        
        step2_time = time.time() - start_time
        successful_separations = sum(1 for _, success in separation_results if success)
        logger.info(f"Step 2 completed: {successful_separations}/{len(files_to_process)} successful in {step2_time:.1f}s")
        
        # Step 3: Aneurysm detection
        logger.info("=== Step 3: Aneurysm detection with doctor correlation ===")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_aneurysm_detection, file_info): file_info 
                             for file_info in files_to_process}
            
            detection_results = []
            for future in as_completed(future_to_file):
                result, success = future.result()
                detection_results.append((result, success))
                if success:
                    logger.info(f"✓ {result}")
                else:
                    logger.error(f"✗ {result}")
        
        step3_time = time.time() - start_time
        successful_detections = sum(1 for _, success in detection_results if success)
        logger.info(f"Step 3 completed: {successful_detections}/{len(files_to_process)} successful in {step3_time:.1f}s")
        
        # Step 4: Cropping for stress analysis
        logger.info("=== Step 4: Cropping for stress analysis ===")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_cropping, file_info): file_info 
                             for file_info in files_to_process}
            
            cropping_results = []
            for future in as_completed(future_to_file):
                result, success = future.result()
                cropping_results.append((result, success))
                if success:
                    logger.info(f"✓ {result}")
                else:
                    logger.error(f"✗ {result}")
        
        step4_time = time.time() - start_time
        successful_crops = sum(1 for _, success in cropping_results if success)
        logger.info(f"Step 4 completed: {successful_crops}/{len(files_to_process)} successful in {step4_time:.1f}s")
        
        # Final summary
        total_time = step1_time + step2_time + step3_time + step4_time
        logger.info(f"\n=== UAN Pipeline Complete ===")
        logger.info(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Files processed: {len(files_to_process)}")
        logger.info(f"Step 1 (Largest islands): {successful_islands}/{len(files_to_process)}")
        logger.info(f"Step 2 (Area separation): {successful_separations}/{len(files_to_process)}")
        logger.info(f"Step 3 (Aneurysm detection): {successful_detections}/{len(files_to_process)}")
        logger.info(f"Step 4 (Cropping): {successful_crops}/{len(files_to_process)}")
        logger.info(f"Output directories:")
        logger.info(f"  - Largest islands: {self.largest_island_dir}")
        logger.info(f"  - Area separation: {self.area_separation_dir}")
        logger.info(f"  - Aneurysm detection: {self.aneurysm_detection_dir}")
        logger.info(f"  - Cropped regions: {self.cropped_dir}")

def main():
    """Main function to run the UAN processing pipeline"""
    
    # Create processor
    processor = UANProcessor()
    
    # Process ALL patients with 8 cores
    logger.info("Starting UAN processing pipeline for ALL 168 patients...")
    processor.run_pipeline(max_workers=8, limit=None)  # Process all files with 8 cores

if __name__ == "__main__":
    main() 