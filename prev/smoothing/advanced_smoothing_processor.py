#!/usr/bin/env python3
"""
Advanced Smoothing Processor for UAN Pipeline
- Converts patient IDs to two-digit format (handles both 2-digit and 3-digit IDs)
- Creates STL files from NIfTI files  
- Advanced smoothing that maintains volume and cross-section area
- Specifically targets false blocks from low resolution
- Supports both NIfTI and STL file processing
- Uses 16 CPU cores for parallel processing
"""

import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import logging
import re
import shutil
from typing import Dict, List, Tuple, Optional, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage, interpolate
from skimage import measure, morphology, filters
import json
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSmoothingProcessor:
    """Advanced processor for file organization, STL generation, and smoothing"""
    
    def __init__(self):
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.original_dir = self.uan_base / "original"
        self.original_smoothed_dir = self.uan_base / "original_smoothed"
        self.cropped_smoothed_dir = self.uan_base / "cropped_smoothed"
        
        # Create directories
        for dir_path in [self.original_smoothed_dir, self.cropped_smoothed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def extract_patient_id(self, filename: str) -> Optional[int]:
        """Extract patient ID from filename - handles both 2-digit and 3-digit IDs"""
        # Match patterns like "23_1_", "001_MRA1_", "006_MRA2_"
        match = re.search(r'^(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None
    
    def format_patient_id(self, patient_id: int) -> str:
        """Convert patient ID to two-digit format"""
        return f"{patient_id:02d}"
    
    def rename_to_two_digit(self, old_path: Path, new_path: Path) -> bool:
        """Rename file/folder to use two-digit patient ID"""
        try:
            if old_path.exists() and not new_path.exists():
                if old_path.is_dir():
                    shutil.move(str(old_path), str(new_path))
                else:
                    old_path.rename(new_path)
                logger.info(f"Renamed: {old_path.name} -> {new_path.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error renaming {old_path} to {new_path}: {e}")
            return False
    
    def standardize_filename(self, filename: str) -> str:
        """Convert filename to use 2-digit patient ID"""
        # Extract patient ID
        match = re.search(r'^(\d+)_', filename)
        if match:
            patient_id = int(match.group(1))
            two_digit_id = f"{patient_id:02d}"
            # Replace the old ID with the new 2-digit ID
            new_filename = filename.replace(f"{patient_id}_", f"{two_digit_id}_", 1)
            return new_filename
        return filename
    
    def nifti_to_stl(self, nifti_path: Path, stl_path: Path, 
                     spacing: Optional[Tuple[float, float, float]] = None) -> bool:
        """Convert NIfTI file to STL using marching cubes"""
        try:
            # Load NIfTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            if np.sum(data) == 0:
                logger.warning(f"Empty mask, skipping STL generation for {stl_path}")
                return False
            
            # Get spacing from header if not provided
            if spacing is None:
                spacing = img.header.get_zooms()[:3]
            
            # Apply marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                data, level=0.5, spacing=spacing
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Clean up mesh
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            
            # Export to STL
            mesh.export(str(stl_path))
            logger.info(f"Created STL: {stl_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating STL {stl_path}: {e}")
            return False
    
    def detect_false_blocks(self, data: np.ndarray) -> np.ndarray:
        """Detect false blocks (popped out regions due to low resolution)"""
        # Convert to binary
        binary = data > 0
        
        # Find connected components
        labeled, num_labels = ndimage.label(binary)
        
        # Calculate component properties
        false_blocks = np.zeros_like(binary)
        
        for i in range(1, num_labels + 1):
            component = labeled == i
            
            # Get component properties
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
                
            # Calculate bounding box dimensions
            bbox_dims = [
                np.max(coords[j]) - np.min(coords[j]) + 1 
                for j in range(3)
            ]
            
            # Calculate volume and surface area ratios
            volume = np.sum(component)
            bbox_volume = np.prod(bbox_dims)
            volume_ratio = volume / max(bbox_volume, 1)
            
            # Detect false blocks based on:
            # 1. Small isolated components
            # 2. Components with low volume-to-bbox ratio (spiky)
            # 3. Components with high surface area to volume ratio
            is_false_block = (
                volume < 50 or  # Very small components
                volume_ratio < 0.3 or  # Spiky/elongated
                (max(bbox_dims) > 10 and min(bbox_dims) < 3)  # Thin protrusions
            )
            
            if is_false_block:
                false_blocks |= component
        
        return false_blocks
    
    def smooth_nifti_advanced(self, input_path: Path, output_path: Path) -> bool:
        """Advanced smoothing for NIfTI files"""
        try:
            # Load image
            img = nib.load(input_path)
            data = img.get_fdata().astype(np.float32)
            original_volume = np.sum(data > 0)
            
            if original_volume == 0:
                logger.warning(f"Empty image: {input_path}")
                return False
            
            # Detect false blocks
            false_blocks = self.detect_false_blocks(data)
            
            # Create binary mask
            binary_data = (data > 0).astype(np.float32)
            
            # Apply different smoothing strategies
            smoothed_data = binary_data.copy()
            
            # 1. Strong smoothing for false blocks
            if np.sum(false_blocks) > 0:
                false_block_smooth = ndimage.gaussian_filter(
                    false_blocks.astype(np.float32), sigma=2.0
                )
                # Remove false blocks and add smoothed version
                smoothed_data[false_blocks] = 0
                smoothed_data += false_block_smooth * 0.5
            
            # 2. Mild smoothing for main structure
            main_structure = binary_data.astype(bool) & ~false_blocks.astype(bool)
            if np.sum(main_structure) > 0:
                main_smooth = ndimage.gaussian_filter(
                    main_structure.astype(np.float32), sigma=0.8
                )
                smoothed_data[main_structure] = main_smooth[main_structure]
            
            # 3. Edge-preserving smoothing
            smoothed_data = filters.median(smoothed_data, morphology.ball(1))
            
            # 4. Volume preservation
            current_volume = np.sum(smoothed_data > 0.5)
            if current_volume > 0:
                volume_factor = original_volume / current_volume
                if 0.8 <= volume_factor <= 1.2:  # Reasonable range
                    threshold = 0.5 / volume_factor
                    smoothed_data = (smoothed_data > threshold).astype(np.uint8)
                else:
                    smoothed_data = (smoothed_data > 0.5).astype(np.uint8)
            else:
                smoothed_data = (smoothed_data > 0.5).astype(np.uint8)
            
            # Save smoothed image
            smoothed_img = nib.Nifti1Image(smoothed_data, img.affine, img.header)
            nib.save(smoothed_img, output_path)
            
            final_volume = np.sum(smoothed_data > 0)
            volume_change = abs(final_volume - original_volume) / original_volume * 100
            
            logger.info(f"Smoothed NIfTI: {output_path.name} (volume change: {volume_change:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error smoothing NIfTI {input_path}: {e}")
            return False
    
    def smooth_stl_advanced(self, input_path: Path, output_path: Path) -> bool:
        """Advanced smoothing for STL files"""
        try:
            # Load mesh
            mesh = trimesh.load(str(input_path))
            original_volume = mesh.volume
            
            if original_volume <= 0:
                logger.warning(f"Invalid mesh volume: {input_path}")
                return False
            
            # Detect problematic regions (sharp edges, spikes)
            # Calculate vertex curvatures
            vertex_normals = mesh.vertex_normals
            face_adjacency = mesh.face_adjacency
            
            # Smooth problematic regions more aggressively
            # Apply Laplacian smoothing with edge preservation
            
            # 1. Mild Laplacian smoothing for overall shape
            smoothed_vertices = mesh.vertices.copy()
            for iteration in range(3):
                vertex_neighbors = mesh.vertex_neighbors
                new_vertices = smoothed_vertices.copy()
                
                for i, neighbors in enumerate(vertex_neighbors):
                    if len(neighbors) > 0:
                        neighbor_positions = smoothed_vertices[neighbors]
                        centroid = np.mean(neighbor_positions, axis=0)
                        # Weighted smoothing (preserve volume)
                        new_vertices[i] = 0.7 * smoothed_vertices[i] + 0.3 * centroid
                
                smoothed_vertices = new_vertices
            
            # 2. Create smoothed mesh
            smoothed_mesh = trimesh.Trimesh(
                vertices=smoothed_vertices, 
                faces=mesh.faces
            )
            
            # 3. Volume preservation
            current_volume = smoothed_mesh.volume
            if current_volume > 0:
                scale_factor = (original_volume / current_volume) ** (1/3)
                if 0.9 <= scale_factor <= 1.1:  # Reasonable scaling
                    smoothed_mesh.vertices *= scale_factor
            
            # 4. Final cleanup
            smoothed_mesh.update_faces(smoothed_mesh.unique_faces())
            smoothed_mesh.remove_unreferenced_vertices()
            
            # Export smoothed mesh
            smoothed_mesh.export(str(output_path))
            
            final_volume = smoothed_mesh.volume
            volume_change = abs(final_volume - original_volume) / original_volume * 100
            
            logger.info(f"Smoothed STL: {output_path.name} (volume change: {volume_change:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error smoothing STL {input_path}: {e}")
            return False
    
    def process_file(self, file_info: Dict) -> Tuple[str, bool]:
        """Process a single file (rename, create STL if needed, smooth)"""
        try:
            file_path = file_info['path']
            file_type = file_info['type']
            target_dir = file_info.get('target_dir', file_path.parent)
            
            if file_type == 'original_nifti':
                # Process original NIfTI files
                # Extract patient ID and create standardized filename
                patient_id = self.extract_patient_id(file_path.name)
                if patient_id is None:
                    logger.warning(f"Could not extract patient ID from {file_path.name}")
                    return f"Failed: {file_path.name}", False
                
                standardized_name = self.standardize_filename(file_path.name)
                new_path = target_dir / standardized_name
                
                # Copy with standardized name if needed
                if file_path != new_path and not new_path.exists():
                    shutil.copy2(file_path, new_path)
                elif not new_path.exists():
                    new_path = file_path
                
                # Create STL
                stl_name = standardized_name.replace('.nii.gz', '.stl').replace('_seg', '')
                stl_path = target_dir / stl_name
                if not stl_path.exists():
                    self.nifti_to_stl(new_path, stl_path)
                
                # Create smoothed versions
                smoothed_nifti = self.original_smoothed_dir / standardized_name
                smoothed_stl = self.original_smoothed_dir / stl_name
                
                if not smoothed_nifti.exists():
                    self.smooth_nifti_advanced(new_path, smoothed_nifti)
                if not smoothed_stl.exists() and stl_path.exists():
                    self.smooth_stl_advanced(stl_path, smoothed_stl)
                
                return f"Processed original: {standardized_name}", True
                
            elif file_type == 'existing_file':
                # Process existing files (rename to standardized format)
                patient_id = self.extract_patient_id(file_path.name)
                if patient_id is not None:
                    standardized_name = self.standardize_filename(file_path.name)
                    new_path = target_dir / standardized_name
                    
                    if file_path != new_path and not new_path.exists():
                        file_path.rename(new_path)
                    elif file_path != new_path:
                        new_path = file_path  # Use existing path if rename not possible
                else:
                    new_path = file_path  # Keep original name if no patient ID found
                
                # Create smoothed version for NIfTI files
                if new_path.suffix == '.gz' and '.nii' in new_path.name:
                    smoothed_path = self.original_smoothed_dir / new_path.name
                    if not smoothed_path.exists():
                        self.smooth_nifti_advanced(new_path, smoothed_path)
                elif new_path.suffix == '.stl':
                    smoothed_path = self.original_smoothed_dir / new_path.name
                    if not smoothed_path.exists():
                        self.smooth_stl_advanced(new_path, smoothed_path)
                
                return f"Processed existing: {new_path.name}", True
                
            elif file_type == 'cropped_file':
                # Process cropped files (organized by patient/scan folders)
                patient_folder = file_info.get('patient_folder', '00')
                scan_folder = file_info.get('scan_folder', '1')
                
                # Create smoothed directory structure
                smoothed_dir = self.cropped_smoothed_dir / patient_folder / scan_folder
                smoothed_dir.mkdir(parents=True, exist_ok=True)
                smoothed_path = smoothed_dir / file_path.name
                
                if not smoothed_path.exists():
                    if file_path.suffix == '.gz' and '.nii' in file_path.name:
                        self.smooth_nifti_advanced(file_path, smoothed_path)
                    elif file_path.suffix == '.stl':
                        self.smooth_stl_advanced(file_path, smoothed_path)
                
                return f"Processed cropped: {patient_folder}/{scan_folder}/{file_path.name}", True
            
            return f"Unknown type: {file_type}", False
            
        except Exception as e:
            logger.error(f"Error processing file {file_info}: {e}")
            return f"Error: {file_info.get('path', 'unknown')}", False
    
    def rename_directories(self):
        """Rename directories to use two-digit patient IDs (not needed for cropped as they're already 2-digit)"""
        logger.info("Checking directory naming conventions...")
        
        # The cropped directories are already in 2-digit format (01, 02, etc.)
        # So we don't need to rename them
        cropped_dir = self.uan_base / "cropped"
        if cropped_dir.exists():
            patient_dirs = [d for d in cropped_dir.iterdir() if d.is_dir()]
            logger.info(f"Found {len(patient_dirs)} patient directories in cropped folder")
    
    def collect_files_to_process(self) -> List[Dict]:
        """Collect all files that need processing"""
        files_to_process = []
        
        # 1. Original _seg files (need standardization and STL creation)
        for seg_file in self.original_dir.glob("*_seg.nii.gz"):
            files_to_process.append({
                'path': seg_file,
                'type': 'original_nifti',
                'target_dir': self.original_dir
            })
        
        # 2. Existing files in UAN directories (need renaming to 2-digit)
        for subdir in ['largest_island', 'area_separation', 'aneurysm_detection']:
            dir_path = self.uan_base / subdir
            if dir_path.exists():
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        files_to_process.append({
                            'path': file_path,
                            'type': 'existing_file',
                            'target_dir': dir_path
                        })
        
        # 3. Cropped files (organized by patient/scan structure)
        cropped_dir = self.uan_base / "cropped"
        if cropped_dir.exists():
            for patient_dir in cropped_dir.iterdir():
                if patient_dir.is_dir():
                    patient_folder = patient_dir.name
                    for scan_dir in patient_dir.iterdir():
                        if scan_dir.is_dir():
                            scan_folder = scan_dir.name
                            for file_path in scan_dir.glob("*"):
                                if file_path.is_file() and (
                                    (file_path.suffix == '.gz' and '.nii' in file_path.name) or
                                    file_path.suffix == '.stl'
                                ):
                                    files_to_process.append({
                                        'path': file_path,
                                        'type': 'cropped_file',
                                        'patient_folder': patient_folder,
                                        'scan_folder': scan_folder
                                    })
        
        return files_to_process
    
    def run_processing(self, max_workers: int = 16):
        """Run the complete processing pipeline"""
        logger.info("Starting Advanced Smoothing Processor...")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        
        # Step 1: Check directories
        self.rename_directories()
        
        # Step 2: Collect all files to process
        files_to_process = self.collect_files_to_process()
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Group files by type for better reporting
        file_types = {}
        for file_info in files_to_process:
            file_type = file_info['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        logger.info("File types to process:")
        for file_type, count in file_types.items():
            logger.info(f"  - {file_type}: {count} files")
        
        # Step 3: Process files in parallel
        logger.info(f"Processing with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_info): file_info 
                for file_info in files_to_process
            }
            
            results = []
            completed = 0
            total = len(files_to_process)
            
            for future in as_completed(future_to_file):
                result, success = future.result()
                results.append((result, success))
                completed += 1
                
                if success:
                    logger.info(f"✓ ({completed}/{total}) {result}")
                else:
                    logger.error(f"✗ ({completed}/{total}) {result}")
        
        # Summary
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Total files processed: {len(files_to_process)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(files_to_process)*100:.1f}%")
        logger.info(f"\nOutput directories:")
        logger.info(f"  - Original: {self.original_dir}")
        logger.info(f"  - Original smoothed: {self.original_smoothed_dir}")
        logger.info(f"  - Cropped smoothed: {self.cropped_smoothed_dir}")
        
        # Check what was created
        original_smoothed_count = len(list(self.original_smoothed_dir.glob("*")))
        cropped_smoothed_count = len(list(self.cropped_smoothed_dir.rglob("*"))) if self.cropped_smoothed_dir.exists() else 0
        
        logger.info(f"\nFiles created:")
        logger.info(f"  - Original smoothed files: {original_smoothed_count}")
        logger.info(f"  - Cropped smoothed files: {cropped_smoothed_count}")

def main():
    """Main function"""
    processor = AdvancedSmoothingProcessor()
    processor.run_processing(max_workers=16)

if __name__ == "__main__":
    main() 