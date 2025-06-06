#!/usr/bin/env python3
"""
Clean UAN Processor - Complete Pipeline from Scratch
1. Copy and rename _seg files from segmentation output with proper 2-digit patient numbering
2. Apply largest island extraction to copied files
3. Create STL versions of all NIfTI files
4. Create smoothed versions with "_smoothed" suffix
5. Focus smoothing on removing popped out blocks from low resolution
6. Use 16 CPUs for parallel processing
"""

import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import logging
import re
import shutil
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage, interpolate
from skimage import measure, morphology, filters
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CleanUANProcessor:
    """Complete UAN processing pipeline from scratch"""
    
    def __init__(self):
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.source_dir = Path("/home/jiwoo/urp/data/segmentation/aneu/UAN_processed/Output")
        self.original_dir = self.uan_base / "original"
        self.original_smoothed_dir = self.uan_base / "original_smoothed"
        
        # Create directories
        for dir_path in [self.original_dir, self.original_smoothed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def extract_patient_info(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract patient number and MRA number from filename"""
        # Pattern: {patient}_MRA{1|2}_seg.nii.gz
        match = re.search(r'^(\d+)_MRA(\d+)_seg\.nii\.gz$', filename)
        if match:
            patient_num = int(match.group(1))
            mra_num = int(match.group(2))
            return patient_num, mra_num
        return None, None
    
    def create_standardized_filename(self, patient_num: int, mra_num: int) -> str:
        """Create standardized filename with 2-digit patient number"""
        return f"{patient_num:02d}_MRA{mra_num}_seg.nii.gz"
    
    def get_largest_island(self, data: np.ndarray) -> np.ndarray:
        """Extract the largest connected component (island)"""
        if np.sum(data) == 0:
            return data
        
        # Convert to binary
        binary_data = data > 0
        
        # Find connected components
        labeled, num_labels = ndimage.label(binary_data)
        
        if num_labels == 0:
            return data
        
        # Find the largest component
        component_sizes = ndimage.sum(binary_data, labeled, range(1, num_labels + 1))
        largest_component = np.argmax(component_sizes) + 1
        
        # Create mask for largest component
        largest_island = (labeled == largest_component).astype(data.dtype)
        
        return largest_island
    
    def nifti_to_stl(self, nifti_path: Path, stl_path: Path) -> bool:
        """Convert NIfTI file to STL using marching cubes"""
        try:
            # Load NIfTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            if np.sum(data) == 0:
                logger.warning(f"Empty mask, skipping STL generation for {stl_path}")
                return False
            
            # Get spacing from header
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
            logger.info(f"Created STL: {stl_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating STL {stl_path}: {e}")
            return False
    
    def detect_low_resolution_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect popped out blocks and lines from low resolution"""
        # Convert to binary
        binary = data > 0
        
        # Find connected components
        labeled, num_labels = ndimage.label(binary)
        
        # Detect artifacts
        artifacts = np.zeros_like(binary)
        
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
            
            # Calculate volume and shape metrics
            volume = np.sum(component)
            bbox_volume = np.prod(bbox_dims)
            volume_ratio = volume / max(bbox_volume, 1)
            
            # Detect artifacts based on:
            # 1. Very small isolated components (noise)
            # 2. Long thin structures (false connections)
            # 3. Square/rectangular blocks (pixelation artifacts)
            # 4. Low volume-to-bbox ratio (spiky structures)
            
            max_dim = max(bbox_dims)
            min_dim = min(bbox_dims)
            aspect_ratio = max_dim / max(min_dim, 1)
            
            is_artifact = (
                volume < 20 or  # Very small components
                aspect_ratio > 15 or  # Very elongated (false connections)
                volume_ratio < 0.2 or  # Low density (spiky)
                (max_dim > 8 and min_dim < 2)  # Thin protrusions
            )
            
            if is_artifact:
                artifacts |= component
        
        return artifacts
    
    def smooth_nifti_conservative(self, input_path: Path, output_path: Path) -> bool:
        """Conservative smoothing focused on removing low-resolution artifacts"""
        try:
            # Load image
            img = nib.load(input_path)
            data = img.get_fdata().astype(np.float32)
            original_volume = np.sum(data > 0)
            
            if original_volume == 0:
                logger.warning(f"Empty image: {input_path}")
                return False
            
            # Detect artifacts
            artifacts = self.detect_low_resolution_artifacts(data)
            
            # Create binary mask
            binary_data = (data > 0).astype(np.float32)
            
            # Conservative smoothing strategy
            smoothed_data = binary_data.copy()
            
            # 1. Remove detected artifacts
            if np.sum(artifacts) > 0:
                smoothed_data[artifacts] = 0
                logger.info(f"Removed {np.sum(artifacts)} artifact voxels from {input_path.name}")
            
            # 2. Very mild smoothing to reduce pixelation (low smoothing factor)
            main_structure = smoothed_data > 0
            if np.sum(main_structure) > 0:
                # Use very small sigma for conservative smoothing
                mild_smooth = ndimage.gaussian_filter(
                    smoothed_data.astype(np.float32), sigma=0.5
                )
                smoothed_data = mild_smooth
            
            # 3. Apply median filter to remove isolated pixels
            smoothed_data = filters.median(smoothed_data, morphology.ball(1))
            
            # 4. Conservative volume preservation
            current_volume = np.sum(smoothed_data > 0.5)
            if current_volume > 0:
                # Allow small volume change (up to 5%)
                volume_factor = original_volume / current_volume
                if 0.95 <= volume_factor <= 1.05:
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
    
    def smooth_stl_conservative(self, input_path: Path, output_path: Path) -> bool:
        """Conservative smoothing for STL files"""
        try:
            # Load mesh
            mesh = trimesh.load(str(input_path))
            original_volume = mesh.volume
            
            if original_volume <= 0:
                logger.warning(f"Invalid mesh volume: {input_path}")
                return False
            
            # Very mild Laplacian smoothing (conservative)
            smoothed_vertices = mesh.vertices.copy()
            for iteration in range(2):  # Only 2 iterations for conservative smoothing
                vertex_neighbors = mesh.vertex_neighbors
                new_vertices = smoothed_vertices.copy()
                
                for i, neighbors in enumerate(vertex_neighbors):
                    if len(neighbors) > 0:
                        neighbor_positions = smoothed_vertices[neighbors]
                        centroid = np.mean(neighbor_positions, axis=0)
                        # Very conservative weighting (90% original, 10% smoothed)
                        new_vertices[i] = 0.9 * smoothed_vertices[i] + 0.1 * centroid
                
                smoothed_vertices = new_vertices
            
            # Create smoothed mesh
            smoothed_mesh = trimesh.Trimesh(
                vertices=smoothed_vertices, 
                faces=mesh.faces
            )
            
            # Volume preservation
            current_volume = smoothed_mesh.volume
            if current_volume > 0:
                scale_factor = (original_volume / current_volume) ** (1/3)
                if 0.95 <= scale_factor <= 1.05:  # Conservative scaling
                    smoothed_mesh.vertices *= scale_factor
            
            # Final cleanup
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
    
    def process_single_file(self, source_file: Path) -> Tuple[str, bool]:
        """Process a single segmentation file through the complete pipeline"""
        try:
            # Extract patient and MRA info
            patient_num, mra_num = self.extract_patient_info(source_file.name)
            if patient_num is None or mra_num is None:
                return f"Failed to parse filename: {source_file.name}", False
            
            # Create standardized filename
            standard_name = self.create_standardized_filename(patient_num, mra_num)
            target_nifti = self.original_dir / standard_name
            target_stl = self.original_dir / standard_name.replace('.nii.gz', '.stl')
            
            # Step 1: Copy and apply largest island extraction
            img = nib.load(source_file)
            data = img.get_fdata()
            
            # Apply largest island extraction
            largest_island_data = self.get_largest_island(data)
            
            # Save processed NIfTI
            processed_img = nib.Nifti1Image(largest_island_data, img.affine, img.header)
            nib.save(processed_img, target_nifti)
            
            # Step 2: Create STL from processed NIfTI
            self.nifti_to_stl(target_nifti, target_stl)
            
            # Step 3: Create smoothed versions
            smoothed_nifti_name = standard_name.replace('.nii.gz', '_smoothed.nii.gz')
            smoothed_stl_name = standard_name.replace('.nii.gz', '_smoothed.stl')
            
            smoothed_nifti_path = self.original_smoothed_dir / smoothed_nifti_name
            smoothed_stl_path = self.original_smoothed_dir / smoothed_stl_name
            
            # Create smoothed NIfTI
            self.smooth_nifti_conservative(target_nifti, smoothed_nifti_path)
            
            # Create STL from smoothed NIfTI
            self.nifti_to_stl(smoothed_nifti_path, smoothed_stl_path)
            
            return f"Processed: {source_file.name} → {standard_name}", True
            
        except Exception as e:
            logger.error(f"Error processing {source_file}: {e}")
            return f"Error: {source_file.name}", False
    
    def collect_source_files(self) -> List[Path]:
        """Collect all _seg files from source directory"""
        seg_files = list(self.source_dir.glob("*_seg.nii.gz"))
        logger.info(f"Found {len(seg_files)} segmentation files to process")
        return seg_files
    
    def run_complete_pipeline(self, max_workers: int = 16):
        """Run the complete processing pipeline"""
        logger.info("Starting Clean UAN Processor...")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        
        # Collect source files
        source_files = self.collect_source_files()
        if not source_files:
            logger.error("No segmentation files found!")
            return
        
        # Show sample of what will be processed
        logger.info("Sample files to process:")
        for i, file in enumerate(source_files[:5]):
            patient_num, mra_num = self.extract_patient_info(file.name)
            if patient_num is not None:
                standard_name = self.create_standardized_filename(patient_num, mra_num)
                logger.info(f"  {file.name} → {standard_name}")
        if len(source_files) > 5:
            logger.info(f"  ... and {len(source_files) - 5} more files")
        
        # Process files in parallel
        logger.info(f"Processing {len(source_files)} files with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, file): file 
                for file in source_files
            }
            
            results = []
            completed = 0
            total = len(source_files)
            
            for future in as_completed(future_to_file):
                result, success = future.result()
                results.append((result, success))
                completed += 1
                
                if success:
                    logger.info(f"✓ ({completed}/{total}) {result}")
                else:
                    logger.error(f"✗ ({completed}/{total}) {result}")
        
        # Final summary
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Total files processed: {len(source_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(source_files)*100:.1f}%")
        
        # Check output directories
        original_count = len(list(self.original_dir.glob("*")))
        smoothed_count = len(list(self.original_smoothed_dir.glob("*")))
        
        logger.info(f"\nOutput directories:")
        logger.info(f"  - Original (with largest island): {self.original_dir}")
        logger.info(f"    Files created: {original_count}")
        logger.info(f"  - Original smoothed: {self.original_smoothed_dir}")
        logger.info(f"    Files created: {smoothed_count}")
        
        # Expected counts
        expected_original = successful * 2  # .nii.gz + .stl per file
        expected_smoothed = successful * 2  # .nii.gz + .stl per file
        
        logger.info(f"\nExpected vs Actual:")
        logger.info(f"  - Original: {original_count}/{expected_original}")
        logger.info(f"  - Smoothed: {smoothed_count}/{expected_smoothed}")

def main():
    """Main function"""
    processor = CleanUANProcessor()
    processor.run_complete_pipeline(max_workers=16)

if __name__ == "__main__":
    main() 