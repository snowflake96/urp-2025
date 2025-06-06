#!/usr/bin/env python3
"""
Create STL files for smoothed NIfTI files
- Generate STL from smoothed NIfTI if not empty
- If smoothed NIfTI is empty, generate STL from original NIfTI
- Use 16 CPUs for parallel processing
"""

import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import logging
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage import measure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmoothedSTLGenerator:
    """Generate STL files for smoothed NIfTI files"""
    
    def __init__(self):
        self.uan_base = Path("/home/jiwoo/urp/data/uan")
        self.original_dir = self.uan_base / "original"
        self.original_smoothed_dir = self.uan_base / "original_smoothed"
    
    def nifti_to_stl(self, nifti_path: Path, stl_path: Path) -> bool:
        """Convert NIfTI file to STL using marching cubes"""
        try:
            # Load NIfTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            if np.sum(data) == 0:
                logger.warning(f"Empty mask: {nifti_path.name}")
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
    
    def process_single_file(self, smoothed_nifti_path: Path) -> Tuple[str, bool]:
        """Process a single smoothed NIfTI file to create STL"""
        try:
            # Create STL filename
            stl_name = smoothed_nifti_path.name.replace('.nii.gz', '.stl')
            stl_path = self.original_smoothed_dir / stl_name
            
            # Skip if STL already exists
            if stl_path.exists():
                return f"Already exists: {stl_name}", True
            
            # Check if smoothed NIfTI is empty
            img = nib.load(smoothed_nifti_path)
            data = img.get_fdata()
            
            if np.sum(data) == 0:
                # Use original NIfTI instead
                original_name = smoothed_nifti_path.name.replace('_smoothed', '')
                original_path = self.original_dir / original_name
                
                if original_path.exists():
                    success = self.nifti_to_stl(original_path, stl_path)
                    if success:
                        return f"Created from original: {stl_name}", True
                    else:
                        return f"Failed to create from original: {stl_name}", False
                else:
                    return f"Original file not found: {original_name}", False
            else:
                # Use smoothed NIfTI
                success = self.nifti_to_stl(smoothed_nifti_path, stl_path)
                if success:
                    return f"Created from smoothed: {stl_name}", True
                else:
                    return f"Failed to create from smoothed: {stl_name}", False
                    
        except Exception as e:
            logger.error(f"Error processing {smoothed_nifti_path}: {e}")
            return f"Error: {smoothed_nifti_path.name}", False
    
    def collect_smoothed_nifti_files(self) -> List[Path]:
        """Collect all smoothed NIfTI files"""
        nifti_files = list(self.original_smoothed_dir.glob("*_smoothed.nii.gz"))
        logger.info(f"Found {len(nifti_files)} smoothed NIfTI files")
        return nifti_files
    
    def run_stl_generation(self, max_workers: int = 16):
        """Run STL generation for all smoothed NIfTI files"""
        logger.info("Starting smoothed STL generation...")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        
        # Collect smoothed NIfTI files
        nifti_files = self.collect_smoothed_nifti_files()
        if not nifti_files:
            logger.error("No smoothed NIfTI files found!")
            return
        
        # Check current STL count
        current_stl_count = len(list(self.original_smoothed_dir.glob("*.stl")))
        logger.info(f"Current STL files: {current_stl_count}")
        logger.info(f"Need to process: {len(nifti_files)} NIfTI files")
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, nifti_file): nifti_file 
                for nifti_file in nifti_files
            }
            
            results = []
            completed = 0
            total = len(nifti_files)
            
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
        already_existed = sum(1 for result, success in results if success and "Already exists" in result)
        created_from_smoothed = sum(1 for result, success in results if success and "Created from smoothed" in result)
        created_from_original = sum(1 for result, success in results if success and "Created from original" in result)
        
        # Check final counts
        final_nifti_count = len(list(self.original_smoothed_dir.glob("*_smoothed.nii.gz")))
        final_stl_count = len(list(self.original_smoothed_dir.glob("*_smoothed.stl")))
        
        logger.info(f"\n=== STL Generation Complete ===")
        logger.info(f"Total files processed: {len(nifti_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(nifti_files)*100:.1f}%")
        logger.info(f"\nBreakdown:")
        logger.info(f"  - Already existed: {already_existed}")
        logger.info(f"  - Created from smoothed: {created_from_smoothed}")
        logger.info(f"  - Created from original: {created_from_original}")
        logger.info(f"\nFinal counts in original_smoothed:")
        logger.info(f"  - NIfTI files: {final_nifti_count}")
        logger.info(f"  - STL files: {final_stl_count}")
        logger.info(f"  - Total files: {final_nifti_count + final_stl_count}")

def main():
    """Main function"""
    generator = SmoothedSTLGenerator()
    generator.run_stl_generation(max_workers=16)

if __name__ == "__main__":
    main() 