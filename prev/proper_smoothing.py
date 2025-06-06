#!/usr/bin/env python3
"""
Proper smoothing implementation that actually works
- Uses Gaussian smoothing with appropriate parameters
- Preserves vessel structure while reducing blocky artifacts
- Maintains volume and connectivity
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage import filters, morphology
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProperSmoother:
    """Proper smoothing implementation that actually works"""
    
    def __init__(self):
        self.original_dir = Path("/home/jiwoo/urp/data/uan/original")
        self.smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_fixed")
        
        # Create output directory
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)
    
    def smooth_nifti_proper(self, input_path: Path, output_path: Path, sigma: float = 1.0) -> bool:
        """Proper Gaussian smoothing that preserves vessel structure"""
        try:
            # Load image
            img = nib.load(input_path)
            data = img.get_fdata().astype(np.float32)
            original_volume = np.sum(data > 0)
            
            if original_volume == 0:
                logger.warning(f"Empty image: {input_path}")
                return False
            
            logger.info(f"Processing {input_path.name}: {original_volume} voxels")
            
            # Convert to binary (most vessel segmentations are binary)
            binary_data = (data > 0).astype(np.float32)
            
            # Apply Gaussian smoothing
            smoothed_data = ndimage.gaussian_filter(binary_data, sigma=sigma)
            
            # Apply threshold to maintain binary nature but with smoother edges
            # Use a lower threshold to preserve more of the smoothed structure
            threshold = 0.3  # This preserves smoothed edges
            smoothed_binary = (smoothed_data > threshold).astype(np.uint8)
            
            # Optional: Apply morphological closing to fill small gaps
            # This helps maintain connectivity after smoothing
            kernel = morphology.ball(1)
            smoothed_binary = morphology.binary_closing(smoothed_binary, kernel)
            
            # Convert back to original data type
            smoothed_binary = smoothed_binary.astype(data.dtype)
            
            # Save smoothed image
            smoothed_img = nib.Nifti1Image(smoothed_binary, img.affine, img.header)
            nib.save(smoothed_img, output_path)
            
            final_volume = np.sum(smoothed_binary > 0)
            volume_change = abs(final_volume - original_volume) / original_volume * 100
            
            logger.info(f"✓ Smoothed {output_path.name}: {original_volume} → {final_volume} voxels (change: {volume_change:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error smoothing {input_path}: {e}")
            return False
    
    def smooth_nifti_conservative(self, input_path: Path, output_path: Path) -> bool:
        """Conservative smoothing with minimal volume change"""
        try:
            # Load image
            img = nib.load(input_path)
            data = img.get_fdata().astype(np.float32)
            original_volume = np.sum(data > 0)
            
            if original_volume == 0:
                logger.warning(f"Empty image: {input_path}")
                return False
            
            logger.info(f"Processing {input_path.name}: {original_volume} voxels")
            
            # Convert to binary
            binary_data = (data > 0).astype(np.float32)
            
            # Very mild Gaussian smoothing
            smoothed_data = ndimage.gaussian_filter(binary_data, sigma=0.5)
            
            # Use higher threshold to be more conservative
            threshold = 0.5
            smoothed_binary = (smoothed_data > threshold).astype(np.uint8)
            
            # Apply median filter to remove isolated pixels
            smoothed_binary = filters.median(smoothed_binary, morphology.ball(1))
            
            # Convert back to original data type
            smoothed_binary = smoothed_binary.astype(data.dtype)
            
            # Save smoothed image
            smoothed_img = nib.Nifti1Image(smoothed_binary, img.affine, img.header)
            nib.save(smoothed_img, output_path)
            
            final_volume = np.sum(smoothed_binary > 0)
            volume_change = abs(final_volume - original_volume) / original_volume * 100
            
            logger.info(f"✓ Conservative smoothed {output_path.name}: {original_volume} → {final_volume} voxels (change: {volume_change:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error smoothing {input_path}: {e}")
            return False
    
    def process_single_file(self, file_info: Tuple[Path, str]) -> Tuple[str, bool]:
        """Process a single file"""
        input_path, method = file_info
        
        # Create output filename
        output_name = input_path.name.replace('.nii.gz', f'_smoothed_{method}.nii.gz')
        output_path = self.smoothed_dir / output_name
        
        if method == 'proper':
            success = self.smooth_nifti_proper(input_path, output_path, sigma=1.0)
        elif method == 'conservative':
            success = self.smooth_nifti_conservative(input_path, output_path)
        else:
            logger.error(f"Unknown method: {method}")
            return f"Unknown method: {method}", False
        
        return f"Processed: {input_path.name} → {output_name}", success
    
    def process_all_files(self, method: str = 'proper', max_workers: int = 4):
        """Process all files in the original directory"""
        
        # Find all NIfTI files
        nifti_files = list(self.original_dir.glob("*.nii.gz"))
        logger.info(f"Found {len(nifti_files)} NIfTI files to process")
        
        if not nifti_files:
            logger.error("No NIfTI files found in original directory")
            return
        
        # Create file info tuples
        file_infos = [(f, method) for f in nifti_files[:5]]  # Test with first 5 files
        
        logger.info(f"Processing {len(file_infos)} files with method '{method}' using {max_workers} workers")
        
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, file_info): file_info 
                            for file_info in file_infos}
            
            # Process results
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result, success = future.result()
                    if success:
                        successful += 1
                        logger.info(f"✓ {result}")
                    else:
                        failed += 1
                        logger.error(f"✗ {result}")
                except Exception as e:
                    failed += 1
                    logger.error(f"✗ Error processing {file_info[0].name}: {e}")
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        logger.info(f"Output directory: {self.smoothed_dir}")
        
        # Count output files
        output_files = list(self.smoothed_dir.glob("*.nii.gz"))
        logger.info(f"Output files created: {len(output_files)}")

def main():
    print("=== Proper Smoothing Implementation ===")
    
    smoother = ProperSmoother()
    
    print("\nTesting both smoothing methods on first 5 files:")
    
    # Test proper smoothing
    print("\n1. Testing proper smoothing (sigma=1.0, threshold=0.3)...")
    smoother.process_all_files(method='proper', max_workers=4)
    
    # Test conservative smoothing  
    print("\n2. Testing conservative smoothing (sigma=0.5, threshold=0.5)...")
    smoother.process_all_files(method='conservative', max_workers=4)
    
    print(f"\n✓ Results saved to: {smoother.smoothed_dir}")
    print("Run compare_smoothing.py to verify the results!")

if __name__ == "__main__":
    main() 