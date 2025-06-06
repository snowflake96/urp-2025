#!/usr/bin/env python3
"""
Fix all smoothing files - replace the broken empty files with properly smoothed ones
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage import filters, morphology
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmoothingFixer:
    """Fix all smoothing files with proper implementation"""
    
    def __init__(self):
        self.original_dir = Path("/home/jiwoo/urp/data/uan/original")
        self.broken_smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed")
        self.fixed_smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_fixed")
        
        # Create backup of broken files
        self.backup_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_broken_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Convert to binary (most vessel segmentations are binary)
            binary_data = (data > 0).astype(np.float32)
            
            # Apply Gaussian smoothing
            smoothed_data = ndimage.gaussian_filter(binary_data, sigma=sigma)
            
            # Apply threshold to maintain binary nature but with smoother edges
            threshold = 0.3  # This preserves smoothed edges
            smoothed_binary = (smoothed_data > threshold).astype(np.uint8)
            
            # Apply morphological closing to fill small gaps
            kernel = morphology.ball(1)
            smoothed_binary = morphology.binary_closing(smoothed_binary, kernel)
            
            # Convert back to original data type
            smoothed_binary = smoothed_binary.astype(data.dtype)
            
            # Save smoothed image
            smoothed_img = nib.Nifti1Image(smoothed_binary, img.affine, img.header)
            nib.save(smoothed_img, output_path)
            
            final_volume = np.sum(smoothed_binary > 0)
            volume_change = abs(final_volume - original_volume) / original_volume * 100
            
            logger.debug(f"✓ Smoothed {output_path.name}: {original_volume} → {final_volume} voxels (change: {volume_change:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error smoothing {input_path}: {e}")
            return False
    
    def process_single_file(self, original_file: Path) -> tuple[str, bool]:
        """Process a single original file to create proper smoothed version"""
        try:
            # Create smoothed filename
            smoothed_name = original_file.name.replace('.nii.gz', '_smoothed.nii.gz')
            broken_smoothed_path = self.broken_smoothed_dir / smoothed_name
            fixed_smoothed_path = self.fixed_smoothed_dir / smoothed_name
            
            # Backup broken file if it exists
            if broken_smoothed_path.exists():
                backup_path = self.backup_dir / smoothed_name
                if not backup_path.exists():
                    shutil.copy2(broken_smoothed_path, backup_path)
            
            # Create properly smoothed version
            success = self.smooth_nifti_proper(original_file, fixed_smoothed_path, sigma=1.0)
            
            if success:
                return f"Fixed: {original_file.name} → {smoothed_name}", True
            else:
                return f"Failed: {original_file.name}", False
                
        except Exception as e:
            logger.error(f"Error processing {original_file}: {e}")
            return f"Error: {original_file.name}", False
    
    def fix_all_smoothing(self, max_workers: int = 4):
        """Fix all smoothing files"""
        
        # Find all original NIfTI files
        original_files = list(self.original_dir.glob("*.nii.gz"))
        logger.info(f"Found {len(original_files)} original files to process")
        
        if not original_files:
            logger.error("No original files found")
            return
        
        logger.info(f"Processing all {len(original_files)} files with {max_workers} workers")
        logger.info(f"Output directory: {self.fixed_smoothed_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
        
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                            for file_path in original_files}
            
            # Process results
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                try:
                    result, success = future.result()
                    if success:
                        successful += 1
                        if i % 20 == 0:  # Log progress every 20 files
                            logger.info(f"Progress: {i}/{len(original_files)} ({i/len(original_files)*100:.1f}%)")
                    else:
                        failed += 1
                        logger.error(f"✗ {result}")
                except Exception as e:
                    failed += 1
                    logger.error(f"✗ Error processing {file_path.name}: {e}")
        
        logger.info(f"\n=== Smoothing Fix Complete ===")
        logger.info(f"Total files processed: {len(original_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        
        # Count output files
        fixed_files = list(self.fixed_smoothed_dir.glob("*.nii.gz"))
        backup_files = list(self.backup_dir.glob("*.nii.gz"))
        
        logger.info(f"\nFiles created:")
        logger.info(f"  - Fixed smoothed files: {len(fixed_files)}")
        logger.info(f"  - Backed up broken files: {len(backup_files)}")
        
        return successful, failed
    
    def replace_broken_files(self):
        """Replace broken smoothed files with fixed ones"""
        logger.info("Replacing broken smoothed files with fixed versions...")
        
        fixed_files = list(self.fixed_smoothed_dir.glob("*.nii.gz"))
        replaced = 0
        
        for fixed_file in fixed_files:
            broken_file = self.broken_smoothed_dir / fixed_file.name
            
            if broken_file.exists():
                # Replace broken file with fixed one
                shutil.copy2(fixed_file, broken_file)
                replaced += 1
                logger.debug(f"Replaced: {broken_file.name}")
        
        logger.info(f"✓ Replaced {replaced} broken smoothed files")
        return replaced

def main():
    print("=== Fixing All Smoothing Files ===")
    print("This will:")
    print("1. Backup all broken smoothed files")
    print("2. Create properly smoothed versions")
    print("3. Replace broken files with fixed ones")
    
    fixer = SmoothingFixer()
    
    # Step 1: Create all fixed smoothed files
    print(f"\nStep 1: Creating properly smoothed files...")
    successful, failed = fixer.fix_all_smoothing(max_workers=4)
    
    if successful > 0:
        # Step 2: Replace broken files
        print(f"\nStep 2: Replacing broken files...")
        replaced = fixer.replace_broken_files()
        
        print(f"\n✅ SUCCESS!")
        print(f"  - Created {successful} properly smoothed files")
        print(f"  - Replaced {replaced} broken files")
        print(f"  - Backed up {replaced} broken files")
        print(f"\nThe original_smoothed directory now contains properly smoothed files!")
    else:
        print(f"\n❌ FAILED: No files were successfully processed")

if __name__ == "__main__":
    main() 