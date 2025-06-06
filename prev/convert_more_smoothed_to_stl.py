#!/usr/bin/env python3
"""
Convert More Smoothed NIfTI Files to STL
- Convert all 504 NIfTI files from original_smoothed_more to STL
- Organize by smoothing level (moderate, aggressive, ultra)
- Use marching cubes for high-quality mesh generation
- Parallel processing for fast conversion
"""

import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
from skimage import measure
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoreSmoothedSTLConverter:
    """Convert more smoothed NIfTI files to STL format"""
    
    def __init__(self):
        self.input_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_more")
        self.output_base = Path("/home/jiwoo/urp/data/uan/original_smoothed_more_stl")
        
        # Create organized output directories
        self.output_dirs = {
            'moderate': self.output_base / "moderate",
            'aggressive': self.output_base / "aggressive", 
            'ultra': self.output_base / "ultra"
        }
        
        # Create all output directories
        for level_dir in self.output_dirs.values():
            level_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create a combined directory for easy browsing
        self.combined_dir = self.output_base / "all_levels"
        self.combined_dir.mkdir(parents=True, exist_ok=True)
    
    def nifti_to_stl_improved(self, nifti_path: Path, stl_path: Path) -> bool:
        """Convert NIfTI to STL with improved quality settings"""
        
        try:
            # Load NIfTI
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            if np.sum(data > 0) == 0:
                logger.warning(f"Empty NIfTI file: {nifti_path.name}")
                return False
            
            # Get voxel spacing for proper scaling
            spacing = img.header.get_zooms()
            
            # Apply marching cubes with appropriate level
            # Use level=0.5 for binary data
            vertices, faces, normals, values = measure.marching_cubes(
                data, 
                level=0.5, 
                spacing=spacing,
                gradient_direction='descent'
            )
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            
            # Improve mesh quality
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Fill holes if any (but be careful not to over-process smoothed data)
            if not mesh.is_watertight:
                mesh.fill_holes()
            
            # Smooth the mesh slightly (very light smoothing to preserve the NIfTI smoothing)
            if hasattr(mesh, 'smoothed'):
                mesh = mesh.smoothed()
            
            # Export STL
            mesh.export(str(stl_path))
            
            # Log quality metrics
            volume_mm3 = mesh.volume
            surface_area_mm2 = mesh.area
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            
            logger.debug(f"Created STL: {stl_path.name} - {num_vertices} vertices, {num_faces} faces, "
                        f"Volume: {volume_mm3:.1f} mm³")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {nifti_path.name} to STL: {e}")
            return False
    
    def process_single_file(self, nifti_file: Path) -> tuple[str, bool]:
        """Process a single NIfTI file to STL"""
        
        try:
            # Determine smoothing level from filename
            if '_moderate.nii.gz' in nifti_file.name:
                level = 'moderate'
            elif '_aggressive.nii.gz' in nifti_file.name:
                level = 'aggressive'
            elif '_ultra.nii.gz' in nifti_file.name:
                level = 'ultra'
            else:
                logger.error(f"Cannot determine smoothing level for: {nifti_file.name}")
                return f"Error: Unknown level - {nifti_file.name}", False
            
            # Create STL filename
            stl_name = nifti_file.name.replace('.nii.gz', '.stl')
            
            # Output to level-specific directory
            stl_path_level = self.output_dirs[level] / stl_name
            
            # Also output to combined directory with level prefix
            combined_name = f"{level}_{stl_name}"
            stl_path_combined = self.combined_dir / combined_name
            
            # Convert to STL
            success = self.nifti_to_stl_improved(nifti_file, stl_path_level)
            
            if success:
                # Copy to combined directory for easy browsing
                import shutil
                shutil.copy2(stl_path_level, stl_path_combined)
                
                return f"✓ {level}: {nifti_file.name}", True
            else:
                return f"✗ {level}: {nifti_file.name}", False
                
        except Exception as e:
            logger.error(f"Error processing {nifti_file.name}: {e}")
            return f"✗ Error: {nifti_file.name}", False
    
    def convert_all_files(self, max_workers: int = 8):
        """Convert all NIfTI files to STL with parallel processing"""
        
        # Find all NIfTI files
        nifti_files = list(self.input_dir.glob("*.nii.gz"))
        logger.info(f"Found {len(nifti_files)} NIfTI files to convert to STL")
        
        if not nifti_files:
            logger.error("No NIfTI files found in input directory")
            return
        
        # Count by level
        levels_count = {}
        for level in ['moderate', 'aggressive', 'ultra']:
            level_files = [f for f in nifti_files if f'_{level}.nii.gz' in f.name]
            levels_count[level] = len(level_files)
            logger.info(f"  - {level.capitalize()}: {len(level_files)} files")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        logger.info(f"Converting {len(nifti_files)} files using {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, nifti_file): nifti_file 
                            for nifti_file in nifti_files}
            
            # Process results
            for i, future in enumerate(as_completed(future_to_file), 1):
                nifti_file = future_to_file[future]
                
                try:
                    message, success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(message)
                    
                    # Progress update every 50 files
                    if i % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = i / elapsed * 60
                        logger.info(f"Progress: {i}/{len(nifti_files)} ({i/len(nifti_files)*100:.1f}%, {rate:.1f} files/min)")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"Task failed for {nifti_file.name}: {e}")
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        logger.info(f"\n=== STL Conversion Complete ===")
        logger.info(f"Total files processed: {len(nifti_files)}")
        logger.info(f"Successful conversions: {successful}")
        logger.info(f"Failed conversions: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Average time per file: {elapsed_time/len(nifti_files):.2f} seconds")
        
        # Check output directories
        logger.info(f"\n=== Output Directories ===")
        logger.info(f"Base directory: {self.output_base}")
        
        total_stl_files = 0
        for level, level_dir in self.output_dirs.items():
            stl_files = list(level_dir.glob("*.stl"))
            total_stl_files += len(stl_files)
            logger.info(f"  - {level.capitalize()}: {len(stl_files)} STL files")
        
        combined_files = list(self.combined_dir.glob("*.stl"))
        logger.info(f"  - Combined (all levels): {len(combined_files)} STL files")
        
        # Calculate storage usage
        total_size = sum(f.stat().st_size for f in self.output_base.rglob("*.stl"))
        logger.info(f"\nTotal STL storage: {total_size / (1024**2):.1f} MB")
        
        return successful, failed
    
    def create_viewing_guide(self):
        """Create a guide for viewing the STL files"""
        
        guide_content = f"""# STL Viewing Guide

## Directory Structure
```
{self.output_base}/
├── moderate/        # 168 moderate smoothed STL files
├── aggressive/      # 168 aggressive smoothed STL files  
├── ultra/          # 168 ultra smoothed STL files
└── all_levels/     # All 504 files with level prefixes
```

## File Naming Convention
- **Level directories**: `[Patient]_MRA[N]_seg_smoothed_[level].stl`
- **Combined directory**: `[level]_[Patient]_MRA[N]_seg_smoothed_[level].stl`

## Recommended Viewers
1. **MeshLab** (Free, cross-platform)
   - Open multiple files for comparison
   - Good lighting and shading options
   
2. **Blender** (Free, advanced)
   - Professional 3D visualization
   - Animation and rendering capabilities
   
3. **3D Slicer** (Medical imaging focused)
   - Medical-grade visualization
   - Measurement tools
   
4. **ParaView** (Scientific visualization)
   - Advanced analysis tools
   - Good for large datasets

## Viewing Tips
- **Compare smoothing levels**: Load same patient with different levels
- **Use proper lighting**: Smooth surfaces show quality better with good lighting
- **Check surface details**: Zoom in to see smoothing quality differences
- **Measure volumes**: Verify volume changes across smoothing levels

## Quality Assessment
- **Moderate**: Good balance of smoothness and detail
- **Aggressive**: Very smooth, ideal for CFD
- **Ultra**: Research-grade smoothness, organic appearance

## Example Comparisons
Try loading these files together for comparison:
- `01_MRA1_seg_smoothed_moderate.stl`
- `01_MRA1_seg_smoothed_aggressive.stl` 
- `01_MRA1_seg_smoothed_ultra.stl`
"""
        
        guide_file = self.output_base / "STL_VIEWING_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Created viewing guide: {guide_file}")

def main():
    print("=== Converting More Smoothed NIfTI Files to STL ===")
    print("This will convert all 504 files to STL format for 3D viewing")
    print("Organizing by smoothing level: moderate, aggressive, ultra")
    
    converter = MoreSmoothedSTLConverter()
    
    # Convert all files
    successful, failed = converter.convert_all_files(max_workers=8)
    
    # Create viewing guide
    converter.create_viewing_guide()
    
    print(f"\n✅ STL conversion complete!")
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    print(f"\nSTL files organized in:")
    print(f"  📁 {converter.output_base}")
    print(f"    ├── moderate/     (168 files)")
    print(f"    ├── aggressive/   (168 files)")
    print(f"    ├── ultra/        (168 files)")
    print(f"    └── all_levels/   (504 files)")
    print(f"\n📖 See STL_VIEWING_GUIDE.md for viewing instructions")

if __name__ == "__main__":
    main() 