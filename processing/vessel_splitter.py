#!/usr/bin/env python3
"""
Vessel Splitter - Fix Kissing Artifacts in Cerebrovascular Segmentation

This script separates vessels that are wrongly connected by thin bridges
in binary vessel masks, using multiple morphological and watershed techniques.

Author: Cerebrovascular Analysis Pipeline
"""

import nibabel as nib
import numpy as np
import os
import argparse
from pathlib import Path
from datetime import datetime

# Core image processing
from skimage.morphology import (
    ball, binary_erosion, binary_dilation, binary_opening, binary_closing,
    remove_small_objects, skeletonize, disk, medial_axis
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import (
    label, binary_fill_holes, distance_transform_edt,
    gaussian_filter, maximum_filter, convolve
)
from scipy import ndimage
from skimage.measure import regionprops
import cv2

class VesselSplitter:
    """
    Advanced vessel splitting for cerebrovascular segmentation
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.methods = {
            'erosion_dilation': self._erosion_dilation_method,
            'watershed': self._watershed_method,
            'skeleton_guided': self._skeleton_guided_method,
            'distance_transform': self._distance_transform_method
        }
    
    def log(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return image object and binary data"""
        self.log(f"Loading NIfTI file: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Convert to binary if not already
        if data.max() > 1:
            self.log("Converting to binary mask (threshold > 0.5)")
            binary_data = (data > 0.5).astype(bool)
        else:
            binary_data = (data > 0).astype(bool)
        
        self.log(f"Loaded volume shape: {binary_data.shape}")
        self.log(f"Vessel voxels: {np.sum(binary_data):,}")
        self.log(f"Vessel volume: {np.sum(binary_data) / binary_data.size * 100:.2f}%")
        
        return img, binary_data
    
    def save_components(self, components, original_img, output_dir, basename, method_name):
        """Save separated components as individual NIfTI files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, component in enumerate(components, 1):
            if np.sum(component) == 0:  # Skip empty components
                continue
                
            # Create output filename
            output_file = output_dir / f"{basename}_{method_name}_part_{i:02d}.nii.gz"
            
            # Create new NIfTI image
            component_img = nib.Nifti1Image(
                component.astype(np.uint8), 
                original_img.affine, 
                original_img.header
            )
            
            # Save file
            nib.save(component_img, output_file)
            
            voxel_count = np.sum(component)
            self.log(f"Saved component {i}: {output_file} ({voxel_count:,} voxels)")
            saved_files.append(output_file)
        
        return saved_files
    
    def _erosion_dilation_method(self, mask, erosion_radius=1, min_size=100):
        """
        Classic erosion-dilation method to break thin bridges
        """
        self.log(f"Method 1: Erosion-Dilation (radius={erosion_radius})")
        
        # Step 1: Erode to break bridges
        structuring_element = ball(erosion_radius)
        eroded = binary_erosion(mask, structuring_element)
        self.log(f"After erosion: {np.sum(eroded):,} voxels remaining")
        
        # Step 2: Label connected components
        labeled, n_components = label(eroded)
        self.log(f"Found {n_components} components after erosion")
        
        # Step 3: Dilate each component back
        components = []
        for i in range(1, n_components + 1):
            component_mask = (labeled == i)
            
            # Skip very small components
            if np.sum(component_mask) < min_size:
                self.log(f"Skipping small component {i} ({np.sum(component_mask)} voxels)")
                continue
            
            # Dilate back to approximate original size
            dilated = binary_dilation(component_mask, structuring_element)
            
            # Ensure we don't go outside original mask
            dilated = dilated & mask
            
            components.append(dilated)
            self.log(f"Component {len(components)}: {np.sum(dilated):,} voxels")
        
        return components
    
    def _watershed_method(self, mask, min_distance=3, min_size=100):
        """
        Watershed segmentation method for splitting connected vessels
        """
        self.log(f"Method 2: Watershed (min_distance={min_distance})")
        
        # Step 1: Distance transform
        distance = distance_transform_edt(mask)
        
        # Step 2: Find local maxima as seeds
        local_maxima_coords = peak_local_max(
            distance, 
            min_distance=min_distance,
            threshold_abs=min_distance/2
        )
        
        # Convert coordinates to boolean mask
        local_maxima = np.zeros_like(distance, dtype=bool)
        if len(local_maxima_coords) > 0:
            local_maxima[tuple(local_maxima_coords.T)] = True
        
        seeds, n_seeds = label(local_maxima)
        self.log(f"Found {n_seeds} watershed seeds")
        
        if n_seeds == 0:
            self.log("No seeds found, returning original mask")
            return [mask]
        
        # Step 3: Watershed segmentation
        watershed_labels = watershed(-distance, seeds, mask=mask)
        
        # Step 4: Extract components
        components = []
        for i in range(1, n_seeds + 1):
            component = (watershed_labels == i)
            
            if np.sum(component) >= min_size:
                components.append(component)
                self.log(f"Watershed component {len(components)}: {np.sum(component):,} voxels")
        
        return components
    
    def _skeleton_guided_method(self, mask, min_size=100):
        """
        Use 3D skeleton to guide vessel splitting
        """
        self.log("Method 3: Skeleton-Guided Splitting")
        
        # Step 1: Create skeleton
        skeleton = skeletonize(mask)
        self.log(f"Skeleton voxels: {np.sum(skeleton):,}")
        
        # Step 2: Find branch points and endpoints
        # This is a simplified approach - real implementation would need
        # more sophisticated topology analysis
        
        # For now, use skeleton-guided erosion
        # Erode less where skeleton is present
        erosion_mask = binary_erosion(mask, ball(1))
        
        # Preserve skeleton areas
        preserved = erosion_mask | skeleton
        
        # Label and process
        labeled, n_components = label(preserved)
        self.log(f"Skeleton-guided components: {n_components}")
        
        components = []
        for i in range(1, n_components + 1):
            component = (labeled == i)
            
            if np.sum(component) >= min_size:
                # Dilate back within original mask
                dilated = binary_dilation(component, ball(1)) & mask
                components.append(dilated)
                self.log(f"Skeleton component {len(components)}: {np.sum(dilated):,} voxels")
        
        return components
    
    def _distance_transform_method(self, mask, threshold_ratio=0.3, min_size=100):
        """
        Use distance transform peaks to identify separate vessels
        """
        self.log(f"Method 4: Distance Transform (threshold_ratio={threshold_ratio})")
        
        # Step 1: Distance transform
        distance = distance_transform_edt(mask)
        
        # Step 2: Smooth distance transform
        smoothed_distance = gaussian_filter(distance, sigma=1.0)
        
        # Step 3: Threshold at percentage of maximum distance
        max_distance = np.max(smoothed_distance)
        threshold = max_distance * threshold_ratio
        
        thresholded = smoothed_distance > threshold
        self.log(f"Distance threshold: {threshold:.2f} (max: {max_distance:.2f})")
        
        # Step 4: Label thresholded regions
        labeled, n_components = label(thresholded)
        self.log(f"Distance-based components: {n_components}")
        
        # Step 5: Dilate each region back to original mask
        components = []
        for i in range(1, n_components + 1):
            component_seed = (labeled == i)
            
            # Iterative dilation within original mask
            current = component_seed.copy()
            for _ in range(5):  # Maximum 5 dilation iterations
                dilated = binary_dilation(current, ball(1))
                new_current = dilated & mask
                if np.array_equal(new_current, current):
                    break
                current = new_current
            
            if np.sum(current) >= min_size:
                components.append(current)
                self.log(f"Distance component {len(components)}: {np.sum(current):,} voxels")
        
        return components
    
    def analyze_connectivity(self, mask):
        """Analyze vessel connectivity to help choose the best method"""
        self.log("Analyzing vessel connectivity...")
        
        # Basic connectivity analysis
        labeled, n_initial = label(mask)
        self.log(f"Initial connected components: {n_initial}")
        
        # Test erosion effect
        eroded = binary_erosion(mask, ball(1))
        labeled_eroded, n_eroded = label(eroded)
        self.log(f"After 1-voxel erosion: {n_eroded} components")
        
        # Suggest best method based on analysis
        if n_eroded > n_initial:
            self.log("✅ Erosion-dilation recommended (thin bridges detected)")
            return 'erosion_dilation'
        elif np.sum(mask) > 10000:  # Large volumes
            self.log("✅ Watershed recommended (large volume)")
            return 'watershed'
        else:
            self.log("✅ Distance transform recommended (moderate volume)")
            return 'distance_transform'
    
    def split_vessels(self, input_file, output_dir=None, method='auto', **kwargs):
        """
        Main function to split vessels in a NIfTI file
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_dir : str, optional
            Output directory (default: same as input with '_split' suffix)
        method : str
            Splitting method ('auto', 'erosion_dilation', 'watershed', 
            'skeleton_guided', 'distance_transform')
        **kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        list : Paths to saved component files
        """
        
        # Load input file
        original_img, mask = self.load_nifti(input_file)
        
        # Set up output directory
        if output_dir is None:
            input_path = Path(input_file)
            output_dir = input_path.parent / f"{input_path.stem}_split"
        
        basename = Path(input_file).stem.replace('.nii', '')
        
        # Choose method
        if method == 'auto':
            method = self.analyze_connectivity(mask)
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        self.log(f"Using method: {method}")
        
        # Apply splitting method - filter parameters for each method
        method_func = self.methods[method]
        
        # Filter kwargs based on method signature
        import inspect
        sig = inspect.signature(method_func)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        components = method_func(mask, **filtered_kwargs)
        
        if not components:
            self.log("⚠️  No components generated. Saving original mask.")
            components = [mask]
        
        # Save components
        saved_files = self.save_components(
            components, original_img, output_dir, basename, method
        )
        
        # Create summary
        self.log("\n" + "="*60)
        self.log(f"VESSEL SPLITTING COMPLETE")
        self.log(f"Input: {input_file}")
        self.log(f"Method: {method}")
        self.log(f"Components generated: {len(components)}")
        self.log(f"Files saved: {len(saved_files)}")
        self.log(f"Output directory: {output_dir}")
        self.log("="*60)
        
        return saved_files

    def remove_kissing_artifacts(self, input_file, output_file=None, erosion_radius=1, 
                                min_bridge_size=50, preserve_large_components=True):
        """
        Remove kissing artifacts by deleting thin bridging voxels between vessels.
        Returns a single cleaned NIfTI file instead of splitting into multiple files.
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_file : str, optional
            Output file path (default: input_path with '_cleaned' suffix)
        erosion_radius : int
            Radius for erosion to identify bridges
        min_bridge_size : int
            Minimum size of bridges to remove (smaller = more aggressive)
        preserve_large_components : bool
            Whether to preserve the largest connected component
        
        Returns:
        --------
        str : Path to cleaned output file
        """
        
        # Load input file
        original_img, mask = self.load_nifti(input_file)
        
        # Set default output file
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem.replace('.nii', '')}_cleaned.nii.gz"
        
        self.log(f"Removing kissing artifacts from: {input_file}")
        self.log(f"Output will be saved to: {output_file}")
        
        # Step 1: Identify the main vessel structure using erosion
        structuring_element = ball(erosion_radius)
        eroded = binary_erosion(mask, structuring_element)
        self.log(f"After erosion: {np.sum(eroded):,} voxels remaining")
        
        # Step 2: Find connected components after erosion
        labeled_eroded, n_components = label(eroded)
        self.log(f"Found {n_components} components after erosion")
        
        if n_components == 0:
            self.log("⚠️  No components found after erosion. Returning original mask.")
            cleaned_mask = mask
        else:
            # Step 3: Identify main components (preserve large ones)
            component_sizes = []
            for i in range(1, n_components + 1):
                size = np.sum(labeled_eroded == i)
                component_sizes.append((i, size))
            
            # Sort by size, largest first
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            self.log(f"Component sizes: {[size for _, size in component_sizes[:10]]}")
            
            # Step 4: Create preserved components mask
            preserved_components = np.zeros_like(eroded, dtype=bool)
            
            if preserve_large_components:
                # Keep components larger than min_bridge_size
                for comp_id, size in component_sizes:
                    if size >= min_bridge_size:
                        preserved_components |= (labeled_eroded == comp_id)
                        self.log(f"Preserving component {comp_id} with {size:,} voxels")
                    else:
                        self.log(f"Removing small component {comp_id} with {size} voxels")
            else:
                # Keep only the largest component
                if component_sizes:
                    largest_id = component_sizes[0][0]
                    preserved_components = (labeled_eroded == largest_id)
                    self.log(f"Preserving only largest component: {component_sizes[0][1]:,} voxels")
            
            # Step 5: Dilate preserved components back to approximate original size
            self.log("Dilating preserved components back...")
            dilated = preserved_components.copy()
            
            # Iterative dilation within original mask bounds
            for iteration in range(erosion_radius + 2):  # A bit more than erosion to recover
                new_dilated = binary_dilation(dilated, ball(1))
                # Only keep voxels that were in the original mask
                new_dilated = new_dilated & mask
                
                # Check for convergence
                if np.array_equal(new_dilated, dilated):
                    self.log(f"Dilation converged after {iteration + 1} iterations")
                    break
                dilated = new_dilated
            
            cleaned_mask = dilated
        
        # Step 6: Calculate statistics
        original_voxels = np.sum(mask)
        cleaned_voxels = np.sum(cleaned_mask)
        removed_voxels = original_voxels - cleaned_voxels
        
        self.log(f"Original voxels: {original_voxels:,}")
        self.log(f"Cleaned voxels: {cleaned_voxels:,}")
        self.log(f"Removed voxels: {removed_voxels:,} ({removed_voxels/original_voxels*100:.2f}%)")
        
        # Step 7: Save cleaned mask
        cleaned_img = nib.Nifti1Image(
            cleaned_mask.astype(np.uint8), 
            original_img.affine, 
            original_img.header
        )
        
        nib.save(cleaned_img, output_file)
        self.log(f"✅ Saved cleaned vessel mask: {output_file}")
        
        return str(output_file)

    def fix_peanut_kissing_artifacts(self, input_file, output_file=None, 
                                     min_neck_ratio=0.6, analysis_radius=5,
                                     min_vessel_radius=2):
        """
        Fix peanut-shaped kissing artifacts by detecting and cutting at narrow necks.
        
        This method analyzes cross-sectional geometry to find where two cylindrical
        vessels touch, creating a peanut or figure-8 shape, and makes minimal cuts
        to separate them.
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_file : str, optional
            Output file path (default: input_path with '_fixed' suffix)
        min_neck_ratio : float
            Ratio of neck width to vessel width to detect kissing (0-1, smaller = more aggressive)
        analysis_radius : int
            Radius for local shape analysis
        min_vessel_radius : int
            Minimum vessel radius to consider (ignore smaller vessels)
        
        Returns:
        --------
        str : Path to fixed output file
        """
        
        # Load input file
        original_img, mask = self.load_nifti(input_file)
        
        # Set default output file
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem.replace('.nii', '')}_fixed.nii.gz"
        
        self.log(f"Fixing peanut-shaped kissing artifacts from: {input_file}")
        self.log(f"Output will be saved to: {output_file}")
        
        # Step 1: Compute distance transform to find vessel centerlines and radii
        self.log("Computing distance transform...")
        distance = distance_transform_edt(mask)
        
        # Step 2: Find potential kissing artifact locations
        # These are voxels with specific geometric properties
        self.log("Analyzing vessel geometry for kissing artifacts...")
        
        # Create a copy of the mask to modify
        fixed_mask = mask.copy()
        removed_voxels = np.zeros_like(mask, dtype=bool)
        
        # Parameters for analysis
        kernel_size = 2 * analysis_radius + 1
        
        # Analyze each slice in each direction for peanut shapes
        total_removed = 0
        
        # Check XY slices (along Z axis)
        self.log("Analyzing XY slices...")
        for z in range(analysis_radius, mask.shape[2] - analysis_radius):
            if z % 10 == 0:
                self.log(f"  Processing slice {z}/{mask.shape[2]}")
            
            slice_mask = fixed_mask[:, :, z]
            if np.sum(slice_mask) < 100:  # Skip nearly empty slices
                continue
                
            # Find connected components in this slice
            labeled_slice, n_components = label(slice_mask)
            
            for comp_id in range(1, n_components + 1):
                component = (labeled_slice == comp_id)
                
                # Skip small components
                if np.sum(component) < np.pi * min_vessel_radius**2:
                    continue
                
                # Analyze component shape for peanut/figure-8 pattern
                neck_voxels = self._find_neck_voxels(component, distance[:, :, z], 
                                                    min_neck_ratio, min_vessel_radius)
                
                if len(neck_voxels) > 0:
                    # Remove neck voxels from 3D mask
                    for x, y in neck_voxels:
                        # Remove in 3D neighborhood to ensure clean cut
                        for dz in range(-1, 2):
                            if 0 <= z + dz < mask.shape[2]:
                                fixed_mask[x, y, z + dz] = False
                                removed_voxels[x, y, z + dz] = True
                    
                    total_removed += len(neck_voxels)
        
        # Check YZ slices (along X axis)
        self.log("Analyzing YZ slices...")
        for x in range(analysis_radius, mask.shape[0] - analysis_radius):
            if x % 10 == 0:
                self.log(f"  Processing slice {x}/{mask.shape[0]}")
            
            slice_mask = fixed_mask[x, :, :]
            if np.sum(slice_mask) < 100:
                continue
                
            labeled_slice, n_components = label(slice_mask)
            
            for comp_id in range(1, n_components + 1):
                component = (labeled_slice == comp_id)
                
                if np.sum(component) < np.pi * min_vessel_radius**2:
                    continue
                
                neck_voxels = self._find_neck_voxels(component, distance[x, :, :], 
                                                    min_neck_ratio, min_vessel_radius)
                
                if len(neck_voxels) > 0:
                    for y, z in neck_voxels:
                        for dx in range(-1, 2):
                            if 0 <= x + dx < mask.shape[0]:
                                fixed_mask[x + dx, y, z] = False
                                removed_voxels[x + dx, y, z] = True
                    
                    total_removed += len(neck_voxels)
        
        # Check XZ slices (along Y axis)
        self.log("Analyzing XZ slices...")
        for y in range(analysis_radius, mask.shape[1] - analysis_radius):
            if y % 10 == 0:
                self.log(f"  Processing slice {y}/{mask.shape[1]}")
            
            slice_mask = fixed_mask[:, y, :]
            if np.sum(slice_mask) < 100:
                continue
                
            labeled_slice, n_components = label(slice_mask)
            
            for comp_id in range(1, n_components + 1):
                component = (labeled_slice == comp_id)
                
                if np.sum(component) < np.pi * min_vessel_radius**2:
                    continue
                
                neck_voxels = self._find_neck_voxels(component, distance[:, y, :], 
                                                    min_neck_ratio, min_vessel_radius)
                
                if len(neck_voxels) > 0:
                    for x, z in neck_voxels:
                        for dy in range(-1, 2):
                            if 0 <= y + dy < mask.shape[1]:
                                fixed_mask[x, y + dy, z] = False
                                removed_voxels[x, y + dy, z] = True
                    
                    total_removed += len(neck_voxels)
        
        # Step 3: Clean up - ensure we haven't completely disconnected vessels
        self.log("Verifying vessel connectivity...")
        
        # Label final components
        final_labeled, final_n_components = label(fixed_mask)
        original_labeled, original_n_components = label(mask)
        
        self.log(f"Original components: {original_n_components}")
        self.log(f"Final components: {final_n_components}")
        
        # Calculate statistics
        original_voxels = np.sum(mask)
        fixed_voxels = np.sum(fixed_mask)
        removed_count = np.sum(removed_voxels)
        
        self.log(f"Original voxels: {original_voxels:,}")
        self.log(f"Fixed voxels: {fixed_voxels:,}")
        self.log(f"Removed voxels: {removed_count:,} ({removed_count/original_voxels*100:.2f}%)")
        self.log(f"Detected and fixed {total_removed//3} kissing artifact locations")
        
        # Save fixed mask
        fixed_img = nib.Nifti1Image(
            fixed_mask.astype(np.uint8), 
            original_img.affine, 
            original_img.header
        )
        
        nib.save(fixed_img, output_file)
        self.log(f"✅ Saved fixed vessel mask: {output_file}")
        
        return str(output_file)
    
    def _find_neck_voxels(self, component, distance_slice, min_neck_ratio, min_vessel_radius):
        """
        Find neck voxels in a 2D component that indicate kissing artifacts.
        
        A neck is detected when the local width is significantly smaller than
        the adjacent vessel widths, creating a peanut or figure-8 shape.
        """
        neck_voxels = []
        
        # Get component boundary and interior
        # Use 2D erosion for slice analysis
        boundary = component & ~binary_erosion(component, disk(1))
        
        # For each point in the component, check if it's a neck
        coords = np.argwhere(component)
        
        for x, y in coords:
            # Skip if not internal enough
            if distance_slice[x, y] < 2:
                continue
            
            # Analyze local shape by checking multiple directions
            is_neck = self._is_neck_point(component, x, y, distance_slice, 
                                         min_neck_ratio, min_vessel_radius)
            
            if is_neck:
                neck_voxels.append((x, y))
        
        return neck_voxels
    
    def _is_neck_point(self, component, x, y, distance_slice, min_neck_ratio, min_vessel_radius):
        """
        Check if a point is at a neck (narrow connection between wider regions).
        """
        # Get local neighborhood
        neighborhood_size = 7
        half_size = neighborhood_size // 2
        
        # Extract local patch
        x_min = max(0, x - half_size)
        x_max = min(component.shape[0], x + half_size + 1)
        y_min = max(0, y - half_size)
        y_max = min(component.shape[1], y + half_size + 1)
        
        local_patch = component[x_min:x_max, y_min:y_max]
        local_distances = distance_slice[x_min:x_max, y_min:y_max]
        
        if local_patch.size == 0:
            return False
        
        # Check width in multiple directions
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # horizontal, vertical, diagonals
        
        neck_score = 0
        for dx, dy in directions:
            # Measure width at current point
            current_width = self._measure_width_at_point(component, x, y, dx, dy)
            
            if current_width < 2 * min_vessel_radius:
                # Check widths before and after
                before_width = self._measure_width_at_point(
                    component, x - 3*dx, y - 3*dy, dx, dy
                )
                after_width = self._measure_width_at_point(
                    component, x + 3*dx, y + 3*dy, dx, dy
                )
                
                # Check if this is a neck (narrow between wider regions)
                max_adjacent = max(before_width, after_width)
                if max_adjacent > 0 and current_width / max_adjacent < min_neck_ratio:
                    neck_score += 1
        
        # Consider it a neck if multiple directions show neck-like behavior
        return neck_score >= 2
    
    def _measure_width_at_point(self, component, x, y, dx, dy):
        """
        Measure the width of the component at a point perpendicular to direction (dx, dy).
        """
        if x < 0 or y < 0 or x >= component.shape[0] or y >= component.shape[1]:
            return 0
        
        if not component[x, y]:
            return 0
        
        # Perpendicular direction
        perp_dx, perp_dy = -dy, dx
        
        # Count in both directions
        width = 1  # Current point
        
        # Positive direction
        for i in range(1, 20):
            nx, ny = x + i * perp_dx, y + i * perp_dy
            if (0 <= nx < component.shape[0] and 0 <= ny < component.shape[1] 
                and component[nx, ny]):
                width += 1
            else:
                break
        
        # Negative direction  
        for i in range(1, 20):
            nx, ny = x - i * perp_dx, y - i * perp_dy
            if (0 <= nx < component.shape[0] and 0 <= ny < component.shape[1] 
                and component[nx, ny]):
                width += 1
            else:
                break
        
        return width

    def advanced_kissing_artifact_fix(self, input_file, output_file=None,
                                     vesselness_scales=[1, 2, 3, 4],
                                     ellipse_threshold=0.8,
                                     branch_angle_threshold=120,
                                     min_removal_confidence=0.7):
        """
        Advanced kissing artifact detection using:
        1. Multi-scale Hessian vesselness analysis
        2. Medial axis transform + branch point detection
        3. Ellipse decomposition for cross-sectional validation
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_file : str, optional
            Output file path (default: input_path with '_advanced_fixed' suffix)
        vesselness_scales : list
            Scales for Frangi vesselness filter
        ellipse_threshold : float
            Minimum ellipse fitting quality (0-1)
        branch_angle_threshold : float
            Maximum angle (degrees) for valid vessel branching
        min_removal_confidence : float
            Minimum confidence score to remove a kissing artifact
        
        Returns:
        --------
        str : Path to fixed output file
        """
        
        # Load input file
        original_img, mask = self.load_nifti(input_file)
        
        # Set default output file
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem.replace('.nii', '')}_advanced_fixed.nii.gz"
        
        self.log(f"Advanced kissing artifact detection")
        self.log(f"Input: {input_file}")
        self.log(f"Output: {output_file}")
        
        # Create working copy
        fixed_mask = mask.copy()
        
        # Step 1: Multi-scale Hessian Vesselness Analysis
        self.log("\n=== Level 1: Multi-scale Hessian Vesselness Analysis ===")
        vesselness_map = self._compute_multiscale_vesselness(mask, vesselness_scales)
        
        # Step 2: Medial Axis Transform and Branch Point Detection
        self.log("\n=== Level 2: Medial Axis Transform + Branch Analysis ===")
        kissing_candidates = self._detect_kissing_via_medial_axis(mask, vesselness_map, 
                                                                 branch_angle_threshold)
        
        # Step 3: Ellipse Decomposition Validation
        self.log("\n=== Level 3: Cross-sectional Ellipse Decomposition ===")
        validated_artifacts = self._validate_with_ellipse_decomposition(
            mask, kissing_candidates, ellipse_threshold
        )
        
        # Step 4: Remove validated kissing artifacts
        self.log("\n=== Removing Validated Kissing Artifacts ===")
        removal_count = 0
        
        for artifact in validated_artifacts:
            if artifact['confidence'] >= min_removal_confidence:
                # Remove the kissing artifact
                removal_mask = self._create_minimal_cut(mask, artifact['location'], 
                                                       artifact['direction'])
                fixed_mask = fixed_mask & ~removal_mask
                removal_count += 1
                
                self.log(f"Removed artifact at {artifact['location']} "
                        f"(confidence: {artifact['confidence']:.2f})")
        
        # Calculate statistics
        original_voxels = np.sum(mask)
        fixed_voxels = np.sum(fixed_mask)
        removed_voxels = original_voxels - fixed_voxels
        
        self.log(f"\n=== Summary ===")
        self.log(f"Detected candidates: {len(kissing_candidates)}")
        self.log(f"Validated artifacts: {len(validated_artifacts)}")
        self.log(f"Removed artifacts: {removal_count}")
        self.log(f"Original voxels: {original_voxels:,}")
        self.log(f"Fixed voxels: {fixed_voxels:,}")
        self.log(f"Removed voxels: {removed_voxels:,} ({removed_voxels/original_voxels*100:.2f}%)")
        
        # Save result
        fixed_img = nib.Nifti1Image(
            fixed_mask.astype(np.uint8),
            original_img.affine,
            original_img.header
        )
        
        nib.save(fixed_img, output_file)
        self.log(f"✅ Saved advanced fixed vessel mask: {output_file}")
        
        return str(output_file)
    
    def _compute_multiscale_vesselness(self, mask, scales):
        """
        Compute multi-scale Frangi vesselness to identify vessel-like structures
        """
        self.log("Computing multi-scale vesselness...")
        
        # Convert mask to float
        vessel_img = mask.astype(float)
        
        # Compute vesselness at multiple scales
        vesselness_responses = []
        
        for scale in scales:
            self.log(f"  Computing at scale {scale}...")
            
            # Apply Gaussian smoothing
            smoothed = gaussian_filter(vessel_img, sigma=scale)
            
            # Compute Hessian eigenvalues (simplified 3D vesselness)
            # Using gradient-based approximation for speed
            gy, gx, gz = np.gradient(smoothed)
            gyy, gyx, gyz = np.gradient(gy, axis=(0, 1, 2))
            gxy, gxx, gxz = np.gradient(gx, axis=(0, 1, 2))
            gzy, gzx, gzz = np.gradient(gz, axis=(0, 1, 2))
            
            # Compute vesselness response (simplified Frangi)
            vesselness = np.sqrt(gxx**2 + gyy**2 + gzz**2 + 
                               2*gxy**2 + 2*gxz**2 + 2*gyz**2)
            
            vesselness_responses.append(vesselness)
        
        # Combine multi-scale responses
        vesselness_map = np.max(vesselness_responses, axis=0)
        vesselness_map = vesselness_map * mask  # Mask to vessel regions
        
        # Normalize
        if vesselness_map.max() > 0:
            vesselness_map = vesselness_map / vesselness_map.max()
        
        self.log(f"Vesselness computation complete")
        return vesselness_map
    
    def _detect_kissing_via_medial_axis(self, mask, vesselness_map, angle_threshold):
        """
        Use medial axis transform to detect abnormal branch points
        """
        self.log("Computing medial axis transform...")
        
        # Compute 3D skeleton (medial axis is 2D only)
        # Use skeletonize for 3D
        skel = skeletonize(mask)
        distance = distance_transform_edt(mask)
        
        # Find branch points (points with >2 neighbors)
        branch_points = self._find_branch_points_3d(skel)
        self.log(f"Found {len(branch_points)} branch points")
        
        # Analyze each branch point
        kissing_candidates = []
        
        for bp in branch_points:
            x, y, z = bp
            
            # Get local neighborhood
            local_region = skel[max(0, x-3):x+4, max(0, y-3):y+4, max(0, z-3):z+4]
            
            # Analyze branch geometry
            branch_vectors = self._get_branch_vectors(skel, bp)
            
            if len(branch_vectors) >= 3:
                # Check angles between branches
                angles = []
                for i in range(len(branch_vectors)):
                    for j in range(i+1, len(branch_vectors)):
                        angle = self._vector_angle(branch_vectors[i], branch_vectors[j])
                        angles.append(angle)
                
                min_angle = min(angles) if angles else 180
                
                # Kissing artifacts often have acute angles
                if min_angle < angle_threshold:
                    # Check vesselness anomaly
                    local_vesselness = vesselness_map[x, y, z]
                    
                    candidate = {
                        'location': bp,
                        'min_angle': min_angle,
                        'num_branches': len(branch_vectors),
                        'vesselness': local_vesselness,
                        'distance': distance[x, y, z],
                        'confidence': 0.0  # Will be updated
                    }
                    
                    # Initial confidence based on angle and vesselness
                    angle_score = 1.0 - (min_angle / angle_threshold)
                    vesselness_score = 1.0 - local_vesselness
                    candidate['confidence'] = (angle_score + vesselness_score) / 2
                    
                    kissing_candidates.append(candidate)
        
        self.log(f"Identified {len(kissing_candidates)} kissing candidates")
        return kissing_candidates
    
    def _validate_with_ellipse_decomposition(self, mask, candidates, ellipse_threshold):
        """
        Validate kissing artifacts by analyzing cross-sectional shapes
        """
        self.log("Validating candidates with ellipse decomposition...")
        
        validated_artifacts = []
        
        for i, candidate in enumerate(candidates):
            x, y, z = candidate['location']
            
            # Extract cross-sections in three orientations
            xy_slice = mask[:, :, z]
            xz_slice = mask[:, y, :]
            yz_slice = mask[x, :, :]
            
            # Analyze each slice
            ellipse_scores = []
            
            for slice_data, orientation in [(xy_slice, 'xy'), (xz_slice, 'xz'), (yz_slice, 'yz')]:
                score = self._analyze_slice_ellipses(slice_data, candidate['location'], orientation)
                ellipse_scores.append(score)
            
            # Update confidence based on ellipse analysis
            max_ellipse_score = max(ellipse_scores)
            candidate['ellipse_score'] = max_ellipse_score
            candidate['confidence'] = (candidate['confidence'] + max_ellipse_score) / 2
            
            # Determine optimal cutting direction
            best_orientation_idx = ellipse_scores.index(max_ellipse_score)
            orientations = ['xy', 'xz', 'yz']
            candidate['direction'] = orientations[best_orientation_idx]
            
            if max_ellipse_score > ellipse_threshold:
                validated_artifacts.append(candidate)
                self.log(f"  Validated artifact {i+1}: confidence={candidate['confidence']:.2f}")
        
        self.log(f"Validated {len(validated_artifacts)} artifacts")
        return validated_artifacts
    
    def _analyze_slice_ellipses(self, slice_data, location_3d, orientation):
        """
        Analyze if a 2D slice contains multiple ellipses (peanut shape)
        """
        # Get the 2D coordinates based on orientation
        if orientation == 'xy':
            center_2d = (location_3d[0], location_3d[1])
        elif orientation == 'xz':
            center_2d = (location_3d[0], location_3d[2])
        else:  # yz
            center_2d = (location_3d[1], location_3d[2])
        
        # Extract local region around the candidate point
        roi_size = 30
        x_min = max(0, center_2d[0] - roi_size)
        x_max = min(slice_data.shape[0], center_2d[0] + roi_size)
        y_min = max(0, center_2d[1] - roi_size)
        y_max = min(slice_data.shape[1], center_2d[1] + roi_size)
        
        roi = slice_data[x_min:x_max, y_min:y_max]
        
        if np.sum(roi) < 10:  # Too small
            return 0.0
        
        # Find contours
        roi_uint8 = (roi * 255).astype(np.uint8)
        contours, _ = cv2.findContours(roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:  # Need at least 5 points for ellipse
            return 0.0
        
        # Try to fit single ellipse
        try:
            single_ellipse = cv2.fitEllipse(largest_contour)
            
            # Check how well single ellipse fits
            ellipse_mask = np.zeros_like(roi_uint8)
            cv2.ellipse(ellipse_mask, single_ellipse, 255, -1)
            
            # Calculate overlap
            intersection = np.sum((ellipse_mask > 0) & (roi_uint8 > 0))
            union = np.sum((ellipse_mask > 0) | (roi_uint8 > 0))
            single_iou = intersection / union if union > 0 else 0
            
            # If single ellipse fits well, not a kissing artifact
            if single_iou > 0.9:
                return 0.0
            
            # Try to detect if it's better fit by two ellipses
            # Use convexity defects as hint for peanut shape
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            if len(hull) > 3 and len(largest_contour) > 10:
                defects = cv2.convexityDefects(largest_contour, hull)
                
                if defects is not None and len(defects) > 0:
                    # Large convexity defects suggest peanut shape
                    max_defect_depth = max(defects[:, :, 3]) / 256.0
                    
                    # Score based on defect depth and poor single ellipse fit
                    peanut_score = (1 - single_iou) * min(1.0, max_defect_depth / 10)
                    return peanut_score
            
        except:
            pass
        
        return 0.0
    
    def _find_branch_points_3d(self, skeleton):
        """
        Find branch points in 3D skeleton
        """
        # Count neighbors for each skeleton point
        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 0
        
        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
        
        # Branch points have >2 neighbors
        branch_mask = (skeleton > 0) & (neighbor_count > 2)
        
        return list(zip(*np.where(branch_mask)))
    
    def _get_branch_vectors(self, skeleton, branch_point):
        """
        Get unit vectors pointing along each branch from a branch point
        """
        x, y, z = branch_point
        vectors = []
        
        # Search in 26-neighborhood for skeleton continuations
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    if (0 <= nx < skeleton.shape[0] and
                        0 <= ny < skeleton.shape[1] and
                        0 <= nz < skeleton.shape[2] and
                        skeleton[nx, ny, nz]):
                        
                        # Trace along this direction to get branch vector
                        vector = self._trace_branch_direction(skeleton, branch_point, (dx, dy, dz))
                        if vector is not None:
                            vectors.append(vector)
        
        return vectors
    
    def _trace_branch_direction(self, skeleton, start_point, initial_direction, max_steps=10):
        """
        Trace along skeleton in given direction to determine branch vector
        """
        x, y, z = start_point
        dx, dy, dz = initial_direction
        
        points = [(x, y, z)]
        
        for _ in range(max_steps):
            x, y, z = x + dx, y + dy, z + dz
            
            if not (0 <= x < skeleton.shape[0] and
                   0 <= y < skeleton.shape[1] and
                   0 <= z < skeleton.shape[2]):
                break
            
            if not skeleton[x, y, z]:
                break
            
            points.append((x, y, z))
        
        if len(points) < 3:
            return None
        
        # Compute average direction vector
        vector = np.array(points[-1]) - np.array(points[0])
        norm = np.linalg.norm(vector)
        
        if norm > 0:
            return vector / norm
        return None
    
    def _vector_angle(self, v1, v2):
        """
        Compute angle between two vectors in degrees
        """
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _create_minimal_cut(self, mask, location, direction, cut_radius=2):
        """
        Create minimal cut mask at kissing artifact location
        """
        x, y, z = location
        cut_mask = np.zeros_like(mask, dtype=bool)
        
        # Create cut plane based on direction
        if direction == 'xy':
            # Cut in Z direction
            for dz in range(-cut_radius, cut_radius + 1):
                if 0 <= z + dz < mask.shape[2]:
                    cut_mask[max(0, x-cut_radius):min(mask.shape[0], x+cut_radius+1),
                            max(0, y-cut_radius):min(mask.shape[1], y+cut_radius+1),
                            z + dz] = True
        
        elif direction == 'xz':
            # Cut in Y direction
            for dy in range(-cut_radius, cut_radius + 1):
                if 0 <= y + dy < mask.shape[1]:
                    cut_mask[max(0, x-cut_radius):min(mask.shape[0], x+cut_radius+1),
                            y + dy,
                            max(0, z-cut_radius):min(mask.shape[2], z+cut_radius+1)] = True
        
        else:  # yz
            # Cut in X direction
            for dx in range(-cut_radius, cut_radius + 1):
                if 0 <= x + dx < mask.shape[0]:
                    cut_mask[x + dx,
                            max(0, y-cut_radius):min(mask.shape[1], y+cut_radius+1),
                            max(0, z-cut_radius):min(mask.shape[2], z+cut_radius+1)] = True
        
        # Only cut where vessel exists
        return cut_mask & mask

    def dual_maxima_kissing_fix(self, input_file, output_file=None,
                                min_peak_distance=4, peak_prominence=2,
                                cut_depth_ratio=0.5, min_vessel_size=100):
        """
        Fix kissing artifacts by detecting dual maxima in distance transform.
        
        When two cylindrical vessels touch, the distance transform shows two local
        maxima (centers of cylinders) with a saddle point between them.
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_file : str, optional
            Output file path (default: input_path with '_dual_fixed' suffix)
        min_peak_distance : int
            Minimum distance between vessel centers to consider as kissing
        peak_prominence : float
            Minimum prominence of peaks in distance transform
        cut_depth_ratio : float
            How deep to cut at the saddle point (0-1)
        min_vessel_size : int
            Minimum vessel component size to process
        
        Returns:
        --------
        str : Path to fixed output file
        """
        
        # Load input file
        original_img, mask = self.load_nifti(input_file)
        
        # Set default output file
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem.replace('.nii', '')}_dual_fixed.nii.gz"
        
        self.log(f"Dual maxima kissing artifact detection")
        self.log(f"Input: {input_file}")
        self.log(f"Output: {output_file}")
        
        # Compute distance transform
        self.log("Computing distance transform...")
        distance = distance_transform_edt(mask)
        
        # Create working copy
        fixed_mask = mask.copy()
        total_removed = 0
        
        # Process each connected component separately
        labeled_mask, n_components = label(mask)
        
        for comp_id in range(1, n_components + 1):
            component_mask = (labeled_mask == comp_id)
            
            if np.sum(component_mask) < min_vessel_size:
                continue
                
            self.log(f"\nAnalyzing component {comp_id} ({np.sum(component_mask)} voxels)")
            
            # Get distance transform for this component
            comp_distance = distance * component_mask
            
            # Find saddle points using multiple strategies
            saddle_points = self._find_saddle_points_multi_strategy(
                component_mask, comp_distance, min_peak_distance, peak_prominence
            )
            
            self.log(f"Found {len(saddle_points)} potential kissing sites")
            
            # Remove kissing artifacts at saddle points
            for saddle in saddle_points:
                cut_mask = self._create_saddle_cut(
                    component_mask, comp_distance, saddle, cut_depth_ratio
                )
                
                fixed_mask = fixed_mask & ~cut_mask
                removed = np.sum(cut_mask)
                total_removed += removed
                
                self.log(f"  Removed {removed} voxels at {saddle['location']}")
        
        # Calculate statistics
        original_voxels = np.sum(mask)
        fixed_voxels = np.sum(fixed_mask)
        
        self.log(f"\n=== Summary ===")
        self.log(f"Original voxels: {original_voxels:,}")
        self.log(f"Fixed voxels: {fixed_voxels:,}")
        self.log(f"Removed voxels: {total_removed:,} ({total_removed/original_voxels*100:.2f}%)")
        
        # Save result
        fixed_img = nib.Nifti1Image(
            fixed_mask.astype(np.uint8),
            original_img.affine,
            original_img.header
        )
        
        nib.save(fixed_img, output_file)
        self.log(f"✅ Saved dual maxima fixed vessel mask: {output_file}")
        
        return str(output_file)
    
    def _find_saddle_points_multi_strategy(self, mask, distance, min_peak_distance, peak_prominence):
        """
        Find saddle points using multiple detection strategies
        """
        saddle_points = []
        
        # Strategy 1: Analyze cross-sectional slices
        self.log("  Strategy 1: Cross-sectional analysis...")
        for axis in range(3):
            for i in range(distance.shape[axis]):
                if axis == 0:
                    slice_mask = mask[i, :, :]
                    slice_dist = distance[i, :, :]
                elif axis == 1:
                    slice_mask = mask[:, i, :]
                    slice_dist = distance[:, i, :]
                else:
                    slice_mask = mask[:, :, i]
                    slice_dist = distance[:, :, i]
                
                if np.sum(slice_mask) < 50:  # Skip small slices
                    continue
                
                # Find dual maxima in this slice
                saddles_2d = self._find_dual_maxima_2d(slice_mask, slice_dist, 
                                                       min_peak_distance, peak_prominence)
                
                for s2d in saddles_2d:
                    # Convert to 3D coordinates
                    if axis == 0:
                        location = (i, s2d[0], s2d[1])
                    elif axis == 1:
                        location = (s2d[0], i, s2d[1])
                    else:
                        location = (s2d[0], s2d[1], i)
                    
                    saddle_points.append({
                        'location': location,
                        'axis': axis,
                        'confidence': 0.8
                    })
        
        # Strategy 2: 3D ridge detection
        self.log("  Strategy 2: 3D ridge detection...")
        ridges = self._detect_3d_ridges(mask, distance)
        for ridge in ridges:
            saddle_points.append({
                'location': ridge,
                'axis': -1,  # 3D detection
                'confidence': 0.9
            })
        
        # Remove duplicates
        unique_saddles = []
        for saddle in saddle_points:
            is_duplicate = False
            for unique in unique_saddles:
                dist = np.sqrt(sum((saddle['location'][i] - unique['location'][i])**2 
                                 for i in range(3)))
                if dist < 3:  # Within 3 voxels
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_saddles.append(saddle)
        
        return unique_saddles
    
    def _find_dual_maxima_2d(self, mask, distance, min_peak_distance, peak_prominence):
        """
        Find locations where two local maxima exist close together (peanut shape)
        """
        from scipy.ndimage import maximum_filter, label
        from scipy.signal import find_peaks
        
        saddle_points = []
        
        # Find local maxima
        local_max = (distance == maximum_filter(distance, size=5)) & mask
        max_coords = np.argwhere(local_max)
        
        if len(max_coords) < 2:
            return saddle_points
        
        # Check each pair of maxima
        for i in range(len(max_coords)):
            for j in range(i + 1, len(max_coords)):
                p1, p2 = max_coords[i], max_coords[j]
                dist = np.sqrt(np.sum((p1 - p2)**2))
                
                if min_peak_distance <= dist <= min_peak_distance * 4:
                    # Check if there's a saddle point between them
                    # Sample along the line between peaks
                    n_samples = int(dist * 2)
                    line_points = []
                    for t in np.linspace(0, 1, n_samples):
                        point = p1 * (1 - t) + p2 * t
                        point = point.astype(int)
                        if (0 <= point[0] < distance.shape[0] and 
                            0 <= point[1] < distance.shape[1]):
                            line_points.append((point, distance[point[0], point[1]]))
                    
                    if len(line_points) > 3:
                        # Find minimum along the line
                        line_distances = [d for _, d in line_points]
                        min_idx = np.argmin(line_distances)
                        
                        # Check if it's a significant saddle
                        min_dist = line_distances[min_idx]
                        peak1_dist = distance[p1[0], p1[1]]
                        peak2_dist = distance[p2[0], p2[1]]
                        
                        prominence1 = peak1_dist - min_dist
                        prominence2 = peak2_dist - min_dist
                        
                        if (prominence1 >= peak_prominence and 
                            prominence2 >= peak_prominence):
                            saddle_point = line_points[min_idx][0]
                            saddle_points.append(saddle_point)
        
        return saddle_points
    
    def _detect_3d_ridges(self, mask, distance, threshold=0.7):
        """
        Detect 3D ridges where vessels meet (simplified version)
        """
        ridges = []
        
        # Compute gradient magnitude
        gy, gx, gz = np.gradient(distance)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Normalize
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        
        # Find low gradient regions with high distance (potential saddles)
        saddle_candidates = (grad_mag < 0.3) & (distance > threshold * distance.max()) & mask
        
        # Get connected components of candidates
        labeled_candidates, n_candidates = label(saddle_candidates)
        
        for i in range(1, min(n_candidates + 1, 20)):  # Limit to 20 candidates
            component = (labeled_candidates == i)
            if 5 < np.sum(component) < 100:  # Reasonable size for saddle region
                # Get centroid
                coords = np.argwhere(component)
                centroid = coords.mean(axis=0).astype(int)
                ridges.append(tuple(centroid))
        
        return ridges
    
    def _create_saddle_cut(self, mask, distance, saddle_info, cut_depth_ratio):
        """
        Create a cut mask at the saddle point
        """
        location = saddle_info['location']
        x, y, z = location
        
        # Determine cut radius based on local distance value
        local_distance = distance[x, y, z]
        cut_radius = max(2, int(local_distance * cut_depth_ratio))
        
        # Create spherical cut
        cut_mask = np.zeros_like(mask, dtype=bool)
        
        for dx in range(-cut_radius, cut_radius + 1):
            for dy in range(-cut_radius, cut_radius + 1):
                for dz in range(-cut_radius, cut_radius + 1):
                    if dx*dx + dy*dy + dz*dz <= cut_radius*cut_radius:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < mask.shape[0] and 
                            0 <= ny < mask.shape[1] and 
                            0 <= nz < mask.shape[2]):
                            cut_mask[nx, ny, nz] = True
        
        # Only cut where vessel exists
        return cut_mask & mask

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Split fused vessels or remove kissing artifacts in cerebrovascular segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove kissing artifacts (single cleaned file)
  python vessel_splitter.py input.nii.gz --remove-artifacts
  
  # Fix peanut-shaped kissing artifacts (geometry-aware method)
  python vessel_splitter.py input.nii.gz --remove-artifacts --peanut-fix
  
  # Advanced 3-level kissing artifact detection (most sophisticated)
  python vessel_splitter.py input.nii.gz --remove-artifacts --advanced-fix
  
  # Dual maxima detection for peanut-shaped artifacts
  python vessel_splitter.py input.nii.gz --remove-artifacts --dual-maxima
  
  # Remove artifacts with custom parameters
  python vessel_splitter.py input.nii.gz --remove-artifacts --erosion-radius 2 --min-bridge-size 30
  
  # Split vessels into separate files (original behavior)
  python vessel_splitter.py input.nii.gz --method watershed
  
  # Auto-detect best splitting method
  python vessel_splitter.py input.nii.gz
        """
    )
    
    parser.add_argument('input_file', help='Input NIfTI file (.nii.gz)')
    parser.add_argument('--output-dir', help='Output directory (for splitting) or output file (for artifact removal)')
    parser.add_argument('--remove-artifacts', action='store_true',
                        help='Remove kissing artifacts and output single cleaned file')
    parser.add_argument('--method', default='auto',
                        choices=['auto', 'erosion_dilation', 'watershed', 
                                'skeleton_guided', 'distance_transform'],
                        help='Splitting method (ignored if --remove-artifacts is used)')
    parser.add_argument('--erosion-radius', type=int, default=1,
                        help='Erosion radius for erosion_dilation method or artifact removal')
    parser.add_argument('--min-distance', type=int, default=3,
                        help='Minimum distance for watershed seeds')
    parser.add_argument('--min-size', type=int, default=100,
                        help='Minimum component size to keep (for splitting)')
    parser.add_argument('--min-bridge-size', type=int, default=50,
                        help='Minimum bridge size to remove (for artifact removal)')
    parser.add_argument('--threshold-ratio', type=float, default=0.3,
                        help='Distance threshold ratio for distance_transform method')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--peanut-fix', action='store_true',
                        help='Use peanut-shaped kissing artifact detection (for --remove-artifacts)')
    parser.add_argument('--neck-ratio', type=float, default=0.6,
                        help='Ratio threshold for neck detection (0-1, smaller = more aggressive)')
    parser.add_argument('--analysis-radius', type=int, default=5,
                        help='Radius for local shape analysis')
    parser.add_argument('--min-vessel-radius', type=int, default=2,
                        help='Minimum vessel radius to consider')
    parser.add_argument('--advanced-fix', action='store_true',
                        help='Use advanced 3-level kissing artifact detection (for --remove-artifacts)')
    parser.add_argument('--vesselness-scales', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Scales for multi-scale vesselness filter')
    parser.add_argument('--ellipse-threshold', type=float, default=0.8,
                        help='Threshold for ellipse decomposition validation (0-1)')
    parser.add_argument('--branch-angle-threshold', type=float, default=120,
                        help='Maximum angle (degrees) for valid vessel branching')
    parser.add_argument('--min-removal-confidence', type=float, default=0.7,
                        help='Minimum confidence score to remove a kissing artifact')
    parser.add_argument('--dual-maxima', action='store_true',
                        help='Use dual maxima detection for kissing artifacts (for --remove-artifacts)')
    parser.add_argument('--min-peak-distance', type=int, default=4,
                        help='Minimum distance between vessel centers')
    parser.add_argument('--peak-prominence', type=float, default=2.0,
                        help='Minimum prominence of distance transform peaks')
    parser.add_argument('--cut-depth-ratio', type=float, default=0.5,
                        help='Cut depth at saddle points (0-1)')
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = VesselSplitter(verbose=not args.quiet)
    
    # Run kissing artifact removal or vessel splitting
    try:
        if args.remove_artifacts:
            # Remove kissing artifacts and output single cleaned file
            if args.peanut_fix:
                # Use the new peanut-shaped kissing artifact fix method
                output_file = splitter.fix_peanut_kissing_artifacts(
                    args.input_file,
                    output_file=args.output_dir,  # Can specify output file path
                    min_neck_ratio=args.neck_ratio,
                    analysis_radius=args.analysis_radius,
                    min_vessel_radius=args.min_vessel_radius
                )
                
                print(f"\n✅ Success! Peanut-shaped kissing artifacts fixed.")
                print(f"  🥜 Fixed file: {output_file}")
            elif args.advanced_fix:
                # Use advanced 3-level detection method
                output_file = splitter.advanced_kissing_artifact_fix(
                    args.input_file,
                    output_file=args.output_dir,
                    vesselness_scales=args.vesselness_scales,
                    ellipse_threshold=args.ellipse_threshold,
                    branch_angle_threshold=args.branch_angle_threshold,
                    min_removal_confidence=args.min_removal_confidence
                )
                
                print(f"\n✅ Success! Advanced kissing artifact detection complete.")
                print(f"  🔬 Fixed file: {output_file}")
            elif args.dual_maxima:
                # Use dual maxima detection method
                output_file = splitter.dual_maxima_kissing_fix(
                    args.input_file,
                    output_file=args.output_dir,
                    min_peak_distance=args.min_peak_distance,
                    peak_prominence=args.peak_prominence,
                    cut_depth_ratio=args.cut_depth_ratio,
                    min_vessel_size=args.min_size
                )
                
                print(f"\n✅ Success! Dual maxima kissing artifacts fixed.")
                print(f"  🎯 Fixed file: {output_file}")
            else:
                # Use the original erosion-based method
                output_file = splitter.remove_kissing_artifacts(
                    args.input_file,
                    output_file=args.output_dir,  # Can specify output file path
                    erosion_radius=args.erosion_radius,
                    min_bridge_size=args.min_bridge_size,
                    preserve_large_components=True
                )
                
                print(f"\n✅ Success! Kissing artifacts removed.")
                print(f"  🧽 Cleaned file: {output_file}")
            
        else:
            # Original vessel splitting behavior
            method_params = {
                'erosion_radius': args.erosion_radius,
                'min_distance': args.min_distance,
                'min_size': args.min_size,
                'threshold_ratio': args.threshold_ratio
            }
            
            saved_files = splitter.split_vessels(
                args.input_file,
                output_dir=args.output_dir,
                method=args.method,
                **method_params
            )
            
            print(f"\n✅ Success! Generated {len(saved_files)} component files.")
            for f in saved_files:
                print(f"  📁 {f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 