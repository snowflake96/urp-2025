"""
NIfTI Aneurysm Detection and Localization
Finds aneurysm locations using NIfTI labels and doctor's diagnosis information.
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
import json
import argparse
from pathlib import Path


class AneurysmDetector:
    """Class for detecting and localizing aneurysms in NIfTI files."""
    
    def __init__(self, nifti_path, diagnosis_info=None):
        """Initialize aneurysm detector."""
        self.nifti_path = nifti_path
        self.diagnosis_info = diagnosis_info or {}
        self.nii_img = None
        self.image_data = None
        self.header = None
        self.affine = None
        self._load_file()
    
    def _load_file(self):
        """Load the NIfTI file."""
        try:
            self.nii_img = nib.load(self.nifti_path)
            self.image_data = self.nii_img.get_fdata()
            self.header = self.nii_img.header
            self.affine = self.nii_img.affine
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file {self.nifti_path}: {str(e)}")
    
    def find_aneurysms(self):
        """Find all aneurysms in the NIfTI file."""
        aneurysm_labels = self.get_aneurysm_labels()
        aneurysms = {}
        aneurysm_counter = 1
        
        for label in aneurysm_labels:
            # Check if this label contains multiple connected components (separate aneurysms)
            if isinstance(label, float):
                label_mask = np.abs(self.image_data - label) < 1e-6
            else:
                label_mask = (self.image_data == label)
            
            # Try to separate potential aneurysms using morphological operations
            # Apply erosion to potentially separate connected aneurysms
            from scipy.ndimage import binary_erosion, binary_dilation
            
            # First, try with original mask
            labeled_array, num_components = ndimage.label(label_mask)
            
            # If only one component but we expect more, try morphological separation
            expected_count = self.diagnosis_info.get('expected_aneurysm_count', 1)
            if num_components == 1 and expected_count > 1:
                # Apply erosion to break connections
                eroded = binary_erosion(label_mask, iterations=2)
                labeled_eroded, num_eroded = ndimage.label(eroded)
                
                if num_eroded > 1:
                    # Get component sizes and keep only the largest ones
                    component_sizes = []
                    for comp_id in range(1, num_eroded + 1):
                        comp_mask = (labeled_eroded == comp_id)
                        size = np.sum(comp_mask)
                        component_sizes.append((comp_id, size))
                    
                    # Sort by size and keep only the top components
                    component_sizes.sort(key=lambda x: x[1], reverse=True)
                    max_components = min(expected_count, len(component_sizes))
                    
                    # Only keep components that are reasonably large
                    min_size = np.sum(label_mask) * 0.05  # At least 5% of total volume
                    kept_components = []
                    for comp_id, size in component_sizes[:max_components]:
                        if size >= min_size:
                            kept_components.append(comp_id)
                    
                    if len(kept_components) > 1:
                        # Dilate each kept component back to approximate original size
                        separated_mask = np.zeros_like(label_mask)
                        component_masks = []
                        
                        # First, dilate all components without assigning labels yet
                        for comp_id in kept_components:
                            comp_mask = (labeled_eroded == comp_id)
                            # Dilate back
                            dilated = binary_dilation(comp_mask, iterations=3)
                            # Only keep parts that were in original mask
                            dilated = dilated & label_mask
                            component_masks.append(dilated)
                        
                        # Handle overlapping regions by assigning to nearest component
                        for new_comp_id, comp_mask in enumerate(component_masks, 1):
                            # For non-overlapping regions, assign directly
                            non_overlapping = comp_mask.copy()
                            for other_mask in component_masks:
                                if not np.array_equal(other_mask, comp_mask):
                                    non_overlapping = non_overlapping & ~other_mask
                            
                            separated_mask[non_overlapping] = new_comp_id
                            
                            # For overlapping regions, assign to the component with closest eroded centroid
                            overlapping = comp_mask & (separated_mask == 0) & label_mask
                            if np.any(overlapping):
                                separated_mask[overlapping] = new_comp_id
                        
                        labeled_array = separated_mask
                        num_components = len(kept_components)
                        print(f"  Morphological separation found {num_components} significant components")
                    else:
                        print(f"  Morphological separation found {num_eroded} components but only {len(kept_components)} were significant enough")
            
            if num_components > 1:
                # Multiple components - treat each as separate aneurysm
                for component_id in range(1, num_components + 1):
                    component_mask = (labeled_array == component_id)
                    component_volume = np.sum(component_mask)
                    
                    # Only include components with reasonable size
                    if component_volume >= 20:  # Minimum size threshold
                        aneurysm_info = self.analyze_aneurysm_component(label, component_mask)
                        aneurysms[f"aneurysm_{aneurysm_counter}"] = aneurysm_info
                        aneurysm_counter += 1
            else:
                # Single component
                aneurysm_info = self.analyze_aneurysm(label)
                aneurysms[f"aneurysm_{aneurysm_counter}"] = aneurysm_info
                aneurysm_counter += 1
        
        return {
            'total_aneurysms_found': len(aneurysms),
            'aneurysm_labels': aneurysm_labels,
            'aneurysms': aneurysms,
            'diagnosis_info': self.diagnosis_info
        }
    
    def get_aneurysm_labels(self):
        """Identify potential aneurysm labels."""
        unique_labels = np.unique(self.image_data)
        unique_labels = unique_labels[unique_labels > 1e-6]  # Filter out background (near 0)
        
        if 'aneurysm_labels' in self.diagnosis_info and self.diagnosis_info['aneurysm_labels'] is not None:
            specified_labels = self.diagnosis_info['aneurysm_labels']
            return [label for label in specified_labels if label in unique_labels]
        
        potential_labels = []
        for label in unique_labels:
            # Handle floating point precision issues
            if isinstance(label, float):
                label_mask = np.abs(self.image_data - label) < 1e-6
            else:
                label_mask = (self.image_data == label)
            label_volume = np.sum(label_mask)
            
            if 10 <= label_volume <= self.image_data.size * 0.1:
                if self._is_aneurysm_like(label_mask, label):
                    potential_labels.append(label)  # Keep original float value
        
        return potential_labels
    
    def _is_aneurysm_like(self, mask, label_value):
        """Check if a region has aneurysm-like characteristics."""
        volume = np.sum(mask)
        if volume < 20:
            return False
        
        labeled_array, num_components = ndimage.label(mask)
        if num_components > 3:
            return False
        
        sphericity = self._calculate_sphericity(mask)
        if sphericity < 0.1:
            return False
        
        return True
    
    def _calculate_sphericity(self, mask):
        """Calculate sphericity of a 3D region."""
        try:
            volume = np.sum(mask)
            eroded = ndimage.binary_erosion(mask)
            surface_voxels = np.sum(mask) - np.sum(eroded)
            
            if surface_voxels == 0:
                return 0
            
            ideal_surface = (volume ** (2/3)) * (np.pi ** (1/3)) * (6 ** (2/3))
            sphericity = ideal_surface / surface_voxels if surface_voxels > 0 else 0
            
            return min(sphericity, 1.0)
        except:
            return 0.5
    
    def analyze_aneurysm_component(self, label_value, component_mask):
        """Analyze a specific aneurysm component."""
        mask = component_mask
        
        volume_voxels = int(np.sum(mask))
        try:
            centroid = ndimage.center_of_mass(mask)
            # Handle NaN values
            if np.any(np.isnan(centroid)):
                centroid = [0.0, 0.0, 0.0]
        except:
            centroid = [0.0, 0.0, 0.0]
        
        if self.affine is not None:
            try:
                centroid_world = self._voxel_to_world(centroid)
            except:
                centroid_world = centroid
        else:
            centroid_world = centroid
        
        labeled_array, num_components = ndimage.label(mask)
        
        coords = np.where(mask)
        if len(coords[0]) > 0:
            bbox = {
                'min_coords': [int(np.min(coord)) for coord in coords],
                'max_coords': [int(np.max(coord)) for coord in coords],
                'size': [int(np.max(coord) - np.min(coord) + 1) for coord in coords]
            }
        else:
            bbox = None
        
        sphericity = self._calculate_sphericity(mask)
        voxel_volume_mm3 = self._get_voxel_volume()
        physical_volume = volume_voxels * voxel_volume_mm3 if voxel_volume_mm3 else None
        
        return {
            'label_value': float(label_value),
            'volume_voxels': volume_voxels,
            'physical_volume_mm3': physical_volume,
            'centroid_voxel': [float(coord) for coord in centroid],
            'centroid_world': [float(coord) for coord in centroid_world],
            'connected_components': int(num_components),
            'bounding_box': bbox,
            'sphericity': float(sphericity)
        }

    def analyze_aneurysm(self, label_value):
        """Analyze a specific aneurysm label."""
        # Handle floating point precision issues
        if isinstance(label_value, float):
            mask = np.abs(self.image_data - label_value) < 1e-6
        else:
            mask = (self.image_data == label_value)
        
        volume_voxels = int(np.sum(mask))
        try:
            centroid = ndimage.center_of_mass(mask)
            # Handle NaN values
            if np.any(np.isnan(centroid)):
                centroid = [0.0, 0.0, 0.0]
        except:
            centroid = [0.0, 0.0, 0.0]
        
        if self.affine is not None:
            try:
                centroid_world = self._voxel_to_world(centroid)
            except:
                centroid_world = centroid
        else:
            centroid_world = centroid
        
        labeled_array, num_components = ndimage.label(mask)
        
        coords = np.where(mask)
        if len(coords[0]) > 0:
            bbox = {
                'min_coords': [int(np.min(coord)) for coord in coords],
                'max_coords': [int(np.max(coord)) for coord in coords],
                'size': [int(np.max(coord) - np.min(coord) + 1) for coord in coords]
            }
        else:
            bbox = None
        
        sphericity = self._calculate_sphericity(mask)
        voxel_volume_mm3 = self._get_voxel_volume()
        physical_volume = volume_voxels * voxel_volume_mm3 if voxel_volume_mm3 else None
        
        return {
            'label_value': int(label_value),
            'volume_voxels': volume_voxels,
            'physical_volume_mm3': physical_volume,
            'centroid_voxel': [float(coord) for coord in centroid],
            'centroid_world': [float(coord) for coord in centroid_world],
            'connected_components': int(num_components),
            'bounding_box': bbox,
            'sphericity': float(sphericity)
        }
    
    def _voxel_to_world(self, voxel_coords):
        """Convert voxel coordinates to world coordinates."""
        voxel_homo = np.array(list(voxel_coords) + [1])
        world_coords = self.affine.dot(voxel_homo)[:3]
        return tuple(world_coords)
    
    def _get_voxel_volume(self):
        """Get the volume of a single voxel in mm³."""
        try:
            pixdim = self.header['pixdim'][1:4]
            return float(np.prod(pixdim))
        except:
            return None


def find_aneurysms(nifti_path, diagnosis_file=None, output_coords=None, verbose=True):
    """Find aneurysms in a NIfTI file."""
    diagnosis_info = {}
    if diagnosis_file and Path(diagnosis_file).exists():
        with open(diagnosis_file, 'r') as f:
            diagnosis_info = json.load(f)
    
    detector = AneurysmDetector(nifti_path, diagnosis_info)
    results = detector.find_aneurysms()
    
    if verbose:
        print(f"\n=== Aneurysm Detection Report: {nifti_path} ===")
        print(f"Total aneurysms found: {results['total_aneurysms_found']}")
        print(f"Aneurysm labels: {results['aneurysm_labels']}")
        
        for aneurysm_id, info in results['aneurysms'].items():
            print(f"\n{aneurysm_id.upper()}:")
            print(f"  Label: {info['label_value']}")
            print(f"  Volume: {info['volume_voxels']} voxels")
            if info['physical_volume_mm3']:
                print(f"  Physical Volume: {info['physical_volume_mm3']:.2f} mm³")
            print(f"  Centroid (voxel): ({info['centroid_voxel'][0]:.1f}, {info['centroid_voxel'][1]:.1f}, {info['centroid_voxel'][2]:.1f})")
            print(f"  Centroid (world): ({info['centroid_world'][0]:.1f}, {info['centroid_world'][1]:.1f}, {info['centroid_world'][2]:.1f})")
    
    if output_coords:
        with open(output_coords, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_coords}")
    
    return results


def main():
    """Command line interface for aneurysm detection."""
    parser = argparse.ArgumentParser(description='Find aneurysms in NIfTI files')
    parser.add_argument('input', help='Input NIfTI file path')
    parser.add_argument('-d', '--diagnosis', help='Diagnosis JSON file path')
    parser.add_argument('-o', '--output', help='Output coordinates JSON file path')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        results = find_aneurysms(
            nifti_path=args.input,
            diagnosis_file=args.diagnosis,
            output_coords=args.output,
            verbose=not args.quiet
        )
        
        if results['total_aneurysms_found'] == 0:
            print("No aneurysms found in the image.")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
