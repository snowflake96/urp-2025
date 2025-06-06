"""
NIfTI Validation and Analysis
Validates NIfTI files and extracts segment information, point counts, labels, and metadata.
"""

import nibabel as nib
import numpy as np
from collections import Counter
import json
import argparse
from pathlib import Path
import warnings


class NIfTIValidator:
    """Class for validating and analyzing NIfTI files."""
    
    def __init__(self, nifti_path):
        """
        Initialize validator with NIfTI file path.
        
        Args:
            nifti_path (str): Path to the NIfTI file
        """
        self.nifti_path = nifti_path
        self.nii_img = None
        self.image_data = None
        self.header = None
        self.affine = None
        self._load_file()
    
    def _load_file(self):
        """Load the NIfTI file and extract basic information."""
        try:
            self.nii_img = nib.load(self.nifti_path)
            self.image_data = self.nii_img.get_fdata()
            self.header = self.nii_img.header
            self.affine = self.nii_img.affine
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file {self.nifti_path}: {str(e)}")
    
    def validate_file_integrity(self):
        """
        Validate the basic integrity of the NIfTI file.
        
        Returns:
            dict: Validation results
        """
        validation_results = {
            'file_exists': Path(self.nifti_path).exists(),
            'file_readable': True,
            'valid_nifti': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if file is readable
            if self.nii_img is None:
                validation_results['file_readable'] = False
                validation_results['errors'].append("File is not readable")
                return validation_results
            
            # Check data integrity
            if self.image_data is None or self.image_data.size == 0:
                validation_results['valid_nifti'] = False
                validation_results['errors'].append("No image data found")
            
            # Check for NaN or infinite values
            if np.isnan(self.image_data).any():
                validation_results['warnings'].append("Image contains NaN values")
            
            if np.isinf(self.image_data).any():
                validation_results['warnings'].append("Image contains infinite values")
            
            # Check dimensions
            if len(self.image_data.shape) < 3:
                validation_results['warnings'].append("Image has less than 3 dimensions")
            
        except Exception as e:
            validation_results['valid_nifti'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def get_basic_info(self):
        """
        Get basic information about the NIfTI file.
        
        Returns:
            dict: Basic file information
        """
        file_size = Path(self.nifti_path).stat().st_size / (1024 * 1024)  # MB
        
        return {
            'file_path': str(self.nifti_path),
            'file_size_mb': round(file_size, 2),
            'dimensions': list(self.image_data.shape),
            'data_type': str(self.image_data.dtype),
            'total_voxels': int(self.image_data.size),
            'non_zero_voxels': int(np.count_nonzero(self.image_data)),
            'voxel_size': self.get_voxel_size(),
            'orientation': self.get_orientation()
        }
    
    def get_voxel_size(self):
        """
        Extract voxel size from the header.
        
        Returns:
            list: Voxel dimensions in mm
        """
        try:
            pixdim = self.header['pixdim'][1:4]  # First element is often 0 or -1
            return [float(dim) for dim in pixdim if dim > 0]
        except:
            return None
    
    def get_orientation(self):
        """
        Get image orientation information.
        
        Returns:
            str: Orientation code
        """
        try:
            return nib.aff2axcodes(self.affine)
        except:
            return None
    
    def get_intensity_statistics(self):
        """
        Calculate intensity statistics for the image.
        
        Returns:
            dict: Intensity statistics
        """
        non_zero_data = self.image_data[self.image_data != 0]
        
        stats = {
            'min_value': float(np.min(self.image_data)),
            'max_value': float(np.max(self.image_data)),
            'mean_value': float(np.mean(self.image_data)),
            'std_value': float(np.std(self.image_data)),
            'median_value': float(np.median(self.image_data))
        }
        
        if len(non_zero_data) > 0:
            stats.update({
                'min_nonzero': float(np.min(non_zero_data)),
                'max_nonzero': float(np.max(non_zero_data)),
                'mean_nonzero': float(np.mean(non_zero_data)),
                'std_nonzero': float(np.std(non_zero_data)),
                'median_nonzero': float(np.median(non_zero_data))
            })
        
        return stats
    
    def get_label_information(self):
        """
        Analyze labels/segments in the image.
        
        Returns:
            dict: Label information including unique labels and their counts
        """
        # Get unique values and their counts
        unique_values, counts = np.unique(self.image_data, return_counts=True)
        
        # Create label dictionary
        labels = {}
        for value, count in zip(unique_values, counts):
            labels[str(int(value))] = {
                'count': int(count),
                'percentage': float(count / self.image_data.size * 100)
            }
        
        # Identify potential background (usually 0 or most common value)
        background_label = str(int(unique_values[np.argmax(counts)]))
        
        # Count non-background labels
        non_background_labels = {k: v for k, v in labels.items() if k != background_label}
        
        return {
            'total_labels': len(unique_values),
            'background_label': background_label,
            'foreground_labels': len(non_background_labels),
            'label_details': labels,
            'non_background_labels': non_background_labels,
            'unique_values': [int(val) for val in unique_values]
        }
    
    def get_segment_analysis(self):
        """
        Perform detailed segment analysis.
        
        Returns:
            dict: Segment analysis results
        """
        from scipy import ndimage
        
        analysis = {}
        label_info = self.get_label_information()
        
        for label_str, info in label_info['non_background_labels'].items():
            label_val = int(label_str)
            binary_mask = (self.image_data == label_val)
            
            if np.any(binary_mask):
                # Connected component analysis
                labeled_array, num_components = ndimage.label(binary_mask)
                
                # Calculate properties for each component
                component_sizes = []
                if num_components > 0:
                    for i in range(1, num_components + 1):
                        component_size = np.sum(labeled_array == i)
                        component_sizes.append(component_size)
                
                # Centroid calculation
                centroid = ndimage.center_of_mass(binary_mask)
                
                analysis[label_str] = {
                    'voxel_count': info['count'],
                    'percentage': info['percentage'],
                    'connected_components': int(num_components),
                    'component_sizes': component_sizes,
                    'largest_component': max(component_sizes) if component_sizes else 0,
                    'centroid': [float(coord) for coord in centroid] if not np.isnan(centroid).any() else None,
                    'bounding_box': self._get_bounding_box(binary_mask)
                }
        
        return analysis
    
    def _get_bounding_box(self, binary_mask):
        """
        Calculate bounding box for a binary mask.
        
        Args:
            binary_mask (np.ndarray): Binary mask
            
        Returns:
            dict: Bounding box coordinates
        """
        coords = np.where(binary_mask)
        if len(coords[0]) == 0:
            return None
        
        return {
            'min_coords': [int(np.min(coord)) for coord in coords],
            'max_coords': [int(np.max(coord)) for coord in coords],
            'size': [int(np.max(coord) - np.min(coord) + 1) for coord in coords]
        }
    
    def generate_full_report(self):
        """
        Generate a comprehensive validation and analysis report.
        
        Returns:
            dict: Complete analysis report
        """
        report = {
            'validation': self.validate_file_integrity(),
            'basic_info': self.get_basic_info(),
            'intensity_stats': self.get_intensity_statistics(),
            'label_info': self.get_label_information(),
            'segment_analysis': self.get_segment_analysis()
        }
        
        return report


def validate_nifti(nifti_path, output_json=None, verbose=True):
    """
    Validate and analyze a NIfTI file.
    
    Args:
        nifti_path (str): Path to the NIfTI file
        output_json (str): Optional path to save results as JSON
        verbose (bool): Print detailed information
        
    Returns:
        dict: Analysis results
    """
    validator = NIfTIValidator(nifti_path)
    report = validator.generate_full_report()
    
    if verbose:
        print(f"\n=== NIfTI Validation Report: {nifti_path} ===")
        print(f"File Size: {report['basic_info']['file_size_mb']} MB")
        print(f"Dimensions: {report['basic_info']['dimensions']}")
        print(f"Data Type: {report['basic_info']['data_type']}")
        print(f"Total Voxels: {report['basic_info']['total_voxels']:,}")
        print(f"Non-zero Voxels: {report['basic_info']['non_zero_voxels']:,}")
        
        if report['basic_info']['voxel_size']:
            print(f"Voxel Size: {report['basic_info']['voxel_size']} mm")
        
        print(f"\nLabels Found: {report['label_info']['total_labels']}")
        print(f"Foreground Labels: {report['label_info']['foreground_labels']}")
        
        for label, details in report['segment_analysis'].items():
            print(f"\nLabel {label}:")
            print(f"  Voxels: {details['voxel_count']:,} ({details['percentage']:.2f}%)")
            print(f"  Connected Components: {details['connected_components']}")
            if details['centroid']:
                print(f"  Centroid: ({details['centroid'][0]:.1f}, {details['centroid'][1]:.1f}, {details['centroid'][2]:.1f})")
        
        if report['validation']['errors']:
            print(f"\nErrors: {report['validation']['errors']}")
        if report['validation']['warnings']:
            print(f"Warnings: {report['validation']['warnings']}")
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_json}")
    
    return report


def main():
    """Command line interface for NIfTI validation."""
    parser = argparse.ArgumentParser(description='Validate and analyze NIfTI files')
    parser.add_argument('input', help='Input NIfTI file path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        report = validate_nifti(
            nifti_path=args.input,
            output_json=args.output,
            verbose=not args.quiet
        )
        
        if not report['validation']['valid_nifti']:
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 