"""
NIfTI Aneurysm Extraction using Random Walk
Extracts aneurysm regions using random walk algorithm to reach the interior of vessel walls.
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.segmentation import random_walker
from skimage.filters import gaussian
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings


class AneurysmExtractor:
    """Class for extracting aneurysms using random walk algorithm."""
    
    def __init__(self, nifti_path, aneurysm_coords=None):
        """
        Initialize aneurysm extractor.
        
        Args:
            nifti_path (str): Path to the NIfTI file
            aneurysm_coords (dict): Aneurysm coordinate information
        """
        self.nifti_path = nifti_path
        self.aneurysm_coords = aneurysm_coords or {}
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
    
    def extract_aneurysms(self, walk_iterations=10, beta=130, tol=1e-3):
        """
        Extract all aneurysms using random walk algorithm.
        
        Args:
            walk_iterations (int): Number of random walk iterations per aneurysm
            beta (float): Random walk beta parameter (controls edge weighting)
            tol (float): Convergence tolerance
            
        Returns:
            dict: Extraction results with individual aneurysm files
        """
        # First, try to find aneurysms if coordinates not provided
        if not self.aneurysm_coords:
            from .NIfTI_find_aneurysm import find_aneurysms
            self.aneurysm_coords = find_aneurysms(self.nifti_path, verbose=False)
        
        aneurysms = self.aneurysm_coords.get('aneurysms', {})
        extracted_files = {}
        
        print(f"Starting random walk extraction for {len(aneurysms)} aneurysms...")
        
        for aneurysm_id, aneurysm_info in aneurysms.items():
            print(f"\nExtracting {aneurysm_id}...")
            
            extracted_results = []
            for iteration in range(walk_iterations):
                print(f"  Random walk iteration {iteration + 1}/{walk_iterations}")
                
                extracted_volume = self._extract_single_aneurysm(
                    aneurysm_info, iteration, beta, tol
                )
                
                if extracted_volume is not None:
                    # Save the extracted volume
                    output_path = self._save_extracted_volume(
                        extracted_volume, aneurysm_id, iteration
                    )
                    extracted_results.append(output_path)
            
            extracted_files[aneurysm_id] = {
                'aneurysm_info': aneurysm_info,
                'extracted_files': extracted_results,
                'walk_iterations': walk_iterations
            }
        
        return {
            'total_aneurysms': len(aneurysms),
            'extracted_aneurysms': extracted_files,
            'parameters': {
                'walk_iterations': walk_iterations,
                'beta': beta,
                'tolerance': tol
            }
        }
    
    def _extract_single_aneurysm(self, aneurysm_info, iteration, beta, tol):
        """
        Extract a single aneurysm using random walk.
        
        Args:
            aneurysm_info (dict): Aneurysm information
            iteration (int): Current iteration number
            beta (float): Random walk beta parameter
            tol (float): Convergence tolerance
            
        Returns:
            np.ndarray: Extracted volume or None if extraction failed
        """
        try:
            # Get aneurysm region
            label_value = aneurysm_info['label_value']
            centroid = aneurysm_info['centroid_voxel']
            bbox = aneurysm_info['bounding_box']
            
            # Create region of interest around the aneurysm
            roi, roi_offset = self._create_roi(bbox, padding=20)
            
            # Prepare image data for random walk
            roi_data = roi.astype(np.float64)
            
            # Apply Gaussian smoothing to reduce noise
            roi_smoothed = gaussian(roi_data, sigma=1.0)
            
            # Create markers for random walk
            markers = self._create_random_walk_markers(
                roi_smoothed, centroid, roi_offset, iteration
            )
            
            if markers is None:
                return None
            
            # Perform random walk segmentation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segmentation = random_walker(
                    roi_smoothed, 
                    markers, 
                    beta=beta, 
                    tol=tol,
                    return_full_prob=False
                )
            
            # Extract the aneurysm region (label 2 in our markers)
            extracted_mask = (segmentation == 2)
            
            # Post-process the extraction
            extracted_volume = self._post_process_extraction(
                extracted_mask, roi_data, roi_offset
            )
            
            return extracted_volume
            
        except Exception as e:
            print(f"    Error in extraction: {str(e)}")
            return None
    
    def _create_roi(self, bbox, padding=20):
        """
        Create region of interest around aneurysm.
        
        Args:
            bbox (dict): Bounding box information
            padding (int): Padding around the bounding box
            
        Returns:
            tuple: (roi_data, roi_offset)
        """
        min_coords = bbox['min_coords']
        max_coords = bbox['max_coords']
        
        # Add padding and ensure we stay within image bounds
        padded_min = [max(0, coord - padding) for coord in min_coords]
        padded_max = [min(self.image_data.shape[i], coord + padding) 
                      for i, coord in enumerate(max_coords)]
        
        # Extract ROI
        roi = self.image_data[
            padded_min[0]:padded_max[0],
            padded_min[1]:padded_max[1],
            padded_min[2]:padded_max[2]
        ]
        
        return roi, padded_min
    
    def _create_random_walk_markers(self, roi_data, centroid, roi_offset, iteration):
        """
        Create markers for random walk segmentation.
        
        Args:
            roi_data (np.ndarray): ROI data
            centroid (list): Aneurysm centroid coordinates
            roi_offset (list): ROI offset coordinates
            iteration (int): Current iteration number
            
        Returns:
            np.ndarray: Marker array or None if failed
        """
        markers = np.zeros_like(roi_data, dtype=np.int32)
        
        # Adjust centroid to ROI coordinates
        roi_centroid = [
            int(centroid[i] - roi_offset[i]) for i in range(3)
        ]
        
        # Check if centroid is within ROI
        if not all(0 <= roi_centroid[i] < roi_data.shape[i] for i in range(3)):
            return None
        
        # Create background markers (label 1) - edges of ROI
        edge_thickness = 3
        markers[:edge_thickness, :, :] = 1
        markers[-edge_thickness:, :, :] = 1
        markers[:, :edge_thickness, :] = 1
        markers[:, -edge_thickness:, :] = 1
        markers[:, :, :edge_thickness] = 1
        markers[:, :, -edge_thickness:] = 1
        
        # Create aneurysm seed markers (label 2) around centroid with some randomness
        seed_radius = 3 + iteration  # Vary seed size with iteration
        np.random.seed(42 + iteration)  # Reproducible randomness
        
        # Add multiple seed points with slight randomness
        num_seeds = 5 + iteration
        for _ in range(num_seeds):
            # Add random offset to centroid
            offset = np.random.randint(-seed_radius, seed_radius + 1, 3)
            seed_coord = [roi_centroid[i] + offset[i] for i in range(3)]
            
            # Ensure seed is within bounds
            seed_coord = [
                max(edge_thickness, min(roi_data.shape[i] - edge_thickness - 1, coord))
                for i, coord in enumerate(seed_coord)
            ]
            
            # Create spherical seed region
            y, x, z = np.ogrid[:roi_data.shape[0], :roi_data.shape[1], :roi_data.shape[2]]
            distance = np.sqrt(
                (x - seed_coord[1])**2 + 
                (y - seed_coord[0])**2 + 
                (z - seed_coord[2])**2
            )
            
            seed_mask = distance <= seed_radius
            markers[seed_mask] = 2
        
        return markers
    
    def _post_process_extraction(self, extracted_mask, roi_data, roi_offset):
        """
        Post-process the extracted aneurysm.
        
        Args:
            extracted_mask (np.ndarray): Extracted binary mask
            roi_data (np.ndarray): Original ROI data
            roi_offset (list): ROI offset coordinates
            
        Returns:
            np.ndarray: Full-size extracted volume
        """
        # Clean up the extraction
        # Remove small isolated components
        labeled_array, num_components = ndimage.label(extracted_mask)
        
        if num_components > 0:
            # Keep only the largest connected component
            component_sizes = [
                np.sum(labeled_array == i) for i in range(1, num_components + 1)
            ]
            largest_component = np.argmax(component_sizes) + 1
            extracted_mask = (labeled_array == largest_component)
        
        # Apply morphological operations to smooth the result
        extracted_mask = ndimage.binary_fill_holes(extracted_mask)
        extracted_mask = ndimage.binary_opening(extracted_mask, iterations=1)
        extracted_mask = ndimage.binary_closing(extracted_mask, iterations=2)
        
        # Create full-size volume
        full_volume = np.zeros_like(self.image_data)
        
        # Place extracted region back into full volume
        end_coords = [
            roi_offset[i] + extracted_mask.shape[i] for i in range(3)
        ]
        
        full_volume[
            roi_offset[0]:end_coords[0],
            roi_offset[1]:end_coords[1],
            roi_offset[2]:end_coords[2]
        ] = extracted_mask.astype(float)
        
        return full_volume
    
    def _save_extracted_volume(self, volume, aneurysm_id, iteration):
        """
        Save extracted volume as NIfTI file.
        
        Args:
            volume (np.ndarray): Extracted volume
            aneurysm_id (str): Aneurysm identifier
            iteration (int): Iteration number
            
        Returns:
            str: Output file path
        """
        # Create output filename
        input_name = Path(self.nifti_path).stem
        if input_name.endswith('.nii'):
            input_name = input_name[:-4]
        
        output_path = f"{input_name}_{aneurysm_id}_walk_{iteration:02d}.nii.gz"
        
        # Create NIfTI image
        extracted_img = nib.Nifti1Image(volume, self.affine, self.header)
        
        # Save file
        nib.save(extracted_img, output_path)
        
        print(f"    Saved: {output_path}")
        return output_path


def extract_aneurysms(nifti_path, aneurysm_coords_file=None, walk_iterations=10, 
                     beta=130, output_summary=None, verbose=True):
    """
    Extract aneurysms from NIfTI file using random walk.
    
    Args:
        nifti_path (str): Path to the NIfTI file
        aneurysm_coords_file (str): Path to aneurysm coordinates JSON file
        walk_iterations (int): Number of random walk iterations per aneurysm
        beta (float): Random walk beta parameter
        output_summary (str): Path to save extraction summary
        verbose (bool): Print detailed information
        
    Returns:
        dict: Extraction results
    """
    # Load aneurysm coordinates if provided
    aneurysm_coords = {}
    if aneurysm_coords_file and Path(aneurysm_coords_file).exists():
        with open(aneurysm_coords_file, 'r') as f:
            aneurysm_coords = json.load(f)
    
    extractor = AneurysmExtractor(nifti_path, aneurysm_coords)
    results = extractor.extract_aneurysms(walk_iterations, beta)
    
    if verbose:
        print(f"\n=== Aneurysm Extraction Report: {nifti_path} ===")
        print(f"Total aneurysms processed: {results['total_aneurysms']}")
        print(f"Random walk iterations per aneurysm: {walk_iterations}")
        print(f"Beta parameter: {beta}")
        
        for aneurysm_id, info in results['extracted_aneurysms'].items():
            print(f"\n{aneurysm_id.upper()}:")
            print(f"  Label: {info['aneurysm_info']['label_value']}")
            print(f"  Extracted files: {len(info['extracted_files'])}")
            for i, filepath in enumerate(info['extracted_files']):
                print(f"    {i+1}. {filepath}")
    
    if output_summary:
        with open(output_summary, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nExtraction summary saved to: {output_summary}")
    
    return results


def main():
    """Command line interface for aneurysm extraction."""
    parser = argparse.ArgumentParser(description='Extract aneurysms using random walk')
    parser.add_argument('input', help='Input NIfTI file path')
    parser.add_argument('-c', '--coords', help='Aneurysm coordinates JSON file path')
    parser.add_argument('-n', '--iterations', type=int, default=10,
                       help='Number of random walk iterations per aneurysm (default: 10)')
    parser.add_argument('-b', '--beta', type=float, default=130,
                       help='Random walk beta parameter (default: 130)')
    parser.add_argument('-o', '--output', help='Output summary JSON file path')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        results = extract_aneurysms(
            nifti_path=args.input,
            aneurysm_coords_file=args.coords,
            walk_iterations=args.iterations,
            beta=args.beta,
            output_summary=args.output,
            verbose=not args.quiet
        )
        
        total_files = sum(
            len(info['extracted_files']) 
            for info in results['extracted_aneurysms'].values()
        )
        
        print(f"\nExtraction completed. Generated {total_files} files.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 