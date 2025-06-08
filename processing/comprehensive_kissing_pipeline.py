#!/usr/bin/env python3
"""
Comprehensive Kissing Artifact Removal Pipeline

Combines multiple advanced methods:
1. VMTK for centerline and vessel analysis
2. GUDHI for topological loop detection
3. Integration points for pretrained models (DeepVesselNet, etc.)

This implements the 3-level approach suggested in the research:
- Level 1: Hessian-based vessel enhancement
- Level 2: Medial axis + ellipse fitting
- Level 3: Persistent homology for loop detection
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from scipy.ndimage import gaussian_filter, binary_erosion, label
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize
import os
import torch

# Optional imports with fallbacks
try:
    import vmtk
    from vmtk import vmtkscripts
    VMTK_AVAILABLE = True
except ImportError:
    VMTK_AVAILABLE = False
    print("Warning: VMTK not available. Some features will be disabled.")

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: GUDHI not available. Topological analysis will be disabled.")


class ComprehensiveKissingPipeline:
    """
    Comprehensive pipeline combining multiple methods for kissing artifact removal
    """
    
    def __init__(self, verbose=True, use_gpu=False):
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.logger = self._setup_logger()
        
        # Check available methods
        self.methods_available = {
            'vmtk': VMTK_AVAILABLE,
            'gudhi': GUDHI_AVAILABLE,
            'gpu': use_gpu and self._check_gpu()
        }
        
        self.logger.info(f"Available methods: {self.methods_available}")
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('ComprehensiveKissing')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_gpu(self):
        """Check if GPU acceleration is available"""
        try:
            import cupy
            return True
        except ImportError:
            return False
    
    def process(self, input_nifti: str, output_nifti: str,
               method: str = 'auto',
               vessel_enhancement: bool = True,
               use_centerlines: bool = True,
               use_topology: bool = True,
               pretrained_model: Optional[str] = None) -> Dict:
        """
        Main processing pipeline
        
        Parameters:
        -----------
        input_nifti : str
            Input segmented vessel NIfTI
        output_nifti : str
            Output cleaned NIfTI
        method : str
            Processing method: 'auto', 'vmtk', 'topology', 'combined'
        vessel_enhancement : bool
            Apply Hessian-based vessel enhancement (Level 1)
        use_centerlines : bool
            Use centerline analysis (Level 2)
        use_topology : bool
            Use topological analysis (Level 3)
        pretrained_model : str, optional
            Path to pretrained model for initial segmentation refinement
        
        Returns:
        --------
        dict : Processing statistics and results
        """
        self.logger.info(f"Processing {input_nifti} with comprehensive pipeline")
        
        # Load data
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        mask = volume > 0.5
        
        # Initialize results
        results = {
            'input_file': input_nifti,
            'output_file': output_nifti,
            'stages': {},
            'artifacts_found': 0,
            'voxels_removed': 0
        }
        
        # Stage 0: Pretrained model refinement (if provided)
        if pretrained_model:
            self.logger.info("Stage 0: Applying pretrained model refinement...")
            mask = self._apply_pretrained_model(mask, pretrained_model)
            results['stages']['pretrained_refinement'] = True
        
        # Stage 1: Vessel enhancement (Hessian-based)
        if vessel_enhancement:
            self.logger.info("Stage 1: Hessian-based vessel enhancement...")
            enhanced_mask, vessel_stats = self._enhance_vessels(mask, affine)
            mask = enhanced_mask
            results['stages']['vessel_enhancement'] = vessel_stats
        
        # Stage 2: Centerline analysis (if VMTK available)
        centerline_artifacts = []
        if use_centerlines and self.methods_available['vmtk']:
            self.logger.info("Stage 2: Centerline-based analysis...")
            centerline_artifacts = self._analyze_centerlines(mask, affine)
            results['stages']['centerline_analysis'] = {
                'artifacts_found': len(centerline_artifacts)
            }
        
        # Stage 3: Topological analysis (if GUDHI available)
        topological_artifacts = []
        if use_topology and self.methods_available['gudhi']:
            self.logger.info("Stage 3: Topological loop detection...")
            topological_artifacts = self._analyze_topology(mask, affine)
            results['stages']['topological_analysis'] = {
                'loops_found': len(topological_artifacts)
            }
        
        # Combine all detected artifacts
        all_artifacts = self._combine_artifacts(
            centerline_artifacts, topological_artifacts, affine
        )
        results['artifacts_found'] = len(all_artifacts)
        
        # Remove artifacts
        if all_artifacts:
            self.logger.info(f"Removing {len(all_artifacts)} artifacts...")
            cleaned_mask, removal_stats = self._remove_artifacts(
                mask, all_artifacts, affine
            )
            results['voxels_removed'] = removal_stats['voxels_removed']
        else:
            self.logger.info("No artifacts found")
            cleaned_mask = mask
        
        # Save result
        cleaned_volume = cleaned_mask.astype(np.float32)
        nib.save(nib.Nifti1Image(cleaned_volume, affine), output_nifti)
        
        # Summary
        results['summary'] = {
            'input_voxels': int(mask.sum()),
            'output_voxels': int(cleaned_mask.sum()),
            'voxels_removed': results['voxels_removed'],
            'percentage_removed': 100 * results['voxels_removed'] / max(1, int(mask.sum()))
        }
        
        self.logger.info(f"Complete! Removed {results['voxels_removed']} voxels "
                        f"({results['summary']['percentage_removed']:.2f}%)")
        
        return results
    
    def _enhance_vessels(self, mask: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Level 1: Hessian-based vessel enhancement
        """
        # Convert mask to grayscale with distance transform
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
        
        # Multi-scale Hessian analysis
        scales = [1.0, 2.0, 3.0]  # mm
        vesselness = np.zeros_like(dist)
        
        for scale in scales:
            # Apply Gaussian smoothing
            smoothed = gaussian_filter(dist, sigma=scale)
            
            # Compute Hessian eigenvalues
            H = hessian_matrix(smoothed, sigma=scale)
            eigvals = hessian_matrix_eigvals(H)
            
            # Frangi vesselness measure
            # For 3D: look for tube-like structures (2 large negative eigenvalues)
            lambda1 = eigvals[0]  # Smallest eigenvalue
            lambda2 = eigvals[1]
            lambda3 = eigvals[2]  # Largest eigenvalue
            
            # Vesselness response
            Rb = np.abs(lambda1) / np.sqrt(np.abs(lambda2 * lambda3) + 1e-10)
            Ra = np.abs(lambda2) / (np.abs(lambda3) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
            
            # Parameters
            alpha = 0.5  # Plate-like structure suppression
            beta = 0.5   # Blob-like structure suppression
            c = 0.5 * np.max(S)  # Noise suppression
            
            # Frangi filter response
            exp_Ra = 1 - np.exp(-(Ra**2) / (2 * alpha**2))
            exp_Rb = np.exp(-(Rb**2) / (2 * beta**2))
            exp_S = 1 - np.exp(-(S**2) / (2 * c**2))
            
            # Combine responses
            response = exp_Ra * exp_Rb * exp_S
            
            # Only enhance where eigenvalues indicate vessel
            vessel_mask = (lambda2 < 0) & (lambda3 < 0)
            response[~vessel_mask] = 0
            
            vesselness = np.maximum(vesselness, response)
        
        # Threshold to get enhanced mask
        threshold = 0.1 * np.max(vesselness)
        enhanced_mask = mask & (vesselness > threshold)
        
        stats = {
            'scales_used': scales,
            'voxels_enhanced': int(enhanced_mask.sum()),
            'voxels_original': int(mask.sum())
        }
        
        return enhanced_mask, stats
    
    def _analyze_centerlines(self, mask: np.ndarray, affine: np.ndarray) -> List[Dict]:
        """
        Level 2: Centerline-based analysis using VMTK
        """
        if not VMTK_AVAILABLE:
            return []
        
        # Import VMTK processor
        from vmtk_kissing_removal import VMTKKissingRemoval
        
        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp_in:
            with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp_out:
                # Save mask to temporary file
                nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), tmp_in.name)
                
                # Process with VMTK
                processor = VMTKKissingRemoval(verbose=False)
                stats = processor.process_nifti(tmp_in.name, tmp_out.name)
                
                # Extract artifact locations
                artifacts = []
                if stats.get('topology', {}).get('suspicious_loops'):
                    for loop in stats['topology']['suspicious_loops']:
                        artifacts.append({
                            'type': 'centerline_loop',
                            'location': loop.get('center', [0, 0, 0]),
                            'score': 0.8,
                            'source': 'vmtk'
                        })
                
                return artifacts
    
    def _analyze_topology(self, mask: np.ndarray, affine: np.ndarray) -> List[Dict]:
        """
        Level 3: Topological analysis using persistent homology
        """
        if not GUDHI_AVAILABLE:
            return []
        
        # Import topology processor
        from topological_kissing_detection import TopologicalKissingDetection
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
            # Save mask to temporary file
            nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), tmp.name)
            
            # Detect loops
            detector = TopologicalKissingDetection(verbose=False)
            detection = detector.detect_kissing_artifacts(tmp.name)
            
            # Extract significant loops
            artifacts = []
            for loop in detection['loops'][:5]:  # Top 5 loops
                if loop['persistence'] > 5.0:  # Significant persistence
                    artifacts.append({
                        'type': 'topological_loop',
                        'persistence': loop['persistence'],
                        'score': min(1.0, loop['persistence'] / 20.0),
                        'source': 'gudhi'
                    })
            
            return artifacts
    
    def _combine_artifacts(self, centerline_artifacts: List[Dict],
                          topological_artifacts: List[Dict],
                          affine: np.ndarray) -> List[Dict]:
        """
        Combine artifacts from different detection methods
        """
        all_artifacts = []
        
        # Add all artifacts with proper indexing
        for i, artifact in enumerate(centerline_artifacts):
            artifact['id'] = f'centerline_{i}'
            all_artifacts.append(artifact)
        
        for i, artifact in enumerate(topological_artifacts):
            artifact['id'] = f'topology_{i}'
            all_artifacts.append(artifact)
        
        # Sort by score
        all_artifacts.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Merge nearby artifacts
        merged = self._merge_nearby_artifacts(all_artifacts, merge_radius=5.0)
        
        return merged
    
    def _merge_nearby_artifacts(self, artifacts: List[Dict], 
                               merge_radius: float) -> List[Dict]:
        """
        Merge artifacts that are close together
        """
        if not artifacts:
            return []
        
        merged = []
        used = set()
        
        for i, artifact in enumerate(artifacts):
            if i in used:
                continue
            
            # Start new cluster
            cluster = {
                'type': 'merged',
                'artifacts': [artifact],
                'score': artifact.get('score', 0.5),
                'sources': [artifact.get('source', 'unknown')]
            }
            
            # Find nearby artifacts
            if 'location' in artifact:
                loc1 = np.array(artifact['location'])
                
                for j, other in enumerate(artifacts[i+1:], i+1):
                    if j not in used and 'location' in other:
                        loc2 = np.array(other['location'])
                        if np.linalg.norm(loc1 - loc2) < merge_radius:
                            used.add(j)
                            cluster['artifacts'].append(other)
                            cluster['score'] = max(cluster['score'], 
                                                 other.get('score', 0.5))
                            cluster['sources'].append(other.get('source', 'unknown'))
            
            merged.append(cluster)
        
        return merged
    
    def _remove_artifacts(self, mask: np.ndarray, artifacts: List[Dict],
                         affine: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove detected artifacts from mask
        """
        cleaned_mask = mask.copy()
        total_removed = 0
        
        # Skeleton for bridge detection
        skeleton = skeletonize(mask)
        
        for artifact in artifacts:
            if artifact.get('score', 0) < 0.5:  # Skip low confidence
                continue
            
            # Find bridge voxels near artifact
            bridge_voxels = self._find_bridge_voxels_near_artifact(
                skeleton, mask, artifact, affine
            )
            
            # Remove bridge voxels
            for voxel in bridge_voxels:
                if cleaned_mask[voxel]:
                    cleaned_mask[voxel] = False
                    total_removed += 1
        
        # Clean up small disconnected components
        labeled, num_labels = label(cleaned_mask)
        if num_labels > 1:
            sizes = [(labeled == i).sum() for i in range(1, num_labels + 1)]
            largest = np.argmax(sizes) + 1
            cleaned_mask = (labeled == largest)
        
        stats = {
            'voxels_removed': total_removed,
            'components_before': num_labels,
            'components_after': 1
        }
        
        return cleaned_mask, stats
    
    def _find_bridge_voxels_near_artifact(self, skeleton: np.ndarray,
                                         mask: np.ndarray,
                                         artifact: Dict,
                                         affine: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Find bridge voxels near a detected artifact
        """
        # Find thin regions
        eroded = binary_erosion(mask, iterations=2)
        thin_regions = mask & ~eroded
        
        # Find skeleton points in thin regions
        bridge_candidates = skeleton & thin_regions
        
        # Get all candidate coordinates
        coords = np.argwhere(bridge_candidates)
        
        # If artifact has location, filter by distance
        if 'location' in artifact and len(coords) > 0:
            # Convert location to voxel coordinates
            world_coord = np.array(artifact['location'] + [1])
            inv_affine = np.linalg.inv(affine)
            voxel_coord = np.dot(inv_affine, world_coord)[:3]
            
            # Find nearby voxels
            distances = np.linalg.norm(coords - voxel_coord, axis=1)
            radius_voxels = 10  # ~10mm radius
            nearby_mask = distances < radius_voxels
            
            return [tuple(c) for c in coords[nearby_mask]]
        
        # Otherwise return a sample of bridge candidates
        if len(coords) > 100:
            indices = np.random.choice(len(coords), 100, replace=False)
            return [tuple(coords[i]) for i in indices]
        
        return [tuple(c) for c in coords]
    
    def _apply_pretrained_model(self, mask: np.ndarray, 
                               model_path: str) -> np.ndarray:
        """
        Apply pretrained model for vessel segmentation refinement
        
        This integrates actual models like:
        - DeepVesselNet
        - VesselFM
        - Retina U-Net
        """
        try:
            # Import the pretrained model module
            from pretrained_vessel_segmentation import PretrainedVesselSegmentation
            
            # Determine model type from path or use default
            if 'vesselfm' in model_path.lower():
                model_name = 'vesselfm'
            elif 'deepvessel' in model_path.lower():
                model_name = 'deepvesselnet'
            elif 'retina' in model_path.lower():
                model_name = 'retina_unet'
            else:
                model_name = 'unet'  # Default
            
            self.logger.info(f"Using pretrained {model_name} model")
            
            # Initialize segmenter
            segmenter = PretrainedVesselSegmentation(
                model_name=model_name,
                device='auto'
            )
            
            # Create temporary files for processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_in:
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_out:
                    # Save current mask to temporary file
                    nib.save(nib.Nifti1Image(mask.astype(np.float32), np.eye(4)), tmp_in.name)
                    
                    # If model_path points to weights, load them
                    if os.path.exists(model_path) and model_path.endswith('.pth'):
                        segmenter.model = segmenter.load_model()
                        if segmenter.model is not None:
                            checkpoint = torch.load(model_path, map_location=segmenter.device)
                            segmenter.model.load_state_dict(checkpoint)
                            self.logger.info(f"Loaded weights from {model_path}")
                    
                    # Refine segmentation
                    stats = segmenter.refine_segmentation(
                        tmp_in.name,
                        tmp_in.name,  # Use same as initial
                        tmp_out.name,
                        iterations=2
                    )
                    
                    # Load refined result
                    refined_nifti = nib.load(tmp_out.name)
                    refined_mask = refined_nifti.get_fdata() > 0.5
                    
                    # Clean up
                    os.unlink(tmp_in.name)
                    os.unlink(tmp_out.name)
                    
                    self.logger.info(f"Pretrained model refinement: {stats['percent_change']:+.1f}% change")
                    
                    return refined_mask
                    
        except ImportError:
            self.logger.warning("Pretrained model module not available, skipping")
            return mask
        except Exception as e:
            self.logger.error(f"Error applying pretrained model: {e}")
            return mask


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive kissing artifact removal pipeline"
    )
    parser.add_argument('input_nifti', help='Input segmented vessel NIfTI')
    parser.add_argument('output_nifti', help='Output cleaned NIfTI')
    parser.add_argument('--method', default='auto',
                       choices=['auto', 'vmtk', 'topology', 'combined'],
                       help='Processing method')
    parser.add_argument('--no-enhancement', action='store_true',
                       help='Skip vessel enhancement')
    parser.add_argument('--no-centerlines', action='store_true',
                       help='Skip centerline analysis')
    parser.add_argument('--no-topology', action='store_true',
                       help='Skip topological analysis')
    parser.add_argument('--pretrained-model',
                       help='Path to pretrained segmentation model')
    parser.add_argument('--save-stats', help='Save processing statistics to JSON')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ComprehensiveKissingPipeline()
    
    # Process
    results = pipeline.process(
        args.input_nifti,
        args.output_nifti,
        method=args.method,
        vessel_enhancement=not args.no_enhancement,
        use_centerlines=not args.no_centerlines,
        use_topology=not args.no_topology,
        pretrained_model=args.pretrained_model
    )
    
    # Save statistics if requested
    if args.save_stats:
        with open(args.save_stats, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Artifacts found: {results['artifacts_found']}")
    print(f"Voxels removed: {results['voxels_removed']} "
          f"({results['summary']['percentage_removed']:.2f}%)")
    print(f"Output saved to: {args.output_nifti}")


if __name__ == "__main__":
    main() 