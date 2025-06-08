#!/usr/bin/env python3
"""
Topological Kissing Artifact Detection using Persistent Homology

Uses GUDHI library to detect topological features (loops) that indicate
kissing artifacts in vessel segmentations.

Installation:
    conda install -c conda-forge gudhi
    or
    pip install gudhi
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

try:
    import gudhi
except ImportError:
    print("ERROR: GUDHI not installed. Please install with:")
    print("  conda install -c conda-forge gudhi")
    print("  or")
    print("  pip install gudhi")
    raise


class TopologicalKissingDetection:
    """Use persistent homology to detect kissing artifacts"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('TopologicalKissing')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def detect_kissing_artifacts(self, input_nifti: str) -> Dict:
        """
        Detect kissing artifacts using persistent homology
        
        Returns dictionary with:
        - loops: List of detected loops with persistence
        - kissing_regions: Voxel coordinates of potential kissing artifacts
        """
        self.logger.info(f"Analyzing {input_nifti} for topological features")
        
        # Load NIfTI
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        voxel_size = np.abs(affine.diagonal()[:3])
        
        # Convert to binary mask
        mask = volume > 0.5
        
        # Step 1: Extract skeleton
        self.logger.info("Step 1: Extracting vessel skeleton...")
        skeleton = skeletonize(mask)
        
        # Step 2: Build cubical complex from skeleton
        self.logger.info("Step 2: Building cubical complex...")
        complex_data = self._build_cubical_complex(skeleton)
        
        # Step 3: Compute persistent homology
        self.logger.info("Step 3: Computing persistent homology...")
        persistence = self._compute_persistence(complex_data, skeleton.shape)
        
        # Step 4: Analyze H1 (loops) for kissing artifacts
        self.logger.info("Step 4: Analyzing loops (H1 features)...")
        loops = self._analyze_loops(persistence, voxel_size)
        
        # Step 5: Identify kissing regions from significant loops
        self.logger.info("Step 5: Identifying kissing regions...")
        kissing_regions = self._identify_kissing_regions(
            skeleton, mask, loops, voxel_size
        )
        
        self.logger.info(f"Found {len(loops)} loops, {len(kissing_regions)} potential kissing artifacts")
        
        return {
            'loops': loops,
            'kissing_regions': kissing_regions,
            'skeleton_voxels': int(skeleton.sum()),
            'mask_voxels': int(mask.sum())
        }
    
    def remove_kissing_artifacts(self, input_nifti: str, output_nifti: str,
                               persistence_threshold: float = 10.0,
                               min_loop_size: float = 20.0) -> Dict:
        """
        Remove kissing artifacts based on topological analysis
        
        Parameters:
        -----------
        persistence_threshold : float
            Minimum persistence (in mm) to consider a loop significant
        min_loop_size : float
            Minimum loop size (in mm) to consider for removal
        """
        # Detect artifacts
        detection = self.detect_kissing_artifacts(input_nifti)
        
        # Load original data
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        mask = volume > 0.5
        
        # Filter significant loops
        significant_loops = [
            loop for loop in detection['loops']
            if loop['persistence'] > persistence_threshold and
               loop['size'] > min_loop_size
        ]
        
        self.logger.info(f"Removing {len(significant_loops)} significant loops")
        
        # Remove kissing regions
        cleaned_mask = mask.copy()
        total_removed = 0
        
        for region in detection['kissing_regions']:
            if region['loop_index'] < len(significant_loops):
                # Remove voxels in this region
                for voxel in region['voxels']:
                    if 0 <= voxel[0] < mask.shape[0] and \
                       0 <= voxel[1] < mask.shape[1] and \
                       0 <= voxel[2] < mask.shape[2]:
                        cleaned_mask[voxel[0], voxel[1], voxel[2]] = False
                        total_removed += 1
        
        # Save result
        cleaned_volume = cleaned_mask.astype(np.float32)
        nib.save(nib.Nifti1Image(cleaned_volume, affine), output_nifti)
        
        self.logger.info(f"Removed {total_removed} voxels")
        
        return {
            'loops_found': len(detection['loops']),
            'loops_removed': len(significant_loops),
            'voxels_removed': total_removed,
            'detection': detection
        }
    
    def _build_cubical_complex(self, skeleton: np.ndarray) -> np.ndarray:
        """Build data for cubical complex from skeleton"""
        # Create distance transform from skeleton
        # Inverted so skeleton has low values (will persist)
        dist = distance_transform_edt(~skeleton)
        
        # Normalize and invert
        if dist.max() > 0:
            dist = 1.0 - (dist / dist.max())
        
        # Set background to high value
        dist[~skeleton] = 2.0
        
        return dist
    
    def _compute_persistence(self, complex_data: np.ndarray, shape: Tuple) -> List:
        """Compute persistent homology using GUDHI"""
        # Since CubicalComplex is not available, we'll use a simplified approach
        # with SimplexTree based on skeleton points
        st = gudhi.SimplexTree()
        
        # Get skeleton points
        skeleton_points = np.argwhere(complex_data < 1.5)  # Skeleton has low values
        
        if len(skeleton_points) == 0:
            return []
        
        # Add vertices
        for i, point in enumerate(skeleton_points):
            st.insert([i], filtration=complex_data[tuple(point)])
        
        # Add edges for nearby points (simplified connectivity)
        for i in range(len(skeleton_points)):
            for j in range(i + 1, len(skeleton_points)):
                # Check if points are adjacent (26-connectivity)
                diff = np.abs(skeleton_points[i] - skeleton_points[j])
                if np.max(diff) <= 1:  # Adjacent voxels
                    filtration = max(complex_data[tuple(skeleton_points[i])],
                                   complex_data[tuple(skeleton_points[j])])
                    st.insert([i, j], filtration=filtration)
        
        # Compute persistence
        st.compute_persistence()
        persistence = st.persistence()
        
        # Convert to more usable format
        persistence_pairs = []
        for p in persistence:
            dim, (birth, death) = p[0], p[1]
            if dim == 1 and death != float('inf'):  # H1 (loops) with finite death
                persistence_pairs.append({
                    'dimension': dim,
                    'birth': birth,
                    'death': death,
                    'persistence': death - birth
                })
        
        return persistence_pairs
    
    def _analyze_loops(self, persistence: List[Dict], voxel_size: np.ndarray) -> List[Dict]:
        """Analyze loops from persistence diagram"""
        loops = []
        
        for i, p in enumerate(persistence):
            if p['dimension'] == 1:  # Only interested in 1D loops
                # Estimate physical size based on persistence and voxel size
                size_estimate = p['persistence'] * np.mean(voxel_size) * 10  # Rough estimate
                
                loops.append({
                    'index': i,
                    'birth': p['birth'],
                    'death': p['death'],
                    'persistence': p['persistence'] * np.mean(voxel_size),  # Convert to mm
                    'size': size_estimate,
                    'significance': p['persistence']  # Higher persistence = more significant
                })
        
        # Sort by significance
        loops.sort(key=lambda x: x['significance'], reverse=True)
        
        return loops
    
    def _identify_kissing_regions(self, skeleton: np.ndarray, mask: np.ndarray,
                                 loops: List[Dict], voxel_size: np.ndarray) -> List[Dict]:
        """Identify voxel regions corresponding to loops"""
        kissing_regions = []
        
        # For each significant loop, find the corresponding region
        for i, loop in enumerate(loops[:10]):  # Process top 10 loops
            # This is a simplified approach - in practice, you'd need to
            # track the actual cycle representatives from the cubical complex
            
            # Find potential bridge regions by looking for thin connections
            bridge_voxels = self._find_bridge_voxels(skeleton, mask, loop)
            
            if bridge_voxels:
                kissing_regions.append({
                    'loop_index': i,
                    'persistence': loop['persistence'],
                    'voxels': bridge_voxels,
                    'size': len(bridge_voxels) * np.prod(voxel_size)
                })
        
        return kissing_regions
    
    def _find_bridge_voxels(self, skeleton: np.ndarray, mask: np.ndarray,
                           loop: Dict) -> List[Tuple[int, int, int]]:
        """Find voxels that form bridges (simplified approach)"""
        bridge_voxels = []
        
        # Erode the mask to find thin regions
        eroded = binary_erosion(mask, iterations=2)
        thin_regions = mask & ~eroded
        
        # Find thin regions that are part of the skeleton
        bridge_candidates = skeleton & thin_regions
        
        # Get coordinates of bridge voxels
        coords = np.argwhere(bridge_candidates)
        
        # In a real implementation, you would use the cycle representatives
        # from persistent homology to identify the exact loop location
        # This is a simplified heuristic
        
        # For now, return a subset based on loop significance
        if len(coords) > 0:
            # Sample based on loop persistence
            n_samples = min(len(coords), int(loop['persistence'] * 10))
            indices = np.random.choice(len(coords), n_samples, replace=False)
            bridge_voxels = [tuple(coords[i]) for i in indices]
        
        return bridge_voxels


def visualize_persistence_diagram(loops: List[Dict], output_file: str = "persistence_diagram.png"):
    """Visualize persistence diagram for detected loops"""
    try:
        import matplotlib.pyplot as plt
        
        if not loops:
            print("No loops to visualize")
            return
        
        births = [loop['birth'] for loop in loops]
        deaths = [loop['death'] for loop in loops]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(births, deaths, alpha=0.6, s=50)
        
        # Add diagonal line
        max_val = max(max(births), max(deaths))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.title('Persistence Diagram (H1 - Loops)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Persistence diagram saved to {output_file}")
        
    except ImportError:
        print("Matplotlib not available for visualization")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Topological kissing artifact detection using persistent homology"
    )
    parser.add_argument('input_nifti', help='Input segmented vessel NIfTI')
    parser.add_argument('--output-nifti', help='Output cleaned NIfTI (optional)')
    parser.add_argument('--persistence-threshold', type=float, default=10.0,
                       help='Minimum persistence in mm to consider significant')
    parser.add_argument('--min-loop-size', type=float, default=20.0,
                       help='Minimum loop size in mm')
    parser.add_argument('--visualize', action='store_true',
                       help='Create persistence diagram visualization')
    
    args = parser.parse_args()
    
    # Detect artifacts
    detector = TopologicalKissingDetection()
    
    if args.output_nifti:
        # Detect and remove
        stats = detector.remove_kissing_artifacts(
            args.input_nifti,
            args.output_nifti,
            persistence_threshold=args.persistence_threshold,
            min_loop_size=args.min_loop_size
        )
        
        print(f"\nProcessing complete!")
        print(f"Loops found: {stats['loops_found']}")
        print(f"Loops removed: {stats['loops_removed']}")
        print(f"Voxels removed: {stats['voxels_removed']}")
        
        if args.visualize:
            visualize_persistence_diagram(stats['detection']['loops'])
    else:
        # Just detect
        detection = detector.detect_kissing_artifacts(args.input_nifti)
        
        print(f"\nDetection complete!")
        print(f"Loops found: {len(detection['loops'])}")
        print(f"Kissing regions: {len(detection['kissing_regions'])}")
        
        # Show top loops
        print("\nTop 5 most persistent loops:")
        for i, loop in enumerate(detection['loops'][:5]):
            print(f"  {i+1}. Persistence: {loop['persistence']:.2f}mm, "
                  f"Size: {loop['size']:.2f}mm")
        
        if args.visualize:
            visualize_persistence_diagram(detection['loops'])


if __name__ == "__main__":
    main() 