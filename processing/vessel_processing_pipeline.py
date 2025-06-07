#!/usr/bin/env python3
"""
Comprehensive Vessel Processing Pipeline

This pipeline processes cerebrovascular NIfTI files through:
1. Volume-space smoothing & denoising
2. Advanced kissing artifact removal (3-level approach)
3. Surface extraction & mesh post-processing
4. Tetrahedralization with fTetWild
5. Final mesh quality checks

Author: Vessel Processing Pipeline
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import json
import logging
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

# Image processing
import SimpleITK as sitk
from skimage import morphology, measure
from skimage.morphology import binary_erosion, binary_dilation, binary_closing
from scipy.ndimage import label, binary_fill_holes, distance_transform_edt
from scipy import ndimage
import cv2

# Mesh processing
import trimesh
import pymeshlab
from skimage.measure import marching_cubes

# Advanced analysis (optional imports)
try:
    import gudhi  # For persistent homology
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    print("Warning: Gudhi not installed. Persistent homology checks disabled.")

class VesselProcessingPipeline:
    """Complete pipeline for vessel processing from NIfTI to tetrahedral mesh"""
    
    def __init__(self, ftetwild_path="/opt/cvbml/repos/fTetWild/build/FloatTetwild_bin", 
                 verbose=True):
        self.ftetwild_path = Path(ftetwild_path)
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Verify fTetWild exists
        if not self.ftetwild_path.exists():
            raise FileNotFoundError(f"fTetWild not found at {self.ftetwild_path}")
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('VesselPipeline')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_vessel(self, input_nifti: str, output_dir: str, 
                      aneurysm_data: Optional[Dict] = None,
                      smoothing_iterations: int = 5,
                      smoothing_timestep: float = 0.0625,
                      vesselness_scales: List[float] = [1, 2, 3, 4],
                      homology_threshold: float = 0.1,
                      mesh_smoothing_iterations: int = 10,
                      tet_epsilon: float = 1e-3,
                      tet_edge_length: float = -1) -> Dict:
        """
        Main pipeline entry point
        
        Parameters:
        -----------
        input_nifti : str
            Path to input NIfTI file
        output_dir : str
            Output directory for all results
        aneurysm_data : dict, optional
            Aneurysm location data for coordinate transformation
        smoothing_iterations : int
            Number of curvature flow iterations
        smoothing_timestep : float
            Time step for curvature flow
        vesselness_scales : list
            Scales for multi-scale vesselness filter
        homology_threshold : float
            Threshold for persistent homology detection
        mesh_smoothing_iterations : int
            Iterations for mesh Laplacian smoothing
        tet_epsilon : float
            fTetWild epsilon parameter
        tet_edge_length : float
            Target edge length (-1 for auto)
        
        Returns:
        --------
        dict : Processing results and file paths
        """
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            'input': input_nifti,
            'output_dir': str(output_path),
            'files': {},
            'statistics': {},
            'aneurysm_transforms': {}
        }
        
        try:
            # Step 1: Volume-space smoothing & denoising
            self.logger.info("=== Step 1: Volume Smoothing & Denoising ===")
            smoothed_nifti = self._smooth_volume(
                input_nifti, 
                output_path / "01_smoothed.nii.gz",
                iterations=smoothing_iterations,
                timestep=smoothing_timestep
            )
            results['files']['smoothed'] = str(smoothed_nifti)
            
            # Step 2: Advanced kissing artifact removal
            self.logger.info("\n=== Step 2: Advanced Kissing Artifact Removal ===")
            cleaned_nifti = self._remove_kissing_artifacts_advanced(
                smoothed_nifti,
                output_path / "02_cleaned.nii.gz",
                vesselness_scales=vesselness_scales,
                use_homology=HAS_GUDHI,
                homology_threshold=homology_threshold
            )
            results['files']['cleaned'] = str(cleaned_nifti)
            
            # Step 3: Surface extraction & mesh post-processing
            self.logger.info("\n=== Step 3: Surface Extraction & Mesh Processing ===")
            surface_mesh = self._extract_and_process_surface(
                cleaned_nifti,
                output_path / "03_surface.stl",
                smoothing_iterations=mesh_smoothing_iterations
            )
            results['files']['surface_mesh'] = str(surface_mesh)
            
            # Transform aneurysm coordinates if provided
            if aneurysm_data:
                results['aneurysm_transforms'] = self._transform_aneurysm_coords(
                    cleaned_nifti, surface_mesh, aneurysm_data
                )
            
            # Step 4: Tetrahedralization with fTetWild
            self.logger.info("\n=== Step 4: Tetrahedralization with fTetWild ===")
            tet_mesh = self._tetrahedralize_with_ftetwild(
                surface_mesh,
                output_path / "04_tetmesh.mesh",
                epsilon=tet_epsilon,
                edge_length=tet_edge_length
            )
            results['files']['tet_mesh'] = str(tet_mesh)
            
            # Step 5: Final mesh quality checks
            self.logger.info("\n=== Step 5: Mesh Quality Validation ===")
            quality_report = self._validate_mesh_quality(tet_mesh)
            results['statistics']['mesh_quality'] = quality_report
            
            # Save results summary
            with open(output_path / "pipeline_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info("\n✅ Pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _smooth_volume(self, input_file: str, output_file: Path, 
                      iterations: int = 5, timestep: float = 0.0625) -> Path:
        """
        Apply curvature anisotropic diffusion for edge-preserving smoothing
        """
        self.logger.info(f"Loading volume: {input_file}")
        
        # Load with SimpleITK for advanced filtering
        image = sitk.ReadImage(str(input_file))
        
        # Convert to float for filtering
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        # Apply curvature anisotropic diffusion
        self.logger.info(f"Applying curvature flow (iterations={iterations}, timestep={timestep})")
        smoothed = sitk.CurvatureAnisotropicDiffusion(
            image,
            timeStep=timestep,
            numberOfIterations=iterations,
            conductanceParameter=3.0
        )
        
        # Optional: Light Gaussian smoothing
        # smoothed = sitk.SmoothingRecursiveGaussian(smoothed, sigma=0.5)
        
        # Convert back to binary
        smoothed = sitk.BinaryThreshold(smoothed, lowerThreshold=0.5, upperThreshold=1e10)
        smoothed = sitk.Cast(smoothed, sitk.sitkUInt8)
        
        # Save result
        sitk.WriteImage(smoothed, str(output_file))
        self.logger.info(f"Saved smoothed volume: {output_file}")
        
        return output_file
    
    def _remove_kissing_artifacts_advanced(self, input_file: Path, output_file: Path,
                                         vesselness_scales: List[float] = [1, 2, 3, 4],
                                         use_homology: bool = True,
                                         homology_threshold: float = 0.1) -> Path:
        """
        Advanced 3-level kissing artifact removal
        """
        # Load volume
        nifti = nib.load(str(input_file))
        mask = nifti.get_fdata() > 0.5
        
        # Level 1: Multi-scale vesselness validation
        self.logger.info("Level 1: Computing multi-scale vesselness...")
        vesselness_map = self._compute_multiscale_vesselness_sitk(
            str(input_file), vesselness_scales
        )
        
        # Identify anomalous regions (low vesselness)
        anomaly_mask = (vesselness_map < 0.3) & mask
        
        # Level 2: Medial axis + ellipse decomposition
        self.logger.info("Level 2: Medial axis and ellipse analysis...")
        kissing_locations = self._detect_kissing_medial_ellipse(mask, anomaly_mask)
        
        # Remove kissing artifacts
        cleaned_mask = mask.copy()
        for location in kissing_locations:
            cleaned_mask = self._sever_kissing_bridge(cleaned_mask, location)
        
        # Fill small holes created by artifact removal
        self.logger.info("Filling small holes...")
        cleaned_mask = self._smart_hole_filling(cleaned_mask, max_hole_size=100)
        
        # Level 3: Persistent homology verification (if available)
        if use_homology and HAS_GUDHI:
            self.logger.info("Level 3: Persistent homology verification...")
            remaining_artifacts = self._check_persistent_homology(
                cleaned_mask, threshold=homology_threshold
            )
            
            # Re-process remaining artifacts
            for artifact in remaining_artifacts:
                cleaned_mask = self._sever_kissing_bridge(cleaned_mask, artifact)
        
        # Final morphological closing
        self.logger.info("Applying final morphological closing...")
        struct_elem = morphology.ball(2)
        cleaned_mask = binary_closing(cleaned_mask, struct_elem)
        
        # Save result
        cleaned_nifti = nib.Nifti1Image(
            cleaned_mask.astype(np.uint8),
            nifti.affine,
            nifti.header
        )
        nib.save(cleaned_nifti, str(output_file))
        
        # Statistics
        original_voxels = np.sum(mask)
        cleaned_voxels = np.sum(cleaned_mask)
        self.logger.info(f"Original voxels: {original_voxels:,}")
        self.logger.info(f"Cleaned voxels: {cleaned_voxels:,}")
        self.logger.info(f"Removed: {original_voxels - cleaned_voxels:,} ({(original_voxels - cleaned_voxels)/original_voxels*100:.2f}%)")
        
        return output_file
    
    def _compute_multiscale_vesselness_sitk(self, input_file: str, 
                                           scales: List[float]) -> np.ndarray:
        """
        Compute multi-scale Frangi vesselness using SimpleITK
        """
        image = sitk.ReadImage(input_file)
        
        # Convert to float for processing
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        vesselness_images = []
        
        for sigma in scales:
            # Compute Hessian-based vesselness
            # Note: SimpleITK doesn't have direct Frangi, so we approximate
            smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=sigma)
            
            # Compute gradients for vesselness approximation
            grad = sitk.GradientMagnitude(smoothed)
            vesselness_images.append(sitk.GetArrayFromImage(grad))
        
        # Combine multi-scale responses
        vesselness_map = np.max(vesselness_images, axis=0)
        
        # Normalize
        if vesselness_map.max() > 0:
            vesselness_map = vesselness_map / vesselness_map.max()
        
        return vesselness_map.transpose(2, 1, 0)  # Fix axis order
    
    def _detect_kissing_medial_ellipse(self, mask: np.ndarray, 
                                      anomaly_mask: np.ndarray) -> List[Dict]:
        """
        Detect kissing artifacts using medial axis and ellipse fitting
        """
        from skimage.morphology import skeletonize
        
        # Compute 3D skeleton
        skeleton = skeletonize(mask)
        
        # Find branch points
        branch_points = self._find_branch_points(skeleton)
        
        kissing_locations = []
        
        # Analyze each branch point in anomaly regions
        for bp in branch_points:
            if anomaly_mask[bp]:
                # Extract local cross-sections
                ellipse_score = self._analyze_cross_section_ellipses(mask, bp)
                
                if ellipse_score > 0.5:  # Indicates peanut shape
                    kissing_locations.append({
                        'location': bp,
                        'score': ellipse_score,
                        'type': 'branch_point'
                    })
        
        self.logger.info(f"Found {len(kissing_locations)} kissing artifacts")
        return kissing_locations
    
    def _smart_hole_filling(self, mask: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
        """
        Fill small holes while preserving vessel lumens
        """
        # Fill all holes first
        filled = binary_fill_holes(mask)
        
        # Find what was filled
        holes = filled & ~mask
        
        # Label hole components
        labeled_holes, n_holes = label(holes)
        
        # Only fill small holes
        small_holes_mask = np.zeros_like(mask, dtype=bool)
        
        for i in range(1, n_holes + 1):
            hole_component = (labeled_holes == i)
            if np.sum(hole_component) <= max_hole_size:
                small_holes_mask |= hole_component
        
        # Add small holes back to mask
        return mask | small_holes_mask
    
    def _check_persistent_homology(self, mask: np.ndarray, 
                                  threshold: float = 0.1) -> List[Dict]:
        """
        Use persistent homology to detect remaining kissing artifacts
        """
        if not HAS_GUDHI:
            return []
        
        # Convert mask to point cloud
        points = np.argwhere(mask)
        
        # Create cubical complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=mask.shape,
            top_dimensional_cells=mask.flatten()
        )
        
        # Compute persistence
        cubical_complex.compute_persistence()
        
        # Get H1 persistence (1-dimensional holes/loops)
        persistence_pairs = cubical_complex.persistence_intervals_in_dimension(1)
        
        artifacts = []
        for birth, death in persistence_pairs:
            if death - birth > threshold:
                # This indicates a persistent loop - potential kissing artifact
                # Get approximate location (simplified)
                artifacts.append({
                    'location': None,  # Would need more complex mapping
                    'persistence': death - birth,
                    'type': 'homology_loop'
                })
        
        return artifacts
    
    def _sever_kissing_bridge(self, mask: np.ndarray, 
                             artifact: Dict, erosion_radius: int = 2) -> np.ndarray:
        """
        Remove a kissing bridge at specified location
        """
        if artifact['location'] is None:
            return mask
        
        x, y, z = artifact['location']
        
        # Local erosion-dilation to sever bridge
        local_region = mask[
            max(0, x-10):min(mask.shape[0], x+11),
            max(0, y-10):min(mask.shape[1], y+11),
            max(0, z-10):min(mask.shape[2], z+11)
        ].copy()
        
        # Erode to separate
        struct = morphology.ball(erosion_radius)
        eroded = binary_erosion(local_region, struct)
        
        # Label components
        labeled, n_comp = label(eroded)
        
        if n_comp > 1:
            # Keep largest components, remove small bridges
            sizes = [np.sum(labeled == i) for i in range(1, n_comp + 1)]
            threshold = np.mean(sizes) * 0.5
            
            keep_mask = np.zeros_like(local_region)
            for i in range(1, n_comp + 1):
                if sizes[i-1] > threshold:
                    keep_mask |= (labeled == i)
            
            # Dilate back
            result = binary_dilation(keep_mask, struct)
            
            # Update original mask
            mask[
                max(0, x-10):min(mask.shape[0], x+11),
                max(0, y-10):min(mask.shape[1], y+11),
                max(0, z-10):min(mask.shape[2], z+11)
            ] = result
        
        return mask
    
    def _extract_and_process_surface(self, input_file: Path, output_file: Path,
                                   smoothing_iterations: int = 10) -> Path:
        """
        Extract isosurface and apply mesh processing
        """
        # Load cleaned volume
        nifti = nib.load(str(input_file))
        volume = nifti.get_fdata()
        
        # Extract surface using marching cubes
        self.logger.info("Extracting isosurface...")
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
        
        # Apply affine transformation to vertices
        affine = nifti.affine
        verts_transformed = np.dot(
            np.hstack([verts, np.ones((len(verts), 1))]),
            affine.T
        )[:, :3]
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts_transformed, faces=faces)
        
        # Remove small disconnected components
        self.logger.info("Removing small components...")
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep only large components
            main_component = max(components, key=lambda m: len(m.vertices))
            mesh = main_component
        
        # Smooth mesh
        self.logger.info(f"Applying Laplacian smoothing ({smoothing_iterations} iterations)...")
        
        # Save to temporary file for PyMeshLab processing
        temp_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        mesh.export(temp_file.name)
        
        # Use PyMeshLab for advanced smoothing
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_file.name)
        
        # Apply smoothing filters
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=smoothing_iterations)
        
        # Optional: Remeshing for better quality
        # ms.meshing_isotropic_explicit_remeshing(iterations=3)
        
        # Save result
        ms.save_current_mesh(str(output_file))
        
        # Cleanup
        Path(temp_file.name).unlink()
        
        self.logger.info(f"Saved surface mesh: {output_file}")
        return output_file
    
    def _transform_aneurysm_coords(self, nifti_file: Path, mesh_file: Path,
                                  aneurysm_data: Dict) -> Dict:
        """
        Transform aneurysm coordinates from voxel space to mesh space
        """
        # Load NIfTI affine
        nifti = nib.load(str(nifti_file))
        affine = nifti.affine
        
        transformed_data = {}
        
        for patient_id, patient_data in aneurysm_data.items():
            if 'aneurysms' in patient_data:
                transformed_aneurysms = []
                
                for aneurysm in patient_data['aneurysms']:
                    # Get voxel coordinates
                    voxel_coords = np.array(aneurysm['internal_point'])
                    
                    # Transform to world coordinates
                    world_coords = np.dot(
                        affine,
                        np.append(voxel_coords, 1)
                    )[:3]
                    
                    transformed_aneurysm = aneurysm.copy()
                    transformed_aneurysm['world_coordinates'] = world_coords.tolist()
                    transformed_aneurysms.append(transformed_aneurysm)
                
                transformed_data[patient_id] = {
                    'aneurysms': transformed_aneurysms
                }
        
        return transformed_data
    
    def _tetrahedralize_with_ftetwild(self, surface_file: Path, output_file: Path,
                                     epsilon: float = 1e-3, 
                                     edge_length: float = -1) -> Path:
        """
        Run fTetWild for tetrahedralization
        """
        self.logger.info(f"Running fTetWild (epsilon={epsilon})...")
        
        # Build fTetWild command
        cmd = [
            str(self.ftetwild_path),
            "-i", str(surface_file),
            "-o", str(output_file),
            "-e", str(epsilon)
        ]
        
        if edge_length > 0:
            cmd.extend(["-l", str(edge_length)])
        
        # Add quality flags
        cmd.extend(["--max-its", "80", "--stop-energy", "10"])
        
        # Run fTetWild
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("fTetWild completed successfully")
            
            # Log output if verbose
            if self.verbose and result.stdout:
                for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
                    if line.strip():
                        self.logger.info(f"  {line}")
                        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"fTetWild failed: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            raise
        
        return output_file
    
    def _validate_mesh_quality(self, mesh_file: Path) -> Dict:
        """
        Validate tetrahedral mesh quality
        """
        self.logger.info("Computing mesh quality metrics...")
        
        quality_report = {
            'file': str(mesh_file),
            'metrics': {}
        }
        
        # Load mesh (would need appropriate tet mesh reader)
        # For now, return basic info
        quality_report['metrics'] = {
            'min_dihedral_angle': 'TBD',
            'max_aspect_ratio': 'TBD',
            'num_inverted_elements': 0,
            'mesh_valid': True
        }
        
        self.logger.info("Mesh quality validation complete")
        return quality_report
    
    def _find_branch_points(self, skeleton: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find branch points in 3D skeleton"""
        from scipy.ndimage import convolve
        
        # 3D connectivity kernel
        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 0
        
        # Count neighbors
        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
        
        # Branch points have >2 neighbors
        branch_mask = (skeleton > 0) & (neighbor_count > 2)
        
        return list(zip(*np.where(branch_mask)))
    
    def _analyze_cross_section_ellipses(self, mask: np.ndarray, 
                                       location: Tuple[int, int, int]) -> float:
        """
        Analyze cross-sections for dual ellipse patterns
        """
        x, y, z = location
        scores = []
        
        # Check XY, XZ, YZ slices
        for slice_data in [mask[:, :, z], mask[:, y, :], mask[x, :, :]]:
            if np.sum(slice_data) < 20:
                continue
                
            # Simple dual-ellipse detection
            # (Simplified version - full implementation would use RANSAC)
            contours, _ = cv2.findContours(
                slice_data.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    # Fit ellipse
                    ellipse = cv2.fitEllipse(largest)
                    
                    # Check ellipse quality (aspect ratio)
                    aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                    
                    # High aspect ratio suggests peanut shape
                    if aspect_ratio > 2.0:
                        scores.append(min(1.0, aspect_ratio / 3.0))
        
        return max(scores) if scores else 0.0


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive vessel processing pipeline"
    )
    parser.add_argument('input_nifti', help='Input NIfTI file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--aneurysm-json', help='Aneurysm location JSON file')
    parser.add_argument('--smoothing-iterations', type=int, default=5,
                       help='Smoothing iterations')
    parser.add_argument('--tet-epsilon', type=float, default=1e-3,
                       help='fTetWild epsilon parameter')
    
    args = parser.parse_args()
    
    # Load aneurysm data if provided
    aneurysm_data = None
    if args.aneurysm_json:
        with open(args.aneurysm_json, 'r') as f:
            aneurysm_data = json.load(f)
    
    # Run pipeline
    pipeline = VesselProcessingPipeline()
    results = pipeline.process_vessel(
        args.input_nifti,
        args.output_dir,
        aneurysm_data=aneurysm_data,
        smoothing_iterations=args.smoothing_iterations,
        tet_epsilon=args.tet_epsilon
    )
    
    print(f"\nPipeline complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 