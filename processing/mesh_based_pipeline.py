#!/usr/bin/env python3
"""
Mesh-Based Vessel Processing Pipeline

This pipeline processes vessels primarily at the mesh level:
1. NIfTI → STL conversion with minimal processing
2. Mesh smoothing
3. Mesh-based kissing artifact detection and removal
4. Tetrahedralization with fTetWild

Working on meshes provides more precise control over geometry modifications.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# Mesh processing
import trimesh
import pymeshlab
from skimage.measure import marching_cubes
from scipy.spatial import KDTree
import networkx as nx
from sklearn.decomposition import PCA


class MeshBasedVesselPipeline:
    """Mesh-based pipeline for vessel processing"""
    
    def __init__(self, ftetwild_path="/opt/cvbml/repos/fTetWild/build/FloatTetwild_bin", 
                 verbose=True, n_jobs=16):
        self.ftetwild_path = Path(ftetwild_path)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.logger = self._setup_logger()
        
        # Verify fTetWild exists
        if not self.ftetwild_path.exists():
            raise FileNotFoundError(f"fTetWild not found at {self.ftetwild_path}")
        
        self.logger.info(f"Using {self.n_jobs} CPUs for parallel processing")
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('MeshPipeline')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_vessel(self, input_nifti: str, output_dir: str,
                      aneurysm_data: Optional[Dict] = None,
                      initial_smooth_iterations: int = 20,
                      remesh_iterations: int = 3,
                      kissing_detection_radius: float = 3.0,
                      min_bridge_ratio: float = 0.6,
                      tet_epsilon: float = 1e-3,
                      detection_threshold_factor: float = 1.0) -> Dict:
        """
        Main pipeline entry point
        
        Parameters:
        -----------
        input_nifti : str
            Path to input NIfTI file
        output_dir : str
            Output directory for results
        aneurysm_data : dict, optional
            Aneurysm location data
        initial_smooth_iterations : int
            Initial mesh smoothing iterations
        remesh_iterations : int
            Isotropic remeshing iterations
        kissing_detection_radius : float
            Radius for kissing artifact detection (mm)
        min_bridge_ratio : float
            Minimum ratio to consider as bridge (0-1)
        tet_epsilon : float
            fTetWild epsilon parameter
        detection_threshold_factor : float
            Factor to multiply detection thresholds (>1 = more aggressive)
            
        Returns:
        --------
        dict : Processing results
        """
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            'input': input_nifti,
            'output_dir': str(output_path),
            'files': {},
            'statistics': {}
        }
        
        try:
            # Step 1: NIfTI to STL conversion (minimal processing)
            self.logger.info("=== Step 1: NIfTI to STL Conversion ===")
            initial_mesh = self._nifti_to_stl(
                input_nifti,
                output_path / "01_initial.stl"
            )
            results['files']['initial_mesh'] = str(initial_mesh)
            
            # Step 2: Initial mesh smoothing and remeshing
            self.logger.info("\n=== Step 2: Mesh Smoothing & Remeshing ===")
            smoothed_mesh = self._smooth_and_remesh(
                initial_mesh,
                output_path / "02_smoothed.stl",
                smooth_iterations=initial_smooth_iterations,
                remesh_iterations=remesh_iterations
            )
            results['files']['smoothed_mesh'] = str(smoothed_mesh)
            
            # Step 3: Skip kissing artifact removal (must be done at voxel level)
            self.logger.info("\n=== Step 3: Kissing Artifact Handling ===")
            self.logger.info("NOTE: Kissing artifacts must be handled at the voxel level")
            self.logger.info("      See kissing_artifact_analysis_summary.md for details")
            self.logger.info("      The mesh is already a single connected component")
            
            # Step 4: Final smoothing and cleanup
            self.logger.info("\n=== Step 4: Final Smoothing & Cleanup ===")
            final_mesh = self._final_smooth(
                smoothed_mesh,
                output_path / "03_final.stl",
                iterations=5
            )
            results['files']['final_mesh'] = str(final_mesh)
            
            # Step 5: Tetrahedralization with fTetWild
            self.logger.info("\n=== Step 5: Tetrahedralization with fTetWild ===")
            tet_mesh = self._tetrahedralize_with_ftetwild(
                final_mesh,
                output_path / "04_tetmesh.mesh",
                epsilon=tet_epsilon
            )
            results['files']['tet_mesh'] = str(tet_mesh)
            
            # Save results summary
            with open(output_path / "pipeline_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info("\n✅ Mesh-based pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _nifti_to_stl(self, input_file: str, output_file: Path) -> Path:
        """
        Convert NIfTI to STL with minimal processing
        """
        self.logger.info(f"Loading NIfTI: {input_file}")
        
        # Load NIfTI
        nifti = nib.load(input_file)
        volume = nifti.get_fdata()
        
        # Simple binary threshold
        mask = volume > 0.5
        
        # Fill small holes to ensure watertight mesh
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
        
        # Extract surface using marching cubes
        self.logger.info("Extracting surface mesh...")
        verts, faces, normals, values = marching_cubes(mask, level=0.5)
        
        # Apply affine transformation
        affine = nifti.affine
        verts_transformed = np.dot(
            np.hstack([verts, np.ones((len(verts), 1))]),
            affine.T
        )[:, :3]
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=verts_transformed, faces=faces)
        
        # Remove small disconnected components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            self.logger.info(f"Found {len(components)} components, keeping largest")
            mesh = max(components, key=lambda m: len(m.vertices))
        
        # Fix mesh issues
        mesh.fix_normals()
        mesh.fill_holes()
        
        # Save
        mesh.export(str(output_file))
        self.logger.info(f"Saved initial mesh: {output_file}")
        self.logger.info(f"  Vertices: {len(mesh.vertices):,}")
        self.logger.info(f"  Faces: {len(mesh.faces):,}")
        
        return output_file
    
    def _smooth_and_remesh(self, input_mesh: Path, output_mesh: Path,
                          smooth_iterations: int = 20,
                          remesh_iterations: int = 3) -> Path:
        """
        Smooth and remesh for better quality
        """
        self.logger.info("Applying mesh smoothing and remeshing...")
        
        # Use PyMeshLab for advanced operations
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_mesh))
        
        # Get initial stats
        initial_vertices = ms.current_mesh().vertex_number()
        
        # Apply Taubin smoothing (less shrinkage than Laplacian)
        self.logger.info(f"  Taubin smoothing ({smooth_iterations} iterations)...")
        ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=smooth_iterations)
        
        # Isotropic remeshing for uniform triangles
        # Skip remeshing for now due to API compatibility issues
        # TODO: Fix remeshing with correct pymeshlab API
        if False and remesh_iterations > 0:
            self.logger.info(f"  Isotropic remeshing ({remesh_iterations} iterations)...")
            # Compute target edge length based on mesh size
            bbox = ms.current_mesh().bounding_box()
            diagonal = bbox.diagonal()
            target_edge_len = diagonal * 0.005  # 0.5% of diagonal
            
            # Try to use percentage of bounding box diagonal
            ms.meshing_isotropic_explicit_remeshing(
                iterations=remesh_iterations,
                targetlen=pymeshlab.Percentage(0.5)  # 0.5% of diagonal
            )
        
        # Save result
        ms.save_current_mesh(str(output_mesh))
        
        final_vertices = ms.current_mesh().vertex_number()
        self.logger.info(f"  Vertices: {initial_vertices:,} → {final_vertices:,}")
        
        return output_mesh
    
    def _detect_and_fix_kissing_artifacts(self, input_mesh: Path, output_mesh: Path,
                                         detection_radius: float = 3.0,
                                         min_bridge_ratio: float = 0.6,
                                         detection_threshold_factor: float = 1.0) -> Tuple[Path, Dict]:
        """
        Detect and fix kissing artifacts at the mesh level
        """
        self.logger.info("Detecting kissing artifacts on mesh...")
        
        # Load mesh
        mesh = trimesh.load(str(input_mesh))
        
        # Find potential kissing artifacts
        kissing_regions = self._find_kissing_regions(mesh, detection_radius, min_bridge_ratio)
        
        self.logger.info(f"Found {len(kissing_regions)} potential kissing artifacts")
        
        # Log details about detected regions
        for i, region in enumerate(kissing_regions):
            if 'avg_thickness' in region:
                self.logger.info(f"  Region {i+1}: {len(region['faces'])} faces, "
                               f"avg thickness: {region['avg_thickness']:.2f}mm, "
                               f"score: {region['score']:.2f}, "
                               f"center: ({region['center'][0]:.1f}, {region['center'][1]:.1f}, {region['center'][2]:.1f})")
        
        # Fix each kissing artifact
        fixed_mesh = mesh.copy()
        removed_faces = set()
        
        for i, region in enumerate(kissing_regions):
            self.logger.info(f"  Fixing artifact {i+1}/{len(kissing_regions)}")
            
            # Remove faces in the bridge region
            faces_to_remove = self._get_bridge_faces(fixed_mesh, region, 
                                                    expansion_factor=detection_threshold_factor)
            self.logger.info(f"    Removing {len(faces_to_remove)} faces (expanded from {len(region['faces'])})")
            removed_faces.update(faces_to_remove)
        
        # Remove marked faces
        if removed_faces:
            mask = np.ones(len(fixed_mesh.faces), dtype=bool)
            mask[list(removed_faces)] = False
            fixed_mesh = fixed_mesh.submesh([mask], append=True)
            
            # Fill holes created by removal
            self.logger.info("Filling holes...")
            fixed_mesh.fill_holes()
        
        # Clean up mesh
        fixed_mesh.remove_duplicate_faces()
        fixed_mesh.remove_degenerate_faces()
        fixed_mesh.fix_normals()
        
        # Save result
        fixed_mesh.export(str(output_mesh))
        
        # Statistics
        info = {
            'num_artifacts': len(kissing_regions),
            'faces_removed': len(removed_faces),
            'original_faces': len(mesh.faces),
            'final_faces': len(fixed_mesh.faces)
        }
        
        self.logger.info(f"  Removed {len(removed_faces)} faces")
        self.logger.info(f"  Final mesh: {len(fixed_mesh.vertices)} vertices, {len(fixed_mesh.faces)} faces")
        
        return output_mesh, info
    
    def _find_kissing_regions(self, mesh: trimesh.Trimesh, 
                             radius: float, min_ratio: float) -> List[Dict]:
        """
        Find regions where vessels are kissing based on mesh geometry
        """
        kissing_regions = []
        
        # Method 1: Detect narrow bridges using mesh thickness analysis
        self.logger.info("  Analyzing mesh thickness...")
        thin_regions = self._find_thin_regions(mesh, radius)
        
        # Skip other methods for speed - thickness analysis is most effective
        # Method 2: Detect high curvature saddle points
        # self.logger.info("  Analyzing surface curvature...")
        # saddle_regions = self._find_saddle_regions(mesh)
        
        # Method 3: Detect constrictions using geodesic analysis
        # self.logger.info("  Analyzing geodesic distances...")
        # constriction_regions = self._find_constrictions(mesh, min_ratio)
        
        # Use only thin regions for now
        all_regions = thin_regions
        
        # Merge nearby regions
        merged_regions = self._merge_nearby_regions(all_regions, merge_distance=radius*2)
        
        return merged_regions
    
    def _find_thin_regions(self, mesh: trimesh.Trimesh, max_thickness: float) -> List[Dict]:
        """
        Find regions where mesh is thin (potential bridges) - PARALLELIZED
        """
        regions = []
        
        # Compute approximate thickness using ray casting
        # Sample points on the mesh - use more samples for better coverage
        samples, face_idx = trimesh.sample.sample_surface_even(mesh, count=20000)  # Doubled samples
        
        # For each sample, cast ray inward to find thickness
        normals = mesh.face_normals[face_idx]
        
        # Create ray origins slightly inside the surface
        origins = samples - normals * 0.1
        
        # Parallel thickness computation
        def compute_thickness_batch(batch_idx):
            batch_thickness = {}
            batch_size = len(batch_idx)
            
            # Cast rays for this batch
            batch_origins = origins[batch_idx]
            batch_normals = -normals[batch_idx]
            
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=batch_origins,
                ray_directions=batch_normals
            )
            
            # Map back to original indices
            for i, loc in enumerate(locations):
                if i < len(index_ray):
                    orig_idx = batch_idx[index_ray[i]]
                    thickness = np.linalg.norm(samples[orig_idx] - loc)
                    if thickness < max_thickness:
                        batch_thickness[face_idx[orig_idx]] = thickness
            
            return batch_thickness
        
        # Split work into batches
        n_samples = len(samples)
        batch_size = max(100, n_samples // self.n_jobs)
        batches = []
        
        for i in range(0, n_samples, batch_size):
            batch_idx = list(range(i, min(i + batch_size, n_samples)))
            batches.append(batch_idx)
        
        # Process batches in parallel
        thickness_map = {}
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(compute_thickness_batch, batch) for batch in batches]
            
            for future in futures:
                batch_result = future.result()
                thickness_map.update(batch_result)
        
        # Cluster thin faces into regions
        if thickness_map:
            thin_faces = list(thickness_map.keys())
            clusters = self._cluster_faces(mesh, thin_faces)
            
            for cluster in clusters:
                avg_thickness = sum(thickness_map.get(f, max_thickness) for f in cluster) / len(cluster)
                # More sensitive scoring - consider anything below 80% of max_thickness as significant
                score = 1.0 - (avg_thickness / max_thickness) * 0.8
                
                regions.append({
                    'type': 'thin',
                    'faces': cluster,
                    'center': mesh.triangles_center[cluster].mean(axis=0),
                    'score': score,
                    'avg_thickness': avg_thickness
                })
        
        return regions
    
    def _find_saddle_regions(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Find saddle-shaped regions (characteristic of kissing artifacts)
        """
        regions = []
        
        # Compute vertex normals and curvature
        vertex_normals = mesh.vertex_normals
        
        # Approximate Gaussian curvature using neighboring normals
        curvatures = []
        for i, vertex in enumerate(mesh.vertices):
            # Get neighboring vertices
            neighbors = mesh.vertex_neighbors[i]
            if len(neighbors) < 3:
                curvatures.append(0)
                continue
            
            # Compute normal variation
            neighbor_normals = vertex_normals[neighbors]
            normal_var = np.var(neighbor_normals, axis=0).sum()
            curvatures.append(normal_var)
        
        curvatures = np.array(curvatures)
        
        # Find high curvature vertices (potential saddle points)
        threshold = np.percentile(curvatures, 90)
        high_curv_vertices = np.where(curvatures > threshold)[0]
        
        # Convert to face regions
        high_curv_faces = set()
        for v in high_curv_vertices:
            faces = mesh.vertex_faces[v]
            high_curv_faces.update(f for f in faces if f >= 0)
        
        if high_curv_faces:
            clusters = self._cluster_faces(mesh, list(high_curv_faces))
            
            for cluster in clusters:
                regions.append({
                    'type': 'saddle',
                    'faces': cluster,
                    'center': mesh.triangles_center[cluster].mean(axis=0),
                    'score': 0.8
                })
        
        return regions
    
    def _find_constrictions(self, mesh: trimesh.Trimesh, min_ratio: float) -> List[Dict]:
        """
        Find constrictions by analyzing cross-sectional areas
        """
        regions = []
        
        # Use simplified approach: find vertices with few connections
        # but high geodesic centrality (bottlenecks)
        
        # Build mesh graph
        graph = nx.Graph()
        for edge in mesh.edges:
            length = np.linalg.norm(mesh.vertices[edge[0]] - mesh.vertices[edge[1]])
            graph.add_edge(edge[0], edge[1], weight=length)
        
        # Sample vertices for analysis
        n_samples = min(500, len(mesh.vertices) // 10)
        sample_vertices = np.random.choice(len(mesh.vertices), n_samples, replace=False)
        
        # Compute betweenness centrality for samples
        centrality = nx.betweenness_centrality_subset(
            graph, 
            sources=sample_vertices[:n_samples//2],
            targets=sample_vertices[n_samples//2:],
            normalized=True
        )
        
        # Find high centrality vertices (potential bottlenecks)
        centrality_array = np.array([centrality.get(i, 0) for i in range(len(mesh.vertices))])
        threshold = np.percentile(centrality_array[centrality_array > 0], 80)
        
        bottleneck_vertices = np.where(centrality_array > threshold)[0]
        
        # Convert to face regions
        bottleneck_faces = set()
        for v in bottleneck_vertices:
            faces = mesh.vertex_faces[v]
            bottleneck_faces.update(f for f in faces if f >= 0)
        
        if bottleneck_faces:
            clusters = self._cluster_faces(mesh, list(bottleneck_faces))
            
            for cluster in clusters:
                regions.append({
                    'type': 'constriction',
                    'faces': cluster,
                    'center': mesh.triangles_center[cluster].mean(axis=0),
                    'score': 0.7
                })
        
        return regions
    
    def _cluster_faces(self, mesh: trimesh.Trimesh, faces: List[int], 
                      max_clusters: int = 10) -> List[List[int]]:
        """
        Cluster connected faces into regions
        """
        if not faces:
            return []
        
        # Build adjacency for selected faces
        face_set = set(faces)
        adjacency = {}
        
        for face in faces:
            adjacency[face] = []
            # Get adjacent faces through shared edges
            for edge in mesh.faces[face]:
                for adj_face in mesh.vertex_faces[edge]:
                    if adj_face >= 0 and adj_face != face and adj_face in face_set:
                        adjacency[face].append(adj_face)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for face in faces:
            if face not in visited:
                # BFS to find connected component
                cluster = []
                queue = [face]
                
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        queue.extend(adjacency.get(current, []))
                
                if cluster:
                    clusters.append(cluster)
        
        # Sort by size and limit number
        clusters.sort(key=len, reverse=True)
        return clusters[:max_clusters]
    
    def _merge_nearby_regions(self, regions: List[Dict], merge_distance: float) -> List[Dict]:
        """
        Merge regions that are close together
        """
        if not regions:
            return []
        
        # Build spatial index of region centers
        centers = np.array([r['center'] for r in regions])
        kdtree = KDTree(centers)
        
        # Find nearby regions
        merged = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            # Find regions within merge distance
            indices = kdtree.query_ball_point(region['center'], merge_distance)
            
            # Merge faces from all nearby regions
            merged_faces = []
            merged_score = 0
            
            for idx in indices:
                if idx not in used:
                    used.add(idx)
                    merged_faces.extend(regions[idx]['faces'])
                    merged_score = max(merged_score, regions[idx]['score'])
            
            if merged_faces:
                merged.append({
                    'type': 'merged',
                    'faces': list(set(merged_faces)),
                    'center': centers[indices].mean(axis=0),
                    'score': merged_score
                })
        
        return merged
    
    def _get_bridge_faces(self, mesh: trimesh.Trimesh, region: Dict, 
                         expansion_factor: float = 1.0) -> Set[int]:
        """
        Get faces to remove for a kissing artifact region
        """
        faces_to_remove = set(region['faces'])
        
        # Expand region based on detection factor - more aggressive expansion
        expansion_iterations = max(1, int(expansion_factor))
        
        for iteration in range(expansion_iterations):
            new_faces = set()
            for face in faces_to_remove:
                # Add adjacent faces
                for vertex in mesh.faces[face]:
                    adjacent_faces = mesh.vertex_faces[vertex]
                    for adj_face in adjacent_faces:
                        if adj_face >= 0 and adj_face not in faces_to_remove:
                            new_faces.add(adj_face)
            
            # Add new faces to removal set
            faces_to_remove.update(new_faces)
        
        return faces_to_remove
    
    def _final_smooth(self, input_mesh: Path, output_mesh: Path, iterations: int = 5) -> Path:
        """
        Final smoothing pass after kissing artifact removal
        """
        self.logger.info(f"Applying final Laplacian smoothing ({iterations} iterations)...")
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_mesh))
        
        # Light Laplacian smoothing to blend areas where artifacts were removed
        # This helps smooth out the regions where faces were removed
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=iterations)
        
        # Save result
        ms.save_current_mesh(str(output_mesh))
        
        self.logger.info("Final smoothing complete")
        return output_mesh
    
    def _tetrahedralize_with_ftetwild(self, surface_file: Path, output_file: Path,
                                     epsilon: float = 1e-3) -> Path:
        """
        Run fTetWild for tetrahedralization
        """
        self.logger.info(f"Running fTetWild (epsilon={epsilon})...")
        
        cmd = [
            str(self.ftetwild_path),
            "-i", str(surface_file),
            "-o", str(output_file),
            "-e", str(epsilon),
            "--max-its", "80",
            "--stop-energy", "10"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("fTetWild completed successfully")
            
            if self.verbose and result.stdout:
                for line in result.stdout.split('\n')[-10:]:
                    if line.strip():
                        self.logger.info(f"  {line}")
                        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"fTetWild failed: {e}")
            if e.stderr:
                self.logger.error(f"Error: {e.stderr}")
            raise
        
        return output_file


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mesh-based vessel processing pipeline"
    )
    parser.add_argument('input_nifti', help='Input NIfTI file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--smooth-iterations', type=int, default=20,
                       help='Initial smoothing iterations')
    parser.add_argument('--kissing-radius', type=float, default=3.0,
                       help='Detection radius for kissing artifacts (mm)')
    parser.add_argument('--tet-epsilon', type=float, default=1e-3,
                       help='fTetWild epsilon parameter')
    parser.add_argument('--n-jobs', type=int, default=32,
                       help='Number of parallel jobs (default: 32)')
    parser.add_argument('--detection-factor', type=float, default=1.0,
                       help='Detection threshold factor (>1 = more aggressive, default: 1.0)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MeshBasedVesselPipeline(n_jobs=args.n_jobs)
    results = pipeline.process_vessel(
        args.input_nifti,
        args.output_dir,
        initial_smooth_iterations=args.smooth_iterations,
        kissing_detection_radius=args.kissing_radius,
        tet_epsilon=args.tet_epsilon,
        detection_threshold_factor=args.detection_factor
    )
    
    print(f"\nPipeline complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 