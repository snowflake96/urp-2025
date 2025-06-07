#!/usr/bin/env python3
"""
Improved kissing artifact detection using topological analysis
"""

import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


class ImprovedKissingDetector:
    """
    Detect kissing artifacts using topological features rather than just thickness
    """
    
    def __init__(self, logger=None, n_jobs=32):
        self.logger = logger or logging.getLogger(__name__)
        self.n_jobs = n_jobs
    
    def detect_kissing_artifacts(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Main detection method combining multiple approaches - PARALLELIZED
        """
        self.logger.info(f"Detecting kissing artifacts using topological analysis (using {self.n_jobs} cores)...")
        
        artifacts = []
        
        # Run all detection methods in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all detection tasks
            self.logger.info("  Starting parallel analysis...")
            
            future_bottlenecks = executor.submit(self._detect_geodesic_bottlenecks, mesh)
            self.logger.info("    - Geodesic bottleneck analysis started")
            
            future_necks = executor.submit(self._detect_curvature_necks, mesh)
            self.logger.info("    - Curvature pattern analysis started")
            
            future_constrictions = executor.submit(self._detect_diameter_constrictions, mesh)
            self.logger.info("    - Diameter change analysis started")
            
            future_kinks = executor.submit(self._detect_sharp_kinks, mesh)
            self.logger.info("    - Vessel angle analysis started")
            
            # Collect results
            self.logger.info("  Collecting results...")
            
            bottlenecks = future_bottlenecks.result()
            self.logger.info(f"    - Found {len(bottlenecks)} geodesic bottlenecks")
            artifacts.extend(bottlenecks)
            
            necks = future_necks.result()
            self.logger.info(f"    - Found {len(necks)} curvature necks")
            artifacts.extend(necks)
            
            constrictions = future_constrictions.result()
            self.logger.info(f"    - Found {len(constrictions)} diameter constrictions")
            artifacts.extend(constrictions)
            
            kinks = future_kinks.result()
            self.logger.info(f"    - Found {len(kinks)} sharp kinks")
            artifacts.extend(kinks)
        
        # Merge and rank artifacts
        self.logger.info("  Merging and ranking artifacts...")
        merged_artifacts = self._merge_and_rank_artifacts(artifacts, mesh)
        
        return merged_artifacts
    
    def _detect_geodesic_bottlenecks(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Find bottlenecks using geodesic distance analysis
        """
        bottlenecks = []
        
        # Build mesh graph with edge weights as distances
        edges = mesh.edges_unique
        edge_lengths = mesh.edges_unique_length
        
        graph = nx.Graph()
        for i, edge in enumerate(edges):
            graph.add_edge(edge[0], edge[1], weight=edge_lengths[i])
        
        # Sample well-distributed points
        n_samples = min(50, len(mesh.vertices) // 100)
        sample_indices = self._get_well_distributed_samples(mesh, n_samples)
        
        # Compute betweenness centrality for bottleneck detection
        centrality = nx.betweenness_centrality_subset(
            graph,
            sources=sample_indices[:n_samples//2],
            targets=sample_indices[n_samples//2:],
            normalized=True,
            weight='weight'
        )
        
        # Find high centrality vertices (bottlenecks)
        centrality_values = np.array([centrality.get(i, 0) for i in range(len(mesh.vertices))])
        threshold = np.percentile(centrality_values[centrality_values > 0], 95)
        
        high_centrality_vertices = np.where(centrality_values > threshold)[0]
        
        if len(high_centrality_vertices) > 0:
            # Cluster high centrality vertices
            clusters = self._cluster_vertices(mesh, high_centrality_vertices)
            
            for cluster in clusters:
                # Get faces for this cluster
                faces = set()
                for v in cluster:
                    faces.update(f for f in mesh.vertex_faces[v] if f >= 0)
                
                if len(faces) > 10:  # Significant cluster
                    center = mesh.vertices[cluster].mean(axis=0)
                    bottlenecks.append({
                        'type': 'geodesic_bottleneck',
                        'vertices': cluster,
                        'faces': list(faces),
                        'center': center,
                        'score': centrality_values[cluster].mean(),
                        'size': len(faces)
                    })
        
        return bottlenecks
    
    def _detect_curvature_necks(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Detect neck-like regions using curvature analysis
        """
        necks = []
        
        # Compute discrete mean curvature
        mean_curvature = self._compute_mean_curvature(mesh)
        
        # Find saddle-like regions (negative mean curvature)
        saddle_threshold = np.percentile(mean_curvature[mean_curvature < 0], 10)
        saddle_vertices = np.where(mean_curvature < saddle_threshold)[0]
        
        if len(saddle_vertices) > 0:
            # Cluster saddle vertices
            clusters = self._cluster_vertices(mesh, saddle_vertices)
            
            for cluster in clusters:
                # Check if cluster forms a ring-like structure
                if self._is_ring_like(mesh, cluster):
                    faces = set()
                    for v in cluster:
                        faces.update(f for f in mesh.vertex_faces[v] if f >= 0)
                    
                    if len(faces) > 20:
                        center = mesh.vertices[cluster].mean(axis=0)
                        necks.append({
                            'type': 'curvature_neck',
                            'vertices': cluster,
                            'faces': list(faces),
                            'center': center,
                            'score': abs(mean_curvature[cluster].mean()),
                            'size': len(faces)
                        })
        
        return necks
    
    def _detect_diameter_constrictions(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Detect sudden diameter changes along vessel - PARALLELIZED
        """
        constrictions = []
        
        # Use mesh cross-sections to estimate local diameter
        # Sample along the principal axis
        bounds = mesh.bounds
        axis_length = np.linalg.norm(bounds[1] - bounds[0])
        n_slices = int(axis_length / 2.0)  # Sample every 2mm
        
        # Get principal axis using PCA
        vertices_centered = mesh.vertices - mesh.vertices.mean(axis=0)
        _, _, vh = np.linalg.svd(vertices_centered.T)
        principal_axis = vh[0]
        
        # Prepare slice parameters
        slice_params = []
        for i in range(n_slices):
            t = i / (n_slices - 1)
            position = bounds[0] + t * (bounds[1] - bounds[0])
            slice_params.append((i, position, principal_axis))
        
        def compute_slice_diameter(params):
            """Compute diameter for a single slice"""
            i, plane_origin, plane_normal = params
            try:
                section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
                if section is not None and isinstance(section, trimesh.Path3D):
                    section_bounds = section.bounds
                    diameter = np.linalg.norm(section_bounds[1] - section_bounds[0])
                    return (i, diameter, plane_origin)
            except:
                pass
            return None
        
        # Process slices sequentially to avoid memory issues
        # Mesh sectioning is memory intensive
        diameters = []
        positions = []
        
        for i, params in enumerate(slice_params):
            if i % 10 == 0:
                self.logger.debug(f"    Processing slice {i}/{len(slice_params)}")
            
            result = compute_slice_diameter(params)
            if result is not None:
                _, diameter, position = result
                diameters.append(diameter)
                positions.append(position)
        
        if len(diameters) > 5:
            diameters = np.array(diameters)
            
            # Find sudden diameter changes
            diameter_changes = np.abs(np.diff(diameters))
            mean_diameter = diameters.mean()
            
            # Significant change is > 30% of mean diameter
            significant_changes = np.where(diameter_changes > 0.3 * mean_diameter)[0]
            
            for idx in significant_changes:
                if 0 < idx < len(positions) - 1:
                    # Find vertices near this position
                    position = positions[idx]
                    distances = np.linalg.norm(mesh.vertices - position, axis=1)
                    near_vertices = np.where(distances < mean_diameter)[0]
                    
                    if len(near_vertices) > 10:
                        faces = set()
                        for v in near_vertices:
                            faces.update(f for f in mesh.vertex_faces[v] if f >= 0)
                        
                        constrictions.append({
                            'type': 'diameter_constriction',
                            'vertices': near_vertices,
                            'faces': list(faces),
                            'center': position,
                            'score': diameter_changes[idx] / mean_diameter,
                            'size': len(faces)
                        })
        
        return constrictions
    
    def _detect_sharp_kinks(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Detect sharp bends/kinks in vessel - PARALLELIZED
        """
        kinks = []
        
        # Analyze angle between adjacent face normals
        face_adjacency = mesh.face_adjacency
        face_adjacency_angles = mesh.face_adjacency_angles
        
        # Find sharp angles (> 60 degrees)
        sharp_angle_threshold = np.pi / 3
        sharp_edges = np.where(face_adjacency_angles > sharp_angle_threshold)[0]
        
        if len(sharp_edges) > 0:
            # Parallelize face collection
            def get_faces_batch(edge_indices):
                """Get faces for a batch of edges"""
                batch_faces = set()
                for edge_idx in edge_indices:
                    face_pair = face_adjacency[edge_idx]
                    batch_faces.update(face_pair)
                return batch_faces
            
            # Split edges into batches
            batch_size = max(100, len(sharp_edges) // self.n_jobs)
            batches = []
            for i in range(0, len(sharp_edges), batch_size):
                batch = sharp_edges[i:i + batch_size]
                batches.append(batch)
            
            # Process batches in parallel
            sharp_faces = set()
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(get_faces_batch, batch) for batch in batches]
                for future in futures:
                    sharp_faces.update(future.result())
            
            # Cluster sharp faces
            if sharp_faces:
                face_centers = mesh.triangles_center[list(sharp_faces)]
                clustering = DBSCAN(eps=5.0, min_samples=5).fit(face_centers)
                
                for label in set(clustering.labels_):
                    if label >= 0:
                        cluster_faces = [f for i, f in enumerate(sharp_faces) 
                                       if clustering.labels_[i] == label]
                        
                        if len(cluster_faces) > 10:
                            center = mesh.triangles_center[cluster_faces].mean(axis=0)
                            
                            # Get vertices
                            vertices = set()
                            for f in cluster_faces:
                                vertices.update(mesh.faces[f])
                            
                            kinks.append({
                                'type': 'sharp_kink',
                                'vertices': list(vertices),
                                'faces': cluster_faces,
                                'center': center,
                                'score': face_adjacency_angles[sharp_edges].max() / np.pi,
                                'size': len(cluster_faces)
                            })
        
        return kinks
    
    def _merge_and_rank_artifacts(self, artifacts: List[Dict], mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Merge nearby artifacts and rank by likelihood
        """
        if not artifacts:
            return []
        
        # Merge nearby artifacts
        centers = np.array([a['center'] for a in artifacts])
        merge_distance = 10.0  # mm
        
        merged = []
        used = set()
        
        for i, artifact in enumerate(artifacts):
            if i in used:
                continue
            
            # Find nearby artifacts
            distances = np.linalg.norm(centers - artifact['center'], axis=1)
            nearby = np.where(distances < merge_distance)[0]
            
            # Merge
            merged_vertices = set()
            merged_faces = set()
            types = []
            scores = []
            
            for j in nearby:
                if j not in used:
                    used.add(j)
                    merged_vertices.update(artifacts[j]['vertices'])
                    merged_faces.update(artifacts[j]['faces'])
                    types.append(artifacts[j]['type'])
                    scores.append(artifacts[j]['score'])
            
            # Combined score based on multiple detections
            combined_score = np.mean(scores) * len(types)
            
            merged.append({
                'vertices': list(merged_vertices),
                'faces': list(merged_faces),
                'center': centers[nearby].mean(axis=0),
                'types': types,
                'score': combined_score,
                'size': len(merged_faces)
            })
        
        # Sort by score
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        return merged
    
    def _compute_mean_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute discrete mean curvature at each vertex - PARALLELIZED
        """
        n_vertices = len(mesh.vertices)
        mean_curvature = np.zeros(n_vertices)
        
        def compute_vertex_curvature(vertex_indices):
            """Compute curvature for a batch of vertices"""
            local_curvatures = {}
            
            for i in vertex_indices:
                # Get one-ring neighborhood
                neighbors = mesh.vertex_neighbors[i]
                if len(neighbors) < 3:
                    local_curvatures[i] = 0
                    continue
                
                # Compute cotangent Laplacian
                vertex = mesh.vertices[i]
                laplacian = np.zeros(3)
                area = 0
                
                for j, neighbor in enumerate(neighbors):
                    # Get the two triangles sharing this edge
                    edge_vector = mesh.vertices[neighbor] - vertex
                    
                    # Approximate using uniform weights for simplicity
                    laplacian += edge_vector
                    area += np.linalg.norm(edge_vector)
                
                if area > 0:
                    laplacian /= area
                    curvature = np.linalg.norm(laplacian)
                    
                    # Sign based on normal direction
                    normal = mesh.vertex_normals[i]
                    if np.dot(laplacian, normal) < 0:
                        curvature *= -1
                    
                    local_curvatures[i] = curvature
                else:
                    local_curvatures[i] = 0
            
            return local_curvatures
        
        # Split vertices into batches
        batch_size = max(100, n_vertices // self.n_jobs)
        batches = []
        
        for i in range(0, n_vertices, batch_size):
            batch = list(range(i, min(i + batch_size, n_vertices)))
            batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(compute_vertex_curvature, batch) for batch in batches]
            
            for future in futures:
                batch_result = future.result()
                for idx, curv in batch_result.items():
                    mean_curvature[idx] = curv
        
        return mean_curvature
    
    def _get_well_distributed_samples(self, mesh: trimesh.Trimesh, n_samples: int) -> List[int]:
        """
        Get well-distributed sample vertices using farthest point sampling
        """
        samples = []
        distances = np.full(len(mesh.vertices), np.inf)
        
        # Start with random vertex
        current = np.random.randint(len(mesh.vertices))
        samples.append(current)
        
        for _ in range(n_samples - 1):
            # Update distances
            current_dists = np.linalg.norm(mesh.vertices - mesh.vertices[current], axis=1)
            distances = np.minimum(distances, current_dists)
            
            # Select farthest point
            current = np.argmax(distances)
            samples.append(current)
        
        return samples
    
    def _cluster_vertices(self, mesh: trimesh.Trimesh, vertices: np.ndarray, 
                         eps: float = 5.0) -> List[List[int]]:
        """
        Cluster vertices spatially
        """
        if len(vertices) == 0:
            return []
        
        positions = mesh.vertices[vertices]
        clustering = DBSCAN(eps=eps, min_samples=3).fit(positions)
        
        clusters = []
        for label in set(clustering.labels_):
            if label >= 0:
                cluster = vertices[clustering.labels_ == label]
                clusters.append(list(cluster))
        
        return clusters
    
    def _is_ring_like(self, mesh: trimesh.Trimesh, vertices: List[int]) -> bool:
        """
        Check if vertices form a ring-like structure
        """
        if len(vertices) < 8:
            return False
        
        # Check connectivity
        vertex_set = set(vertices)
        connected_count = 0
        
        for v in vertices:
            neighbors = mesh.vertex_neighbors[v]
            connected_neighbors = sum(1 for n in neighbors if n in vertex_set)
            if connected_neighbors >= 2:
                connected_count += 1
        
        # Most vertices should have 2+ connections within the set
        return connected_count > 0.7 * len(vertices)


def visualize_detection(mesh_file: str, output_file: str, n_jobs: int = 32):
    """
    Run detection and create visualization
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', 
                       datefmt='%H:%M:%S')
    
    # Show system info
    print(f"System info: {multiprocessing.cpu_count()} CPUs available")
    print(f"Will use {n_jobs} cores for parallel processing")
    
    # Load mesh
    print(f"\nLoading mesh from {mesh_file}...")
    mesh = trimesh.load(mesh_file)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Detect artifacts
    print(f"\nStarting detection with {n_jobs} cores...")
    detector = ImprovedKissingDetector(n_jobs=n_jobs)
    
    try:
        artifacts = detector.detect_kissing_artifacts(mesh)
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nFound {len(artifacts)} potential kissing artifacts:")
    for i, artifact in enumerate(artifacts[:5]):  # Show top 5
        print(f"\n{i+1}. Score: {artifact['score']:.2f}")
        print(f"   Types: {artifact.get('types', ['unknown'])}")
        print(f"   Size: {artifact['size']} faces")
        print(f"   Center: ({artifact['center'][0]:.1f}, "
              f"{artifact['center'][1]:.1f}, {artifact['center'][2]:.1f})")
    
    # Create visualization
    colors = np.ones((len(mesh.faces), 4)) * [0.8, 0.8, 0.8, 1.0]
    
    # Color artifacts by rank
    color_map = [
        [1.0, 0.0, 0.0, 1.0],  # Red - highest score
        [1.0, 0.5, 0.0, 1.0],  # Orange
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0],  # Blue - lower score
    ]
    
    for i, artifact in enumerate(artifacts[:5]):
        color = color_map[min(i, len(color_map)-1)]
        for face in artifact['faces']:
            if face < len(colors):
                colors[face] = color
    
    # Save colored mesh
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = colors
    colored_mesh.export(output_file)
    
    print(f"\nVisualization saved to {output_file}")
    print("Colors: Red=highest score → Blue=lower score, Gray=normal")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', help='Input mesh file')
    parser.add_argument('--output', default='kissing_artifacts_detected.ply',
                       help='Output visualization file')
    parser.add_argument('--n-jobs', type=int, default=32,
                       help='Number of parallel jobs (default: 32)')
    
    args = parser.parse_args()
    visualize_detection(args.mesh_file, args.output, args.n_jobs) 