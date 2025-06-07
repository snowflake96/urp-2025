#!/usr/bin/env python3
"""
Simplified kissing artifact detection focusing on self-intersections
"""

import numpy as np
import trimesh
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class SimpleKissingDetector:
    """
    Detect kissing artifacts by finding self-intersections and near-touches
    """
    
    def __init__(self, logger=None, n_jobs=32):
        self.logger = logger or logging.getLogger(__name__)
        self.n_jobs = n_jobs
    
    def detect_kissing_artifacts(self, mesh: trimesh.Trimesh, 
                                proximity_threshold: float = 2.0) -> List[Dict]:
        """
        Detect kissing artifacts using proximity analysis
        
        Parameters:
        - proximity_threshold: Distance (mm) below which non-adjacent faces are considered touching
        """
        self.logger.info(f"Detecting kissing artifacts (proximity < {proximity_threshold}mm, using {self.n_jobs} cores)...")
        
        # Method 1: Find self-intersections
        self.logger.info("  Checking for self-intersections...")
        intersections = self._find_self_intersections(mesh)
        
        # Method 2: Find near-touching regions
        self.logger.info("  Finding near-touching regions...")
        near_touches = self._find_near_touches(mesh, proximity_threshold)
        
        # Combine and cluster results
        all_artifacts = intersections + near_touches
        
        # Merge nearby artifacts
        if all_artifacts:
            merged = self._merge_artifacts(all_artifacts, mesh)
            self.logger.info(f"  Found {len(merged)} kissing artifact regions")
            return merged
        
        return []
    
    def _find_self_intersections(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Find regions where the mesh intersects with itself
        """
        artifacts = []
        
        # Use trimesh's built-in self-intersection detection
        try:
            # Check if mesh is self-intersecting
            if mesh.is_self_intersecting:
                self.logger.info("    Mesh has self-intersections!")
                
                # Find intersecting face pairs
                # This is expensive, so we sample
                n_samples = min(10000, len(mesh.faces))
                sample_faces = np.random.choice(len(mesh.faces), n_samples, replace=False)
                
                intersecting_pairs = []
                
                # Check face pairs in batches
                def check_face_batch(face_indices):
                    local_pairs = []
                    for i, face_idx in enumerate(face_indices):
                        if i % 100 == 0:
                            self.logger.debug(f"      Checking face {i}/{len(face_indices)}")
                        
                        # Get face triangle
                        triangle = mesh.triangles[face_idx]
                        
                        # Find potentially intersecting faces (nearby but not adjacent)
                        center = triangle.mean(axis=0)
                        nearby_faces = mesh.triangles_tree.query_ball_point(center, r=10.0)
                        
                        for other_idx in nearby_faces:
                            if other_idx != face_idx:
                                # Check if faces share vertices (adjacent)
                                face_verts = set(mesh.faces[face_idx])
                                other_verts = set(mesh.faces[other_idx])
                                
                                if not face_verts.intersection(other_verts):
                                    # Non-adjacent faces - check intersection
                                    other_triangle = mesh.triangles[other_idx]
                                    
                                    # Simple proximity check
                                    dist = np.min(np.linalg.norm(
                                        triangle[:, np.newaxis, :] - other_triangle[np.newaxis, :, :], 
                                        axis=2
                                    ))
                                    
                                    if dist < 0.1:  # Very close
                                        local_pairs.append((face_idx, other_idx, dist))
                    
                    return local_pairs
                
                # Process in parallel
                batch_size = max(100, n_samples // self.n_jobs)
                batches = []
                
                for i in range(0, len(sample_faces), batch_size):
                    batch = sample_faces[i:i + batch_size]
                    batches.append(batch)
                
                with ThreadPoolExecutor(max_workers=min(8, self.n_jobs)) as executor:
                    futures = [executor.submit(check_face_batch, batch) for batch in batches]
                    
                    for future in futures:
                        intersecting_pairs.extend(future.result())
                
                # Cluster intersecting faces
                if intersecting_pairs:
                    self.logger.info(f"    Found {len(intersecting_pairs)} intersecting face pairs")
                    
                    # Extract unique faces
                    intersecting_faces = set()
                    for f1, f2, dist in intersecting_pairs:
                        intersecting_faces.add(f1)
                        intersecting_faces.add(f2)
                    
                    # Create artifact entry
                    center = mesh.triangles_center[list(intersecting_faces)].mean(axis=0)
                    artifacts.append({
                        'type': 'self_intersection',
                        'faces': list(intersecting_faces),
                        'center': center,
                        'score': 1.0,  # High confidence
                        'size': len(intersecting_faces)
                    })
        except Exception as e:
            self.logger.warning(f"    Error checking self-intersections: {e}")
        
        return artifacts
    
    def _find_near_touches(self, mesh: trimesh.Trimesh, threshold: float) -> List[Dict]:
        """
        Find regions where non-adjacent faces are very close
        """
        artifacts = []
        
        # Sample points on mesh surface
        n_samples = min(20000, len(mesh.faces) * 10)
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, count=n_samples)
        
        self.logger.info(f"    Sampling {n_samples} points on surface...")
        
        # Build KDTree for fast proximity queries
        tree = KDTree(samples)
        
        # Find close point pairs
        close_pairs = tree.query_pairs(r=threshold)
        self.logger.info(f"    Found {len(close_pairs)} close point pairs")
        
        if close_pairs:
            # Filter to non-adjacent faces
            non_adjacent_pairs = []
            
            for i, j in close_pairs:
                face_i = face_indices[i]
                face_j = face_indices[j]
                
                if face_i != face_j:
                    # Check if faces are adjacent
                    verts_i = set(mesh.faces[face_i])
                    verts_j = set(mesh.faces[face_j])
                    
                    if not verts_i.intersection(verts_j):
                        # Non-adjacent
                        distance = np.linalg.norm(samples[i] - samples[j])
                        non_adjacent_pairs.append({
                            'faces': (face_i, face_j),
                            'points': (samples[i], samples[j]),
                            'distance': distance
                        })
            
            self.logger.info(f"    Found {len(non_adjacent_pairs)} non-adjacent close pairs")
            
            if non_adjacent_pairs:
                # Cluster close faces
                close_faces = set()
                for pair in non_adjacent_pairs:
                    close_faces.update(pair['faces'])
                
                close_faces = list(close_faces)
                
                # Use DBSCAN to cluster spatially
                if len(close_faces) > 10:
                    face_centers = mesh.triangles_center[close_faces]
                    clustering = DBSCAN(eps=5.0, min_samples=5).fit(face_centers)
                    
                    for label in set(clustering.labels_):
                        if label >= 0:
                            cluster_mask = clustering.labels_ == label
                            cluster_faces = [close_faces[i] for i in range(len(close_faces)) if cluster_mask[i]]
                            
                            if len(cluster_faces) > 10:
                                center = face_centers[cluster_mask].mean(axis=0)
                                
                                # Calculate average proximity
                                avg_dist = np.mean([p['distance'] for p in non_adjacent_pairs 
                                                  if any(f in cluster_faces for f in p['faces'])])
                                
                                artifacts.append({
                                    'type': 'near_touch',
                                    'faces': cluster_faces,
                                    'center': center,
                                    'score': (threshold - avg_dist) / threshold,  # Closer = higher score
                                    'size': len(cluster_faces),
                                    'avg_distance': avg_dist
                                })
        
        return artifacts
    
    def _merge_artifacts(self, artifacts: List[Dict], mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Merge nearby artifacts
        """
        if not artifacts:
            return []
        
        # Sort by score
        artifacts.sort(key=lambda x: x['score'], reverse=True)
        
        # Merge overlapping artifacts
        merged = []
        used_faces = set()
        
        for artifact in artifacts:
            artifact_faces = set(artifact['faces'])
            
            # Check overlap with already used faces
            overlap = artifact_faces.intersection(used_faces)
            
            if len(overlap) < len(artifact_faces) * 0.5:  # Less than 50% overlap
                # Keep this artifact
                new_faces = artifact_faces - used_faces
                if len(new_faces) > 5:
                    used_faces.update(new_faces)
                    
                    # Update artifact with non-overlapping faces
                    artifact['faces'] = list(new_faces)
                    artifact['size'] = len(new_faces)
                    merged.append(artifact)
        
        return merged


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified kissing artifact detection")
    parser.add_argument('mesh_file', help='Input mesh file')
    parser.add_argument('--output', default='simple_kissing_detection.ply',
                       help='Output visualization file')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Proximity threshold in mm (default: 2.0)')
    parser.add_argument('--n-jobs', type=int, default=32,
                       help='Number of parallel jobs (default: 32)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', 
                       datefmt='%H:%M:%S')
    
    print(f"System info: {multiprocessing.cpu_count()} CPUs available")
    print(f"Will use {args.n_jobs} cores for parallel processing")
    
    # Load mesh
    print(f"\nLoading mesh from {args.mesh_file}...")
    mesh = trimesh.load(args.mesh_file)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Detect artifacts
    detector = SimpleKissingDetector(n_jobs=args.n_jobs)
    artifacts = detector.detect_kissing_artifacts(mesh, proximity_threshold=args.threshold)
    
    print(f"\nFound {len(artifacts)} kissing artifact regions:")
    for i, artifact in enumerate(artifacts):
        print(f"\n{i+1}. Type: {artifact['type']}")
        print(f"   Score: {artifact['score']:.2f}")
        print(f"   Size: {artifact['size']} faces")
        print(f"   Center: ({artifact['center'][0]:.1f}, "
              f"{artifact['center'][1]:.1f}, {artifact['center'][2]:.1f})")
        if 'avg_distance' in artifact:
            print(f"   Avg distance: {artifact['avg_distance']:.2f}mm")
    
    # Create visualization
    if artifacts:
        colors = np.ones((len(mesh.faces), 4)) * [0.8, 0.8, 0.8, 1.0]
        
        # Color artifacts
        color_map = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [1.0, 0.5, 0.0, 1.0],  # Orange
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
        ]
        
        for i, artifact in enumerate(artifacts[:5]):
            color = color_map[min(i, len(color_map)-1)]
            for face in artifact['faces']:
                if face < len(colors):
                    colors[face] = color
        
        # Save colored mesh
        colored_mesh = mesh.copy()
        colored_mesh.visual.face_colors = colors
        colored_mesh.export(args.output)
        
        print(f"\nVisualization saved to {args.output}")
        print("Colors: Red=highest score → Blue=lower score, Gray=normal")
    else:
        print("\nNo kissing artifacts detected!")
        print("This suggests the mesh is clean or the artifacts are at the voxel level, not mesh level.")


if __name__ == "__main__":
    main() 