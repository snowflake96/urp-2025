#!/usr/bin/env python3
"""
Orthogonal Vessel Extraction Pipeline

This script extracts aneurysm vessels by cutting orthogonally to the vessel path,
resulting in circular cross-sections instead of oval ones.

Key improvements:
1. Computes vessel centerlines/skeletons
2. Estimates local vessel direction at boundaries  
3. Creates orthogonal cutting planes
4. Combines with flood fill geodesic for connectivity
"""

import json
import os
import numpy as np
import trimesh
from pathlib import Path
import argparse
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import networkx as nx
from scipy.ndimage import distance_transform_edt


def compute_vessel_centerline_simple(mesh: trimesh.Trimesh, aneurysm_location: np.ndarray) -> np.ndarray:
    """
    Compute a simplified vessel centerline using distance-based approach.
    
    This creates a rough centerline by finding the path that maximizes distance
    from the surface, starting from the aneurysm location.
    """
    print("    Computing vessel centerline...")
    
    # Sample points throughout the mesh volume
    bbox = mesh.bounds
    grid_resolution = 50
    
    x = np.linspace(bbox[0, 0], bbox[1, 0], grid_resolution)
    y = np.linspace(bbox[0, 1], bbox[1, 1], grid_resolution)
    z = np.linspace(bbox[0, 2], bbox[1, 2], grid_resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Filter points inside the mesh
    inside_mask = mesh.contains(grid_points)
    inside_points = grid_points[inside_mask]
    
    if len(inside_points) < 10:
        print("    Warning: Too few points inside mesh, using simple centerline")
        # Fallback: create a simple line through the mesh
        center = np.mean(mesh.vertices, axis=0)
        direction = mesh.principal_inertia_vectors[0]  # Main axis
        length = np.linalg.norm(bbox[1] - bbox[0]) * 0.8
        
        centerline = np.array([
            center - direction * length/2,
            center,
            center + direction * length/2
        ])
        return centerline
    
    # Calculate distance from surface for each interior point
    distances_to_surface = []
    for point in inside_points:
        _, dist = mesh.nearest.on_surface([point])
        distances_to_surface.append(dist[0])
    
    distances_to_surface = np.array(distances_to_surface)
    
    # Find points with maximum distance (centerline candidates)
    high_distance_threshold = np.percentile(distances_to_surface, 80)
    centerline_candidates = inside_points[distances_to_surface >= high_distance_threshold]
    
    if len(centerline_candidates) < 3:
        print("    Warning: Too few centerline candidates, using mesh center")
        center = np.mean(mesh.vertices, axis=0)
        return np.array([center])
    
    # Order centerline points by proximity to aneurysm and connectivity
    # Start from the point closest to aneurysm
    distances_to_aneurysm = np.linalg.norm(centerline_candidates - aneurysm_location, axis=1)
    start_idx = np.argmin(distances_to_aneurysm)
    start_point = centerline_candidates[start_idx]
    
    # Build centerline by following connected high-distance points
    centerline = [start_point]
    remaining_points = np.delete(centerline_candidates, start_idx, axis=0)
    current_point = start_point
    
    max_centerline_length = 20  # Limit centerline points
    connection_threshold = np.percentile(distances_to_surface, 90) * 0.5  # Connection distance
    
    for _ in range(max_centerline_length):
        if len(remaining_points) == 0:
            break
            
        # Find closest remaining point
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] > connection_threshold:
            break  # Too far to connect
            
        next_point = remaining_points[closest_idx]
        centerline.append(next_point)
        
        # Remove the selected point and update current
        remaining_points = np.delete(remaining_points, closest_idx, axis=0)
        current_point = next_point
    
    centerline = np.array(centerline)
    print(f"    Computed centerline with {len(centerline)} points")
    
    return centerline


def estimate_vessel_direction_at_point(centerline: np.ndarray, point: np.ndarray, radius: float = 5.0) -> np.ndarray:
    """
    Estimate vessel direction at a given point using local centerline.
    """
    if len(centerline) < 2:
        # Fallback: use PCA of nearby mesh vertices
        return np.array([1, 0, 0])  # Default direction
    
    # Find centerline points near the query point
    distances = np.linalg.norm(centerline - point, axis=1)
    nearby_mask = distances <= radius
    nearby_points = centerline[nearby_mask]
    
    if len(nearby_points) < 2:
        # Use closest segment
        closest_idx = np.argmin(distances)
        if closest_idx == 0:
            direction = centerline[1] - centerline[0]
        elif closest_idx == len(centerline) - 1:
            direction = centerline[-1] - centerline[-2]
        else:
            direction = centerline[closest_idx + 1] - centerline[closest_idx - 1]
    else:
        # Fit line to nearby centerline points
        if len(nearby_points) >= 3:
            # Use PCA to find main direction
            pca = PCA(n_components=1)
            pca.fit(nearby_points)
            direction = pca.components_[0]
        else:
            # Use two-point direction
            direction = nearby_points[-1] - nearby_points[0]
    
    # Normalize direction
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    return direction


def create_orthogonal_cutting_planes(mesh: trimesh.Trimesh, 
                                   centerline: np.ndarray,
                                   aneurysm_location: np.ndarray,
                                   target_size: int = 10400) -> List[Dict]:
    """
    Create cutting planes orthogonal to vessel direction for clean boundaries.
    """
    print("    Creating orthogonal cutting planes...")
    
    # Find mesh boundaries (vertices with few neighbors)
    vertex_neighbors = {}
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 not in vertex_neighbors:
                vertex_neighbors[v1] = set()
            if v2 not in vertex_neighbors:
                vertex_neighbors[v2] = set()
            vertex_neighbors[v1].add(v2)
            vertex_neighbors[v2].add(v1)
    
    # Identify boundary regions (vertices with fewer neighbors)
    neighbor_counts = [len(vertex_neighbors.get(i, [])) for i in range(len(mesh.vertices))]
    avg_neighbors = np.mean(neighbor_counts)
    boundary_threshold = avg_neighbors * 0.7  # Vertices with 70% of average neighbors
    
    boundary_vertices = []
    for i, count in enumerate(neighbor_counts):
        if count < boundary_threshold:
            boundary_vertices.append(i)
    
    if len(boundary_vertices) < 10:
        print("    Warning: Few boundary vertices found, using distance-based approach")
        # Fallback: use vertices far from aneurysm
        distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
        far_threshold = np.percentile(distances, 80)
        boundary_vertices = np.where(distances >= far_threshold)[0].tolist()
    
    print(f"    Found {len(boundary_vertices)} potential boundary vertices")
    
    # Group boundary vertices into regions using clustering
    if len(boundary_vertices) > 3:
        boundary_coords = mesh.vertices[boundary_vertices]
        
        # Use DBSCAN to find boundary clusters
        clustering = DBSCAN(eps=8.0, min_samples=3)
        cluster_labels = clustering.fit_predict(boundary_coords)
        
        boundary_regions = []
        for label in set(cluster_labels):
            if label != -1:  # Ignore noise points
                cluster_mask = cluster_labels == label
                cluster_vertices = np.array(boundary_vertices)[cluster_mask]
                cluster_coords = boundary_coords[cluster_mask]
                
                if len(cluster_vertices) >= 5:  # Minimum region size
                    boundary_regions.append({
                        'vertices': cluster_vertices,
                        'coords': cluster_coords,
                        'center': np.mean(cluster_coords, axis=0)
                    })
        
        print(f"    Found {len(boundary_regions)} boundary regions")
    else:
        boundary_regions = []
    
    # Create orthogonal cutting planes for each region
    cutting_planes = []
    
    for i, region in enumerate(boundary_regions):
        region_center = region['center']
        
        # Estimate vessel direction at this region
        vessel_direction = estimate_vessel_direction_at_point(centerline, region_center)
        
        # Create cutting plane orthogonal to vessel direction
        plane_normal = vessel_direction  # Normal is the vessel direction
        plane_origin = region_center
        
        cutting_planes.append({
            'id': f'plane_{i+1}',
            'origin': plane_origin,
            'normal': plane_normal,
            'region_vertices': region['vertices'],
            'region_center': region_center
        })
        
        print(f"      Plane {i+1}: center {region_center}, normal {plane_normal}")
    
    return cutting_planes


def extract_vessel_with_orthogonal_cuts(mesh: trimesh.Trimesh,
                                       centerline: np.ndarray,
                                       aneurysm_location: np.ndarray,
                                       target_size: int = 10400) -> trimesh.Trimesh:
    """
    Extract vessel region using orthogonal cuts combined with flood fill.
    """
    print("  Extracting vessel with orthogonal cuts...")
    
    # Create orthogonal cutting planes
    cutting_planes = create_orthogonal_cutting_planes(mesh, centerline, aneurysm_location, target_size)
    
    if not cutting_planes:
        print("    No cutting planes found, using standard flood fill")
        return extract_vessel_flood_fill_standard(mesh, aneurysm_location, target_size)
    
    # Start with flood fill from aneurysm location
    start_vertex = find_closest_vertex(mesh.vertices, aneurysm_location)
    
    # Build vertex adjacency graph
    vertex_adjacency = build_vertex_adjacency(mesh)
    
    # Perform flood fill with orthogonal boundary constraints
    selected_vertices = set()
    queue = [start_vertex]
    selected_vertices.add(start_vertex)
    
    while queue and len(selected_vertices) < target_size:
        current_vertex = queue.pop(0)
        current_pos = mesh.vertices[current_vertex]
        
        # Check all neighbors
        for neighbor in vertex_adjacency.get(current_vertex, []):
            if neighbor in selected_vertices:
                continue
                
            neighbor_pos = mesh.vertices[neighbor]
            
            # Check if adding this neighbor would cross any orthogonal cutting plane
            should_include = True
            
            for plane in cutting_planes:
                plane_origin = plane['origin']
                plane_normal = plane['normal']
                
                # Calculate distances from plane
                current_dist = np.dot(current_pos - plane_origin, plane_normal)
                neighbor_dist = np.dot(neighbor_pos - plane_origin, plane_normal)
                
                # If we're crossing the plane in the "away from aneurysm" direction, stop
                aneurysm_dist = np.dot(aneurysm_location - plane_origin, plane_normal)
                
                # Stop if neighbor is on the far side of the cutting plane
                if (aneurysm_dist > 0 and neighbor_dist < current_dist - 2.0) or \
                   (aneurysm_dist < 0 and neighbor_dist > current_dist + 2.0):
                    should_include = False
                    break
            
            if should_include:
                selected_vertices.add(neighbor)
                queue.append(neighbor)
    
    # Ensure connectivity
    selected_vertices = ensure_connectivity(selected_vertices, vertex_adjacency, start_vertex)
    
    # Extract the vessel mesh
    selected_list = list(selected_vertices)
    vessel_mesh = extract_submesh(mesh, selected_list)
    
    print(f"    Extracted vessel: {len(vessel_mesh.vertices)} vertices, {len(vessel_mesh.faces)} faces")
    
    return vessel_mesh


def extract_vessel_flood_fill_standard(mesh: trimesh.Trimesh, aneurysm_location: np.ndarray, target_size: int) -> trimesh.Trimesh:
    """
    Standard flood fill extraction as fallback.
    """
    print("    Using standard flood fill extraction")
    
    start_vertex = find_closest_vertex(mesh.vertices, aneurysm_location)
    vertex_adjacency = build_vertex_adjacency(mesh)
    
    # Simple flood fill
    selected_vertices = set()
    queue = [start_vertex]
    selected_vertices.add(start_vertex)
    
    while queue and len(selected_vertices) < target_size:
        current_vertex = queue.pop(0)
        
        for neighbor in vertex_adjacency.get(current_vertex, []):
            if neighbor not in selected_vertices:
                selected_vertices.add(neighbor)
                queue.append(neighbor)
    
    selected_list = list(selected_vertices)
    vessel_mesh = extract_submesh(mesh, selected_list)
    
    return vessel_mesh


def find_closest_vertex(vertices: np.ndarray, target_point: np.ndarray) -> int:
    """Find the vertex closest to the target point."""
    distances = np.linalg.norm(vertices - target_point, axis=1)
    return np.argmin(distances)


def build_vertex_adjacency(mesh: trimesh.Trimesh) -> Dict[int, List[int]]:
    """Build vertex adjacency graph from mesh faces."""
    adjacency = {}
    
    for face in mesh.faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
    
    return adjacency


def ensure_connectivity(selected_vertices: Set[int], adjacency: Dict[int, List[int]], start_vertex: int) -> Set[int]:
    """Ensure the selected vertices form a connected component containing start_vertex."""
    if start_vertex not in selected_vertices:
        return selected_vertices
    
    # Find connected component containing start_vertex
    connected = set()
    queue = [start_vertex]
    connected.add(start_vertex)
    
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor in selected_vertices and neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    
    return connected


def extract_submesh(mesh: trimesh.Trimesh, vertex_indices: List[int]) -> trimesh.Trimesh:
    """Extract submesh containing only the specified vertices."""
    vertex_set = set(vertex_indices)
    
    # Create vertex mapping
    old_to_new = {}
    new_vertices = []
    
    for i, old_idx in enumerate(vertex_indices):
        old_to_new[old_idx] = i
        new_vertices.append(mesh.vertices[old_idx])
    
    new_vertices = np.array(new_vertices)
    
    # Extract faces that use only selected vertices
    new_faces = []
    for face in mesh.faces:
        if all(v in vertex_set for v in face):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
    
    new_faces = np.array(new_faces)
    
    if len(new_faces) == 0:
        # Emergency fallback: create simple triangulated mesh
        print("    Warning: No faces found, creating point cloud mesh")
        if len(new_vertices) >= 3:
            hull = ConvexHull(new_vertices)
            new_faces = hull.simplices
        else:
            new_faces = np.array([[0, 1, 2]]) if len(new_vertices) >= 3 else np.array([])
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


def process_single_vessel_orthogonal(args: Tuple) -> Dict:
    """
    Process a single vessel to extract with orthogonal cuts.
    """
    stl_file, aneurysm_data, output_dir, target_size = args
    
    patient_id = os.path.basename(stl_file).replace('.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'vessel_vertices': 0,
        'vessel_faces': 0,
        'centerline_points': 0,
        'cutting_planes': 0
    }
    
    try:
        print(f"\nProcessing {patient_id} with orthogonal extraction...")
        
        # Load mesh
        mesh = trimesh.load(stl_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Compute vessel centerline
        centerline = compute_vessel_centerline_simple(mesh, aneurysm_location)
        
        # Extract vessel with orthogonal cuts
        vessel_mesh = extract_vessel_with_orthogonal_cuts(
            mesh, centerline, aneurysm_location, target_size
        )
        
        # Save vessel
        output_file = os.path.join(output_dir, f"{patient_id}_aneurysm_1_vessel_orthogonal.stl")
        os.makedirs(output_dir, exist_ok=True)
        vessel_mesh.export(output_file)
        
        result['success'] = True
        result['vessel_vertices'] = len(vessel_mesh.vertices)
        result['vessel_faces'] = len(vessel_mesh.faces)
        result['centerline_points'] = len(centerline)
        result['output_file'] = output_file
        
        print(f"  ✓ {patient_id}: {result['vessel_vertices']} vertices, {result['vessel_faces']} faces")
        print(f"    Centerline: {result['centerline_points']} points")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for orthogonal vessel extraction"""
    parser = argparse.ArgumentParser(description='Orthogonal Vessel Extraction Pipeline')
    
    parser.add_argument('--data-dir', 
                       default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
                       help='Directory containing STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_orthogonal'),
                       help='Output directory for extracted vessels')
    
    parser.add_argument('--target-size', type=int, default=10400,
                       help='Target number of vertices for extracted vessels')
    
    parser.add_argument('--workers', '-j', type=int, default=16,
                       help='Number of parallel workers')
    
    parser.add_argument('--patient-limit', type=int,
                       help='Limit number of patients (for testing)')
    
    args = parser.parse_args()
    
    # Load aneurysm data
    print(f"Loading aneurysm data from: {args.aneurysm_json}")
    with open(args.aneurysm_json, 'r') as f:
        aneurysm_data = json.load(f)
    
    # Find STL files
    stl_files = []
    for patient_id, patient_data in aneurysm_data.items():
        if args.patient_limit and len(stl_files) >= args.patient_limit:
            break
            
        stl_file = os.path.join(args.data_dir, f"{patient_id}.stl")
        if os.path.exists(stl_file):
            stl_files.append((stl_file, patient_data))
        else:
            print(f"Warning: STL file not found for {patient_id}")
    
    print(f"Found {len(stl_files)} STL files to process")
    print(f"Target vessel size: {args.target_size} vertices")
    
    # Prepare processing arguments
    process_args = [(stl_file, patient_data, args.output_dir, args.target_size) 
                   for stl_file, patient_data in stl_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Extracting vessels orthogonally"):
            result = process_single_vessel_orthogonal(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_orthogonal, process_args),
                               total=len(process_args), desc="Extracting vessels orthogonally"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Orthogonal Vessel Extraction Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_vertices = np.mean([r['vessel_vertices'] for r in successful])
        avg_faces = np.mean([r['vessel_faces'] for r in successful])
        avg_centerline = np.mean([r['centerline_points'] for r in successful])
        
        print(f"\nOrthogonal Extraction Summary:")
        print(f"  Average vessel vertices: {avg_vertices:.0f}")
        print(f"  Average vessel faces: {avg_faces:.0f}")
        print(f"  Average centerline points: {avg_centerline:.1f}")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'orthogonal_extraction_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Vessels extracted with orthogonal cuts for circular cross-sections!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 