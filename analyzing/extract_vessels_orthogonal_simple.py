#!/usr/bin/env python3
"""
Simplified Orthogonal Vessel Extraction

This script improves vessel extraction by estimating local vessel direction
and constraining flood fill to create more circular cross-sections.

Key improvements:
1. Estimates local vessel direction using PCA of nearby vertices
2. Constrains flood fill to respect vessel orientation  
3. Creates more circular openings instead of oval ones
4. Fast and practical approach without complex centerline computation
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
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def estimate_local_vessel_direction(mesh: trimesh.Trimesh, point: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """
    Estimate local vessel direction using PCA of nearby vertices.
    """
    # Find vertices within radius
    distances = np.linalg.norm(mesh.vertices - point, axis=1)
    nearby_mask = distances <= radius
    nearby_vertices = mesh.vertices[nearby_mask]
    
    if len(nearby_vertices) < 10:
        # Fallback: use vessel's principal direction
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        return pca.components_[0]  # Main direction
    
    # Use PCA to find main direction of nearby vertices
    pca = PCA(n_components=3)
    pca.fit(nearby_vertices)
    
    # Return the principal direction (first component)
    direction = pca.components_[0]
    
    return direction / np.linalg.norm(direction)


def extract_vessel_with_directional_constraints(mesh: trimesh.Trimesh,
                                               aneurysm_location: np.ndarray,
                                               target_size: int = 10400) -> trimesh.Trimesh:
    """
    Extract vessel using flood fill with directional constraints for circular openings.
    """
    print("  Extracting vessel with directional constraints...")
    
    # Find starting vertex closest to aneurysm
    distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    start_vertex = np.argmin(distances)
    
    # Build vertex adjacency graph
    vertex_adjacency = build_vertex_adjacency_fast(mesh)
    
    # Estimate vessel direction at aneurysm location
    vessel_direction = estimate_local_vessel_direction(mesh, aneurysm_location)
    print(f"    Estimated vessel direction: {vessel_direction}")
    
    # Perform constrained flood fill
    selected_vertices = set()
    queue = [start_vertex]
    selected_vertices.add(start_vertex)
    
    # Parameters for directional constraints
    max_deviation_angle = np.pi / 3  # 60 degrees maximum deviation
    boundary_detection_threshold = 15.0  # Distance to consider as boundary
    
    while queue and len(selected_vertices) < target_size:
        current_vertex = queue.pop(0)
        current_pos = mesh.vertices[current_vertex]
        
        # Estimate local vessel direction at current position
        local_direction = estimate_local_vessel_direction(mesh, current_pos, radius=6.0)
        
        # Check all neighbors
        for neighbor in vertex_adjacency.get(current_vertex, []):
            if neighbor in selected_vertices:
                continue
                
            neighbor_pos = mesh.vertices[neighbor]
            
            # Vector from current to neighbor
            step_vector = neighbor_pos - current_pos
            step_length = np.linalg.norm(step_vector)
            
            if step_length < 1e-6:
                continue
                
            step_direction = step_vector / step_length
            
            # Check if we're at a potential boundary (far from aneurysm)
            distance_from_aneurysm = np.linalg.norm(neighbor_pos - aneurysm_location)
            
            if distance_from_aneurysm > boundary_detection_threshold:
                # We're at a potential boundary - apply directional constraints
                
                # Calculate angle between step direction and local vessel direction
                dot_product = np.clip(np.dot(step_direction, local_direction), -1.0, 1.0)
                angle = np.arccos(abs(dot_product))  # Use absolute for bidirectional
                
                # Only include if the step is roughly along the vessel direction
                if angle > max_deviation_angle:
                    continue  # Skip this neighbor - it's too far off the vessel direction
            
            # Include this neighbor
            selected_vertices.add(neighbor)
            queue.append(neighbor)
    
    # Ensure connectivity
    selected_vertices = ensure_connectivity_simple(selected_vertices, vertex_adjacency, start_vertex)
    
    # Extract submesh
    selected_list = list(selected_vertices)
    vessel_mesh = extract_submesh_fast(mesh, selected_list)
    
    print(f"    Extracted vessel: {len(vessel_mesh.vertices)} vertices, {len(vessel_mesh.faces)} faces")
    
    return vessel_mesh


def build_vertex_adjacency_fast(mesh: trimesh.Trimesh) -> Dict[int, List[int]]:
    """Build vertex adjacency graph efficiently."""
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


def ensure_connectivity_simple(selected_vertices: Set[int], 
                              adjacency: Dict[int, List[int]], 
                              start_vertex: int) -> Set[int]:
    """Ensure selected vertices form a connected component."""
    if start_vertex not in selected_vertices:
        return selected_vertices
    
    # BFS to find connected component
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


def extract_submesh_fast(mesh: trimesh.Trimesh, vertex_indices: List[int]) -> trimesh.Trimesh:
    """Extract submesh efficiently."""
    vertex_set = set(vertex_indices)
    
    # Create new vertices
    old_to_new = {old_idx: i for i, old_idx in enumerate(vertex_indices)}
    new_vertices = mesh.vertices[vertex_indices]
    
    # Extract faces
    new_faces = []
    for face in mesh.faces:
        if all(v in vertex_set for v in face):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
    
    new_faces = np.array(new_faces) if new_faces else np.array([]).reshape(0, 3)
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


def process_single_vessel_directional(args: Tuple) -> Dict:
    """
    Process a single vessel with directional constraints.
    """
    stl_file, aneurysm_data, output_dir, target_size = args
    
    patient_id = os.path.basename(stl_file).replace('.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'vessel_vertices': 0,
        'vessel_faces': 0
    }
    
    try:
        print(f"\nProcessing {patient_id} with directional constraints...")
        
        # Load mesh
        mesh = trimesh.load(stl_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Extract vessel with directional constraints
        vessel_mesh = extract_vessel_with_directional_constraints(
            mesh, aneurysm_location, target_size
        )
        
        # Save vessel
        output_file = os.path.join(output_dir, f"{patient_id}_aneurysm_1_vessel_directional.stl")
        os.makedirs(output_dir, exist_ok=True)
        vessel_mesh.export(output_file)
        
        result['success'] = True
        result['vessel_vertices'] = len(vessel_mesh.vertices)
        result['vessel_faces'] = len(vessel_mesh.faces)
        result['output_file'] = output_file
        
        print(f"  ✓ {patient_id}: {result['vessel_vertices']} vertices, {result['vessel_faces']} faces")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for directional vessel extraction"""
    parser = argparse.ArgumentParser(description='Directional Vessel Extraction Pipeline')
    
    parser.add_argument('--data-dir', 
                       default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
                       help='Directory containing STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_directional'),
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
        for process_arg in tqdm(process_args, desc="Extracting vessels with direction"):
            result = process_single_vessel_directional(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_directional, process_args),
                               total=len(process_args), desc="Extracting vessels with direction"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Directional Vessel Extraction Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_vertices = np.mean([r['vessel_vertices'] for r in successful])
        avg_faces = np.mean([r['vessel_faces'] for r in successful])
        
        print(f"\nDirectional Extraction Summary:")
        print(f"  Average vessel vertices: {avg_vertices:.0f}")
        print(f"  Average vessel faces: {avg_faces:.0f}")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'directional_extraction_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Vessels extracted with directional constraints for more circular openings!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 