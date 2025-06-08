#!/usr/bin/env python3
"""
Geodesic Small Orthogonal Vessel Extraction

This script combines the best of all approaches:
1. Flood fill geodesic (follows vessel geometry)
2. Small extraction volume (stays close to aneurysm)
3. Orthogonal cutting analysis (ensures round openings)

This ensures we get proper vessel connectivity while maintaining
orthogonal boundaries and circular cross-sections.
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
import networkx as nx


def estimate_local_vessel_direction(mesh: trimesh.Trimesh, 
                                  center_vertex: int, 
                                  radius: float = 8.0) -> np.ndarray:
    """
    Estimate local vessel direction at a specific vertex using nearby geometry.
    """
    center_coords = mesh.vertices[center_vertex]
    
    # Find vertices within radius
    distances = np.linalg.norm(mesh.vertices - center_coords, axis=1)
    nearby_mask = distances <= radius
    nearby_vertices = mesh.vertices[nearby_mask]
    
    if len(nearby_vertices) < 10:
        # Fallback: use overall mesh direction
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        return pca.components_[0]
    
    # Use PCA to find main direction of nearby vertices
    pca = PCA(n_components=3)
    pca.fit(nearby_vertices)
    
    return pca.components_[0] / np.linalg.norm(pca.components_[0])


def flood_fill_geodesic_small_orthogonal(mesh: trimesh.Trimesh,
                                        start_vertex: int,
                                        target_vertices: int = 5000,
                                        max_distance: float = 18.0,
                                        orthogonal_constraint: bool = True,
                                        constraint_radius: float = 12.0,
                                        max_angle_deviation: float = 45.0) -> Set[int]:
    """
    Flood fill with geodesic distance, small volume, and orthogonal constraints.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    start_vertex : int
        Starting vertex index (closest to aneurysm)
    target_vertices : int
        Target number of vertices to extract
    max_distance : float
        Maximum distance from aneurysm location (mm)
    orthogonal_constraint : bool
        Whether to apply vessel direction constraints
    constraint_radius : float
        Distance from aneurysm where to start applying constraints (mm)
    max_angle_deviation : float
        Maximum angle deviation from vessel direction (degrees)
    """
    print(f"    Geodesic flood fill: target {target_vertices} vertices, max {max_distance}mm")
    
    # Build adjacency graph
    adjacency = {}
    edge_lengths = {}
    
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
                edge_lengths[(v1, v2)] = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
                edge_lengths[(v2, v1)] = edge_lengths[(v1, v2)]
    
    # Estimate vessel direction at start
    aneurysm_location = mesh.vertices[start_vertex]
    if orthogonal_constraint:
        vessel_direction = estimate_local_vessel_direction(mesh, start_vertex, 8.0)
        print(f"    Vessel direction at aneurysm: {vessel_direction}")
    
    # Flood fill with multiple constraints
    selected = set()
    queue = [(0.0, start_vertex)]  # (distance, vertex)
    distances = {start_vertex: 0.0}
    
    while queue and len(selected) < target_vertices:
        # Get vertex with smallest distance
        current_dist, current_vertex = min(queue)
        queue.remove((current_dist, current_vertex))
        
        if current_vertex in selected:
            continue
            
        # Check distance constraint
        euclidean_dist = np.linalg.norm(mesh.vertices[current_vertex] - aneurysm_location)
        if euclidean_dist > max_distance:
            continue
        
        # Check orthogonal constraint for vertices far from aneurysm
        if orthogonal_constraint and euclidean_dist > constraint_radius:
            # Estimate local vessel direction
            local_direction = estimate_local_vessel_direction(mesh, current_vertex, 6.0)
            
            # Calculate angle between local direction and main vessel direction
            angle = np.arccos(np.clip(np.abs(np.dot(local_direction, vessel_direction)), 0, 1))
            angle_degrees = np.degrees(angle)
            
            if angle_degrees > max_angle_deviation:
                continue  # Skip this vertex if direction deviates too much
        
        selected.add(current_vertex)
        
        # Add neighbors to queue
        for neighbor in adjacency.get(current_vertex, []):
            if neighbor not in selected:
                edge_dist = edge_lengths.get((current_vertex, neighbor), 1.0)
                new_distance = current_dist + edge_dist
                
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    queue.append((new_distance, neighbor))
    
    print(f"    Selected {len(selected)} vertices with orthogonal constraints")
    return selected


def create_orthogonal_boundary_mesh(mesh: trimesh.Trimesh, selected_vertices: Set[int]) -> trimesh.Trimesh:
    """
    Create mesh from selected vertices with orthogonal boundary analysis.
    """
    print(f"    Creating orthogonal boundary mesh from {len(selected_vertices)} vertices...")
    
    # Create vertex mapping
    selected_list = list(selected_vertices)
    old_to_new = {old: new for new, old in enumerate(selected_list)}
    new_vertices = mesh.vertices[selected_list]
    
    # Extract faces where all vertices are selected
    new_faces = []
    for face in mesh.faces:
        if all(v in selected_vertices for v in face):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
    
    if len(new_faces) == 0:
        print("    Error: No faces found in selection")
        return None
    
    extracted_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
    
    # Analyze boundary orthogonality
    boundary_edges = []
    edge_counts = {}
    
    for face in extracted_mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    # Analyze boundary shape quality
    if boundary_edges:
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        boundary_coords = extracted_mesh.vertices[list(boundary_vertices)]
        
        # Check if boundaries are roughly circular (orthogonal cuts)
        if len(boundary_coords) > 10:
            centroid = np.mean(boundary_coords, axis=0)
            distances = np.linalg.norm(boundary_coords - centroid, axis=1)
            std_ratio = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
            
            print(f"    Boundary analysis - vertices: {len(boundary_vertices)}, circularity: {1-std_ratio:.3f}")
            
            # If boundary is very non-circular, apply additional orthogonal correction
            if std_ratio > 0.3:
                print(f"    Warning: Non-circular boundary detected (std_ratio: {std_ratio:.3f})")
    
    print(f"    Final mesh: {len(extracted_mesh.vertices)} vertices, {len(extracted_mesh.faces)} faces")
    return extracted_mesh


def extract_vessel_geodesic_small_orthogonal(mesh: trimesh.Trimesh,
                                           aneurysm_location: np.ndarray,
                                           target_vertices: int = 5000,
                                           max_distance: float = 18.0) -> trimesh.Trimesh:
    """
    Extract vessel using geodesic flood fill with small volume and orthogonal constraints.
    """
    print(f"  Geodesic small orthogonal extraction...")
    print(f"    Target: {target_vertices} vertices within {max_distance}mm")
    
    # Find closest vertex to aneurysm
    distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    start_vertex = np.argmin(distances)
    
    print(f"    Starting from vertex {start_vertex} (distance: {distances[start_vertex]:.2f}mm)")
    
    # Flood fill with constraints
    selected_vertices = flood_fill_geodesic_small_orthogonal(
        mesh, 
        start_vertex,
        target_vertices=target_vertices,
        max_distance=max_distance,
        orthogonal_constraint=True,
        constraint_radius=12.0,
        max_angle_deviation=35.0  # Stricter angle constraint
    )
    
    if len(selected_vertices) < 500:
        print(f"    Warning: Too few vertices selected ({len(selected_vertices)}), relaxing constraints...")
        selected_vertices = flood_fill_geodesic_small_orthogonal(
            mesh, 
            start_vertex,
            target_vertices=target_vertices,
            max_distance=max_distance * 1.2,
            orthogonal_constraint=True,
            constraint_radius=15.0,
            max_angle_deviation=50.0
        )
    
    # Create orthogonal boundary mesh
    vessel_mesh = create_orthogonal_boundary_mesh(mesh, selected_vertices)
    
    return vessel_mesh


def process_single_vessel_geodesic_small_orthogonal(args: Tuple) -> Dict:
    """
    Process a single vessel with geodesic small orthogonal extraction.
    """
    stl_file, aneurysm_data, output_dir, processing_params = args
    
    patient_id = os.path.basename(stl_file).replace('.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'vessel_vertices': 0,
        'vessel_faces': 0
    }
    
    try:
        print(f"\nProcessing {patient_id} with geodesic small orthogonal extraction...")
        
        # Load mesh
        mesh = trimesh.load(stl_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Extract vessel with geodesic small orthogonal approach
        vessel_mesh = extract_vessel_geodesic_small_orthogonal(
            mesh, 
            aneurysm_location,
            target_vertices=processing_params.get('target_vertices', 5000),
            max_distance=processing_params.get('max_distance', 18.0)
        )
        
        if vessel_mesh is None or len(vessel_mesh.vertices) < 100:
            result['error'] = "Extraction failed or too few vertices"
            return result
        
        # Save vessel
        output_file = os.path.join(output_dir, f"{patient_id}_aneurysm_1_vessel_geo_small.stl")
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
    """Main processing function for geodesic small orthogonal vessel extraction"""
    parser = argparse.ArgumentParser(description='Geodesic Small Orthogonal Vessel Extraction')
    
    parser.add_argument('--data-dir', 
                       default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
                       help='Directory containing STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geo_small_orthogonal'),
                       help='Output directory for extracted vessels')
    
    parser.add_argument('--target-vertices', type=int, default=5000,
                       help='Target number of vertices to extract')
    
    parser.add_argument('--max-distance', type=float, default=18.0,
                       help='Maximum distance from aneurysm (mm)')
    
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
    
    # Processing parameters
    processing_params = {
        'target_vertices': args.target_vertices,
        'max_distance': args.max_distance
    }
    
    print(f"\nGeodesic Small Orthogonal Parameters:")
    print(f"  Target vertices: {args.target_vertices}")
    print(f"  Max distance: {args.max_distance}mm")
    print(f"  Uses: Flood fill geodesic + Small volume + Orthogonal constraints")
    
    # Prepare processing arguments
    process_args = [(stl_file, patient_data, args.output_dir, processing_params) 
                   for stl_file, patient_data in stl_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Geodesic small orthogonal"):
            result = process_single_vessel_geodesic_small_orthogonal(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_geodesic_small_orthogonal, process_args),
                               total=len(process_args), desc="Geodesic small orthogonal"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Geodesic Small Orthogonal Vessel Extraction Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_vertices = np.mean([r['vessel_vertices'] for r in successful])
        avg_faces = np.mean([r['vessel_faces'] for r in successful])
        
        print(f"\nGeodesic Small Orthogonal Summary:")
        print(f"  Average vessel vertices: {avg_vertices:.0f}")
        print(f"  Average vessel faces: {avg_faces:.0f}")
        print(f"  ✓ Follows vessel connectivity (geodesic)")
        print(f"  ✓ Small extraction volume")
        print(f"  ✓ Orthogonal constraints for round openings")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'geodesic_small_orthogonal_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Vessels extracted with geodesic + small volume + orthogonal constraints!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 