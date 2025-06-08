#!/usr/bin/env python3
"""
Geodesic Region Growing Aneurysm Vessel Extraction

This script uses geodesic distance-based region growing to extract connected 
vessel regions from aneurysm points. This approach ensures proper connectivity
without islands or disconnected components.
"""

import json
import os
import numpy as np
import trimesh
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import time


def build_mesh_connectivity_graph(mesh: trimesh.Trimesh) -> nx.Graph:
    """
    Build a connectivity graph from mesh faces and vertices.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    
    Returns:
    --------
    nx.Graph : Graph where nodes are vertices and edges connect adjacent vertices
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(mesh.vertices)))
    
    # Add edges from face connectivity
    for face in mesh.faces:
        # Each face creates edges between all pairs of vertices
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = face[i], face[j]
                if not G.has_edge(v1, v2):
                    # Weight is Euclidean distance between vertices
                    distance = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
                    G.add_edge(v1, v2, weight=distance)
    
    return G


def geodesic_region_growing(mesh: trimesh.Trimesh,
                           start_vertex: int,
                           max_geodesic_distance: float = 40.0,
                           max_vertices: int = 10000) -> Set[int]:
    """
    Extract vessel region using geodesic distance-based region growing.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    start_vertex : int
        Starting vertex index
    max_geodesic_distance : float
        Maximum geodesic distance from start vertex
    max_vertices : int
        Maximum number of vertices to include
    
    Returns:
    --------
    Set[int] : Set of vertex indices in the extracted region
    """
    print(f"    Building connectivity graph...")
    
    # Build adjacency matrix for geodesic distance computation
    n_vertices = len(mesh.vertices)
    rows, cols, data = [], [], []
    
    # Add edges from mesh faces
    for face in mesh.faces:
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = face[i], face[j]
                distance = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
                
                rows.extend([v1, v2])
                cols.extend([v2, v1])
                data.extend([distance, distance])
    
    # Create sparse adjacency matrix
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    print(f"    Computing geodesic distances from vertex {start_vertex}...")
    
    # Compute geodesic distances using Dijkstra's algorithm
    geodesic_distances = dijkstra(adjacency_matrix, indices=start_vertex, directed=False)
    
    # Find vertices within geodesic distance threshold
    valid_distances = geodesic_distances[geodesic_distances != np.inf]
    within_threshold = geodesic_distances <= max_geodesic_distance
    
    # Get vertex indices within threshold
    region_vertices = set(np.where(within_threshold)[0])
    
    # Limit number of vertices if necessary
    if len(region_vertices) > max_vertices:
        # Sort by geodesic distance and take closest vertices
        distances_with_indices = [(geodesic_distances[i], i) for i in region_vertices]
        distances_with_indices.sort()
        region_vertices = set([idx for _, idx in distances_with_indices[:max_vertices]])
    
    print(f"    Found {len(region_vertices)} vertices within geodesic distance {max_geodesic_distance}")
    
    return region_vertices


def flood_fill_region_growing(mesh: trimesh.Trimesh,
                             start_vertex: int,
                             max_euclidean_distance: float = 40.0,
                             max_vertices: int = 10000) -> Set[int]:
    """
    Extract vessel region using flood fill with connectivity constraints.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    start_vertex : int
        Starting vertex index
    max_euclidean_distance : float
        Maximum Euclidean distance from start vertex
    max_vertices : int
        Maximum number of vertices to include
    
    Returns:
    --------
    Set[int] : Set of vertex indices in the extracted region
    """
    print(f"    Performing flood fill region growing...")
    
    # Build vertex adjacency from faces
    vertex_adjacency = {i: set() for i in range(len(mesh.vertices))}
    for face in mesh.faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vertex_adjacency[face[i]].add(face[j])
    
    # Flood fill starting from start_vertex
    start_pos = mesh.vertices[start_vertex]
    visited = set()
    queue = [start_vertex]
    region_vertices = set()
    
    while queue and len(region_vertices) < max_vertices:
        current_vertex = queue.pop(0)
        
        if current_vertex in visited:
            continue
            
        visited.add(current_vertex)
        
        # Check distance constraint
        current_pos = mesh.vertices[current_vertex]
        distance = np.linalg.norm(current_pos - start_pos)
        
        if distance <= max_euclidean_distance:
            region_vertices.add(current_vertex)
            
            # Add adjacent vertices to queue
            for neighbor in vertex_adjacency[current_vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    print(f"    Flood fill found {len(region_vertices)} vertices")
    
    return region_vertices


def ensure_connected_component(mesh: trimesh.Trimesh, 
                              vertex_indices: Set[int], 
                              start_vertex: int) -> Set[int]:
    """
    Ensure the extracted region is a single connected component containing start_vertex.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    vertex_indices : Set[int]
        Set of vertex indices
    start_vertex : int
        Starting vertex that must be included
    
    Returns:
    --------
    Set[int] : Largest connected component containing start_vertex
    """
    if len(vertex_indices) <= 1:
        return vertex_indices
    
    # Build adjacency for the subgraph
    vertex_list = list(vertex_indices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    # Create adjacency matrix for subgraph
    n = len(vertex_list)
    adjacency = np.zeros((n, n), dtype=bool)
    
    for face in mesh.faces:
        face_vertices_in_region = [v for v in face if v in vertex_indices]
        if len(face_vertices_in_region) >= 2:
            # Add edges between vertices in this face
            for i in range(len(face_vertices_in_region)):
                for j in range(i + 1, len(face_vertices_in_region)):
                    v1, v2 = face_vertices_in_region[i], face_vertices_in_region[j]
                    idx1, idx2 = vertex_to_idx[v1], vertex_to_idx[v2]
                    adjacency[idx1, idx2] = True
                    adjacency[idx2, idx1] = True
    
    # Find connected components using iterative DFS
    visited = set()
    components = []
    
    def iterative_dfs(start_node):
        component = []
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
                
            visited.add(node)
            component.append(node)
            
            # Add unvisited neighbors to stack
            for neighbor in range(n):
                if adjacency[node, neighbor] and neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    for i in range(n):
        if i not in visited:
            component = iterative_dfs(i)
            if component:
                components.append(component)
    
    # Find component containing start_vertex
    if start_vertex not in vertex_indices:
        # Return largest component
        largest_component = max(components, key=len) if components else []
    else:
        start_idx = vertex_to_idx[start_vertex]
        start_component = None
        for component in components:
            if start_idx in component:
                start_component = component
                break
        
        if start_component is None:
            largest_component = max(components, key=len) if components else []
        else:
            largest_component = start_component
    
    # Convert back to vertex indices
    connected_vertices = set(vertex_list[i] for i in largest_component)
    
    print(f"    Ensured connectivity: {len(vertex_indices)} -> {len(connected_vertices)} vertices")
    
    return connected_vertices


def extract_mesh_from_vertices(mesh: trimesh.Trimesh, 
                              vertex_indices: Set[int]) -> trimesh.Trimesh:
    """
    Extract mesh from vertex indices.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original mesh
    vertex_indices : Set[int]
        Vertex indices to extract
    
    Returns:
    --------
    trimesh.Trimesh : Extracted mesh
    """
    if not vertex_indices:
        return trimesh.Trimesh()
    
    # Find faces that have all vertices in the selected set
    vertex_set = set(vertex_indices)
    face_mask = np.all(np.isin(mesh.faces, list(vertex_indices)), axis=1)
    selected_faces = mesh.faces[face_mask]
    
    if len(selected_faces) == 0:
        return trimesh.Trimesh()
    
    # Create mapping from old to new vertex indices
    unique_vertices = np.unique(selected_faces)
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    
    # Extract vertices and remap faces
    extracted_vertices = mesh.vertices[unique_vertices]
    remapped_faces = np.array([[vertex_map[face[i]] for i in range(3)] for face in selected_faces])
    
    # Create new mesh
    extracted_mesh = trimesh.Trimesh(vertices=extracted_vertices, faces=remapped_faces)
    
    # Clean up the mesh
    extracted_mesh.remove_duplicate_faces()
    extracted_mesh.remove_unreferenced_vertices()
    
    return extracted_mesh


def process_single_aneurysm_geodesic(args: Tuple) -> Dict:
    """
    Process a single aneurysm for geodesic vessel extraction
    """
    patient_id, aneurysm_data, stl_path, output_dir, extraction_params = args
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'extracted_aneurysms': []
    }
    
    try:
        print(f"Processing {patient_id}...")
        
        # Load mesh
        mesh = trimesh.load(stl_path)
        
        # Process each aneurysm
        for aneurysm in aneurysm_data['aneurysms']:
            aneurysm_id = aneurysm['aneurysm_id']
            start_vertex = aneurysm['mesh_vertex_index']
            
            print(f"  Extracting {aneurysm_id} from vertex {start_vertex}...")
            
            # Choose extraction method
            method = extraction_params.get('method', 'geodesic')
            
            if method == 'geodesic':
                # Method 1: Geodesic distance-based region growing
                region_vertices = geodesic_region_growing(
                    mesh=mesh,
                    start_vertex=start_vertex,
                    max_geodesic_distance=extraction_params.get('max_distance', 40.0),
                    max_vertices=extraction_params.get('max_vertices', 10000)
                )
            elif method == 'flood_fill':
                # Method 2: Flood fill region growing
                region_vertices = flood_fill_region_growing(
                    mesh=mesh,
                    start_vertex=start_vertex,
                    max_euclidean_distance=extraction_params.get('max_distance', 40.0),
                    max_vertices=extraction_params.get('max_vertices', 10000)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if not region_vertices:
                print(f"    Warning: No region found for {aneurysm_id}")
                continue
            
            # Ensure connectivity
            connected_vertices = ensure_connected_component(mesh, region_vertices, start_vertex)
            
            if not connected_vertices:
                print(f"    Warning: No connected region found for {aneurysm_id}")
                continue
            
            # Extract mesh
            extracted_mesh = extract_mesh_from_vertices(mesh, connected_vertices)
            
            if len(extracted_mesh.vertices) == 0:
                print(f"    Warning: Empty mesh extracted for {aneurysm_id}")
                continue
            
            # Save extracted mesh
            output_filename = f"{patient_id}_{aneurysm_id}_vessel.stl"
            output_path = os.path.join(output_dir, output_filename)
            os.makedirs(output_dir, exist_ok=True)
            extracted_mesh.export(output_path)
            
            aneurysm_result = {
                'aneurysm_id': aneurysm_id,
                'description': aneurysm['description'],
                'start_vertex': start_vertex,
                'method': method,
                'region_vertices': len(region_vertices),
                'connected_vertices': len(connected_vertices),
                'extracted_vertices': len(extracted_mesh.vertices),
                'extracted_faces': len(extracted_mesh.faces),
                'output_file': output_path
            }
            
            result['extracted_aneurysms'].append(aneurysm_result)
            print(f"    ✓ Extracted {len(extracted_mesh.vertices)} vertices, {len(extracted_mesh.faces)} faces")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ERROR: {e}")
    
    return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Extract aneurysm-connected vessel regions using geodesic methods'
    )
    parser.add_argument(
        '--aneurysm-json',
        default='../all_patients_aneurysms_for_stl.json',
        help='JSON file with STL aneurysm coordinates'
    )
    parser.add_argument(
        '--stl-dir',
        default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
        help='Directory containing STL files'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geodesic'),
        help='Output directory for extracted vessel meshes'
    )
    parser.add_argument(
        '--method',
        choices=['geodesic', 'flood_fill'],
        default='geodesic',
        help='Extraction method (default: geodesic)'
    )
    parser.add_argument(
        '--max-distance',
        type=float,
        default=40.0,
        help='Maximum distance from aneurysm center (default: 40.0)'
    )
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=10000,
        help='Maximum number of vertices to extract (default: 10000)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=32,
        help='Number of parallel workers (default: 32)'
    )
    parser.add_argument(
        '--patient-limit',
        type=int,
        help='Limit number of patients to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load aneurysm coordinates
    print(f"Loading aneurysm coordinates from: {args.aneurysm_json}")
    with open(args.aneurysm_json, 'r') as f:
        aneurysm_data = json.load(f)
    
    print(f"Found {len(aneurysm_data)} patients")
    
    # Prepare extraction parameters
    extraction_params = {
        'method': args.method,
        'max_distance': args.max_distance,
        'max_vertices': args.max_vertices
    }
    
    print(f"\nGeodesic Extraction Parameters:")
    print(f"  - Method: {args.method}")
    print(f"  - Max distance: {args.max_distance}")
    print(f"  - Max vertices: {args.max_vertices}")
    
    # Prepare processing arguments
    process_args = []
    patient_count = 0
    
    for patient_id, patient_data in aneurysm_data.items():
        if args.patient_limit and patient_count >= args.patient_limit:
            break
        
        stl_path = os.path.join(args.stl_dir, f"{patient_id}.stl")
        if not os.path.exists(stl_path):
            print(f"Warning: STL file not found for {patient_id}")
            continue
        
        process_args.append((patient_id, patient_data, stl_path, args.output_dir, extraction_params))
        patient_count += 1
    
    print(f"\nProcessing {len(process_args)} patients with {args.method} method")
    
    # Process patients
    start_time = time.time()
    if args.workers == 1:
        # Sequential processing
        results = []
        for process_arg in tqdm(process_args, desc="Extracting vessels"):
            result = process_single_aneurysm_geodesic(process_arg)
            results.append(result)
    else:
        # Parallel processing
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_aneurysm_geodesic, process_args), 
                               total=len(process_args), desc="Extracting vessels"))
    
    # Summary
    total_time = time.time() - start_time
    successful_patients = sum(1 for r in results if r['success'])
    total_aneurysms = sum(len(r['extracted_aneurysms']) for r in results if r['success'])
    
    print(f"\n" + "="*60)
    print(f"Geodesic vessel extraction complete in {total_time:.1f} seconds")
    print(f"Successful patients: {successful_patients}/{len(results)}")
    print(f"Total aneurysms extracted: {total_aneurysms}")
    print(f"Output directory: {args.output_dir}")
    
    # Save results summary
    results_file = os.path.join(args.output_dir, 'extraction_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 