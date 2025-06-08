#!/usr/bin/env python3
"""
Random Walk Aneurysm Vessel Extraction

This script performs random walk analysis starting from aneurysm locations
to extract connected vessel regions from STL meshes.
"""

import json
import os
import numpy as np
import trimesh
from pathlib import Path
import argparse
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import time


def build_mesh_graph(mesh: trimesh.Trimesh, k_neighbors: int = 6) -> nx.Graph:
    """
    Build a graph representation of the mesh for random walk
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    k_neighbors : int
        Number of neighbors to connect each vertex to
    
    Returns:
    --------
    nx.Graph : Graph where nodes are vertex indices and edges connect nearby vertices
    """
    print(f"Building mesh graph with {len(mesh.vertices)} vertices...")
    
    # Build k-NN graph based on vertex positions
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(mesh.vertices)
    distances, indices = nbrs.kneighbors(mesh.vertices)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(mesh.vertices)))
    
    # Add edges to k-nearest neighbors
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors[1:]):  # Skip self (first neighbor)
            weight = 1.0 / (distances[i][j + 1] + 1e-8)  # Inverse distance weight
            G.add_edge(i, neighbor, weight=weight)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def random_walk_region_growing(graph: nx.Graph, 
                              start_vertex: int,
                              max_steps: int = 5000,
                              step_probability: float = 0.8,
                              distance_threshold: float = 50.0,
                              mesh_vertices: np.ndarray = None) -> Set[int]:
    """
    Perform random walk region growing from start vertex
    
    Parameters:
    -----------
    graph : nx.Graph
        Mesh graph
    start_vertex : int
        Starting vertex index
    max_steps : int
        Maximum number of random walk steps
    step_probability : float
        Probability of continuing walk vs stopping
    distance_threshold : float
        Maximum distance from start to include vertices
    mesh_vertices : np.ndarray
        Vertex coordinates for distance checking
    
    Returns:
    --------
    Set[int] : Set of vertex indices in the extracted region
    """
    if start_vertex not in graph:
        return set()
    
    visited = set()
    region = set([start_vertex])
    start_pos = mesh_vertices[start_vertex] if mesh_vertices is not None else None
    
    # Multiple random walks from the start vertex
    num_walks = 20
    for walk_id in range(num_walks):
        current = start_vertex
        
        for step in range(max_steps // num_walks):
            # Add current vertex to region
            region.add(current)
            
            # Check distance constraint
            if mesh_vertices is not None and start_pos is not None:
                current_pos = mesh_vertices[current]
                distance = np.linalg.norm(current_pos - start_pos)
                if distance > distance_threshold:
                    break
            
            # Get neighbors
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break
            
            # Probability of stopping vs continuing
            if np.random.random() > step_probability:
                break
            
            # Choose next vertex (weighted by edge weights)
            weights = [graph[current][neighbor].get('weight', 1.0) for neighbor in neighbors]
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            next_vertex = np.random.choice(neighbors, p=weights)
            current = next_vertex
    
    return region


def extract_connected_component(mesh: trimesh.Trimesh,
                               vertex_indices: Set[int],
                               expansion_radius: float = 10.0) -> trimesh.Trimesh:
    """
    Extract mesh component from vertex indices and expand slightly
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original mesh
    vertex_indices : Set[int]
        Indices of vertices to extract
    expansion_radius : float
        Radius for expanding the region
    
    Returns:
    --------
    trimesh.Trimesh : Extracted mesh component
    """
    if not vertex_indices:
        return trimesh.Trimesh()
    
    # Convert to list and get vertices
    vertex_list = list(vertex_indices)
    selected_vertices = mesh.vertices[vertex_list]
    
    # Expand region by including nearby vertices
    if expansion_radius > 0:
        distances = cdist(mesh.vertices, selected_vertices)
        min_distances = distances.min(axis=1)
        expanded_indices = np.where(min_distances <= expansion_radius)[0]
        vertex_list = list(set(vertex_list) | set(expanded_indices))
    
    # Find faces that have all vertices in the selected set
    vertex_set = set(vertex_list)
    face_mask = np.all(np.isin(mesh.faces, vertex_list), axis=1)
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


def process_single_aneurysm(args: Tuple) -> Dict:
    """
    Process a single aneurysm for vessel extraction
    
    Parameters:
    -----------
    args : tuple
        (patient_id, aneurysm_data, stl_path, output_dir, extraction_params)
    
    Returns:
    --------
    Dict : Processing results
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
        
        # Build mesh graph
        graph = build_mesh_graph(mesh, k_neighbors=extraction_params.get('k_neighbors', 6))
        
        # Process each aneurysm
        for aneurysm in aneurysm_data['aneurysms']:
            aneurysm_id = aneurysm['aneurysm_id']
            start_vertex = aneurysm['mesh_vertex_index']
            
            print(f"  Extracting {aneurysm_id} from vertex {start_vertex}...")
            
            # Perform random walk region growing
            region_vertices = random_walk_region_growing(
                graph=graph,
                start_vertex=start_vertex,
                max_steps=extraction_params.get('max_steps', 5000),
                step_probability=extraction_params.get('step_probability', 0.8),
                distance_threshold=extraction_params.get('distance_threshold', 50.0),
                mesh_vertices=mesh.vertices
            )
            
            if not region_vertices:
                print(f"    Warning: No region found for {aneurysm_id}")
                continue
            
            # Extract connected component
            extracted_mesh = extract_connected_component(
                mesh=mesh,
                vertex_indices=region_vertices,
                expansion_radius=extraction_params.get('expansion_radius', 10.0)
            )
            
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
                'region_size': len(region_vertices),
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
        description='Extract aneurysm-connected vessel regions using random walk'
    )
    parser.add_argument(
        '--aneurysm-json',
        default='../all_patients_aneurysms_for_stl.json',
        help='JSON file with STL aneurysm coordinates'
    )
    parser.add_argument(
        '--stl-dir',
        default=os.path.expanduser('~/urp/data/uan/original'),
        help='Directory containing STL files'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels'),
        help='Output directory for extracted vessel meshes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=5000,
        help='Maximum random walk steps (default: 5000)'
    )
    parser.add_argument(
        '--step-probability',
        type=float,
        default=0.8,
        help='Probability of continuing random walk (default: 0.8)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=50.0,
        help='Maximum distance from aneurysm center (default: 50.0)'
    )
    parser.add_argument(
        '--expansion-radius',
        type=float,
        default=10.0,
        help='Radius for expanding extracted region (default: 10.0)'
    )
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=6,
        help='Number of neighbors in mesh graph (default: 6)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, recommend 1 for memory)'
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
        'max_steps': args.max_steps,
        'step_probability': args.step_probability,
        'distance_threshold': args.distance_threshold,
        'expansion_radius': args.expansion_radius,
        'k_neighbors': args.k_neighbors
    }
    
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
    
    print(f"Processing {len(process_args)} patients")
    
    # Process patients
    start_time = time.time()
    if args.workers == 1:
        # Sequential processing (recommended for memory efficiency)
        results = []
        for process_arg in tqdm(process_args, desc="Extracting vessels"):
            result = process_single_aneurysm(process_arg)
            results.append(result)
    else:
        # Parallel processing (may use too much memory)
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_aneurysm, process_args), 
                               total=len(process_args), desc="Extracting vessels"))
    
    # Summary
    total_time = time.time() - start_time
    successful_patients = sum(1 for r in results if r['success'])
    total_aneurysms = sum(len(r['extracted_aneurysms']) for r in results if r['success'])
    
    print(f"\n" + "="*60)
    print(f"Vessel extraction complete in {total_time:.1f} seconds")
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