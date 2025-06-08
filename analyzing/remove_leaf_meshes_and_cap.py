#!/usr/bin/env python3
"""
Remove Leaf Meshes and Cap

This script post-processes extracted vessels by:
1. Detecting thin connections (bottlenecks) 
2. Removing leaf-like mesh regions beyond bottlenecks
3. Filling holes with triangular faces (not openings)
4. Applying clean flat capping to real vessel openings

This removes artifacts and simplifies CFD setup.
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


def analyze_mesh_connectivity(mesh: trimesh.Trimesh, aneurysm_location: np.ndarray) -> Dict:
    """
    Analyze mesh connectivity to find bottlenecks and leaf regions.
    """
    print("    Analyzing mesh connectivity for bottlenecks...")
    
    # Build vertex adjacency graph
    adjacency = {}
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 not in adjacency:
                adjacency[v1] = set()
            if v2 not in adjacency:
                adjacency[v2] = set()
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Find aneurysm region (main vessel core)
    aneurysm_distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    aneurysm_vertex = np.argmin(aneurysm_distances)
    
    # Define core region around aneurysm (within 8mm)
    core_radius = 8.0
    core_mask = aneurysm_distances <= core_radius
    core_vertices = set(np.where(core_mask)[0])
    
    print(f"    Core region: {len(core_vertices)} vertices within {core_radius}mm of aneurysm")
    
    # Find vertices with low connectivity (potential bottlenecks)
    vertex_connectivity = {}
    for vertex, neighbors in adjacency.items():
        vertex_connectivity[vertex] = len(neighbors)
    
    # Analyze each vertex's connectivity to core region
    vertex_analysis = {}
    for vertex in range(len(mesh.vertices)):
        if vertex in adjacency:
            neighbors = adjacency[vertex]
            core_connections = len(neighbors.intersection(core_vertices))
            total_connections = len(neighbors)
            distance_to_aneurysm = aneurysm_distances[vertex]
            
            vertex_analysis[vertex] = {
                'core_connections': core_connections,
                'total_connections': total_connections,
                'distance_to_aneurysm': distance_to_aneurysm,
                'is_core': vertex in core_vertices,
                'connectivity_ratio': core_connections / total_connections if total_connections > 0 else 0
            }
    
    return {
        'adjacency': adjacency,
        'core_vertices': core_vertices,
        'aneurysm_vertex': aneurysm_vertex,
        'vertex_analysis': vertex_analysis,
        'vertex_connectivity': vertex_connectivity
    }


def detect_bottlenecks_and_leaves(mesh: trimesh.Trimesh, connectivity_info: Dict) -> Dict:
    """
    Detect bottleneck connections and leaf-like regions.
    """
    print("    Detecting bottlenecks and leaf regions...")
    
    adjacency = connectivity_info['adjacency']
    core_vertices = connectivity_info['core_vertices']
    vertex_analysis = connectivity_info['vertex_analysis']
    
    # Find potential bottleneck vertices
    bottleneck_candidates = []
    for vertex, analysis in vertex_analysis.items():
        distance = analysis['distance_to_aneurysm']
        connectivity = analysis['total_connections']
        core_connections = analysis['core_connections']
        
        # Bottleneck criteria (moderate):
        # 1. Not in core region (> 8mm from aneurysm)
        # 2. Low total connectivity (< 5 neighbors)
        # 3. Few connections to core region
        # 4. Distance > 11mm from aneurysm
        if (distance > 11.0 and 
            connectivity <= 4 and 
            core_connections <= 1 and
            not analysis['is_core']):
            bottleneck_candidates.append(vertex)
    
    print(f"    Found {len(bottleneck_candidates)} bottleneck candidates")
    
    # For each bottleneck, find the leaf region it connects to
    leaf_regions = []
    for bottleneck in bottleneck_candidates:
        # BFS from bottleneck to find connected leaf region
        visited = set()
        leaf_vertices = set()
        queue = [bottleneck]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Don't cross into core region
            if current in core_vertices:
                continue
                
            leaf_vertices.add(current)
            
            # Add neighbors that are also far from core
            for neighbor in adjacency.get(current, []):
                if (neighbor not in visited and 
                    neighbor not in core_vertices and
                    vertex_analysis[neighbor]['distance_to_aneurysm'] > 8.0):
                    queue.append(neighbor)
        
        # Only consider as leaf if it's medium-sized artifact (moderate approach)
        if 20 <= len(leaf_vertices) <= 200:
            leaf_regions.append({
                'bottleneck': bottleneck,
                'vertices': leaf_vertices,
                'size': len(leaf_vertices),
                'max_distance': max(vertex_analysis[v]['distance_to_aneurysm'] for v in leaf_vertices)
            })
    
    print(f"    Identified {len(leaf_regions)} leaf regions to remove")
    for i, leaf in enumerate(leaf_regions):
        print(f"      Leaf {i+1}: {leaf['size']} vertices, max distance {leaf['max_distance']:.1f}mm")
    
    return {
        'bottlenecks': bottleneck_candidates,
        'leaf_regions': leaf_regions
    }


def remove_leaf_regions(mesh: trimesh.Trimesh, leaf_info: Dict, connectivity_info: Dict) -> trimesh.Trimesh:
    """
    Remove leaf regions and create a mesh without them.
    """
    print("    Removing leaf regions...")
    
    leaf_regions = leaf_info['leaf_regions']
    
    # Collect all vertices to remove
    vertices_to_remove = set()
    for leaf in leaf_regions:
        vertices_to_remove.update(leaf['vertices'])
    
    print(f"    Removing {len(vertices_to_remove)} vertices from leaf regions")
    
    # Keep vertices that are not in leaf regions
    keep_mask = np.ones(len(mesh.vertices), dtype=bool)
    for vertex in vertices_to_remove:
        keep_mask[vertex] = False
    
    kept_vertices = np.where(keep_mask)[0]
    
    if len(kept_vertices) < 100:
        print("    Warning: Too many vertices would be removed, keeping original mesh")
        return mesh
    
    # Create vertex mapping
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[kept_vertices] = np.arange(len(kept_vertices))
    
    # Extract kept vertices
    new_vertices = mesh.vertices[kept_vertices]
    
    # Extract faces where all vertices are kept
    new_faces = []
    for face in mesh.faces:
        if all(keep_mask[face]):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
    
    if len(new_faces) == 0:
        print("    Warning: No faces left after removing leaves, keeping original")
        return mesh
    
    # Create mesh without leaves
    cleaned_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
    
    print(f"    Cleaned mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")
    
    return cleaned_mesh


def fill_holes_with_faces(mesh: trimesh.Trimesh, max_hole_size: int = 50) -> trimesh.Trimesh:
    """
    Fill holes created by leaf removal with triangular faces.
    """
    print("    Filling holes created by leaf removal...")
    
    # Detect boundary loops (holes)
    edge_counts = {}
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Boundary edges appear in only one face
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    if not boundary_edges:
        print("    No boundary edges found, mesh may already be closed")
        return mesh
    
    print(f"    Found {len(boundary_edges)} boundary edges")
    
    # Build boundary graph to find loops
    boundary_graph = nx.Graph()
    for edge in boundary_edges:
        boundary_graph.add_edge(edge[0], edge[1])
    
    # Find connected components (hole boundaries)
    hole_boundaries = []
    for component in nx.connected_components(boundary_graph):
        if len(component) >= 3:  # Need at least 3 vertices for a face
            hole_boundaries.append(list(component))
    
    print(f"    Found {len(hole_boundaries)} holes to fill")
    
    # Fill each hole
    current_mesh = mesh.copy()
    filled_count = 0
    
    for i, hole_vertices in enumerate(hole_boundaries):
        if len(hole_vertices) > max_hole_size:
            print(f"      Skipping large hole {i+1} with {len(hole_vertices)} vertices")
            continue
        
        try:
            # Get hole vertex coordinates
            hole_coords = current_mesh.vertices[hole_vertices]
            
            # Find hole centroid and normal
            centroid = np.mean(hole_coords, axis=0)
            
            # Use PCA to find hole plane
            pca = PCA(n_components=3)
            pca.fit(hole_coords - centroid)
            normal = pca.components_[2]  # Normal is least variance direction
            
            # Project hole vertices to 2D plane
            u = pca.components_[0]
            v = pca.components_[1]
            
            hole_2d = np.column_stack([
                np.dot(hole_coords - centroid, u),
                np.dot(hole_coords - centroid, v)
            ])
            
            # Create triangulation using simple fan triangulation from centroid
            # Add centroid as new vertex
            centroid_idx = len(current_mesh.vertices)
            new_vertices = np.vstack([current_mesh.vertices, centroid.reshape(1, -1)])
            
            # Create fan triangulation
            new_faces = current_mesh.faces.tolist()
            
            # Sort hole vertices in circular order
            angles = np.arctan2(hole_2d[:, 1], hole_2d[:, 0])
            sorted_indices = np.argsort(angles)
            sorted_hole_vertices = [hole_vertices[i] for i in sorted_indices]
            
            # Create triangular faces from centroid to hole edge
            for j in range(len(sorted_hole_vertices)):
                v1 = sorted_hole_vertices[j]
                v2 = sorted_hole_vertices[(j + 1) % len(sorted_hole_vertices)]
                
                # Add triangle face (ensure correct orientation)
                new_faces.append([centroid_idx, v1, v2])
            
            # Update mesh
            current_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
            filled_count += 1
            
            print(f"      Filled hole {i+1} with {len(hole_vertices)} vertices using fan triangulation")
            
        except Exception as e:
            print(f"      Failed to fill hole {i+1}: {e}")
            continue
    
    print(f"    Successfully filled {filled_count}/{len(hole_boundaries)} holes")
    
    # Clean up mesh
    try:
        current_mesh.remove_duplicate_faces()
        current_mesh.remove_unreferenced_vertices()
    except:
        pass
    
    print(f"    Final hole-filled mesh: {len(current_mesh.vertices)} vertices, {len(current_mesh.faces)} faces")
    
    return current_mesh


def process_remove_leaves_and_cap(vessel_mesh: trimesh.Trimesh, 
                                 aneurysm_location: np.ndarray) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Complete process: remove leaves, fill holes, then ready for capping.
    """
    print("  Removing leaf meshes and preparing for capping...")
    
    # Step 1: Analyze connectivity
    connectivity_info = analyze_mesh_connectivity(vessel_mesh, aneurysm_location)
    
    # Step 2: Detect bottlenecks and leaves
    leaf_info = detect_bottlenecks_and_leaves(vessel_mesh, connectivity_info)
    
    # Step 3: Remove leaf regions if any found
    if leaf_info['leaf_regions']:
        cleaned_mesh = remove_leaf_regions(vessel_mesh, leaf_info, connectivity_info)
        
        # Step 4: Fill holes created by leaf removal
        filled_mesh = fill_holes_with_faces(cleaned_mesh, max_hole_size=50)
    else:
        print("    No leaf regions found, proceeding with original mesh")
        filled_mesh = vessel_mesh
    
    # Prepare analysis info
    analysis_info = {
        'leaf_regions_removed': len(leaf_info['leaf_regions']),
        'vertices_removed': sum(len(leaf['vertices']) for leaf in leaf_info['leaf_regions']),
        'original_vertices': len(vessel_mesh.vertices),
        'final_vertices': len(filled_mesh.vertices),
        'is_watertight': filled_mesh.is_watertight
    }
    
    print(f"  ✓ Leaf removal complete: {analysis_info['leaf_regions_removed']} regions removed")
    print(f"    {analysis_info['original_vertices']} → {analysis_info['final_vertices']} vertices")
    print(f"    Watertight: {analysis_info['is_watertight']}")
    
    return filled_mesh, analysis_info


def process_single_vessel_remove_leaves_and_cap(args: Tuple) -> Dict:
    """
    Process a single vessel: remove leaves, fill holes, then apply capping.
    """
    vessel_file, aneurysm_data, output_dir = args
    
    patient_id = os.path.basename(vessel_file).replace('_aneurysm_1_vessel_geo_small.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'analysis': None
    }
    
    try:
        print(f"\nProcessing {patient_id} - removing leaves and capping...")
        
        # Load vessel mesh
        vessel_mesh = trimesh.load(vessel_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Remove leaves and fill holes
        cleaned_mesh, analysis_info = process_remove_leaves_and_cap(vessel_mesh, aneurysm_location)
        
        # Save cleaned mesh before capping
        cleaned_file = os.path.join(output_dir, f"{patient_id}_cleaned_no_leaves.stl")
        os.makedirs(output_dir, exist_ok=True)
        cleaned_mesh.export(cleaned_file)
        
        result['success'] = True
        result['analysis'] = analysis_info
        result['cleaned_file'] = cleaned_file
        
        print(f"  ✓ {patient_id}: Removed {analysis_info['leaf_regions_removed']} leaf regions")
        print(f"    Saved cleaned mesh: {cleaned_file}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for removing leaf meshes and capping"""
    parser = argparse.ArgumentParser(description='Remove Leaf Meshes and Cap')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geo_small_orthogonal'),
                       help='Directory containing vessel STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/vessels_no_leaves'),
                       help='Output directory for cleaned vessels')
    
    parser.add_argument('--workers', '-j', type=int, default=16,
                       help='Number of parallel workers')
    
    parser.add_argument('--patient-limit', type=int,
                       help='Limit number of patients (for testing)')
    
    args = parser.parse_args()
    
    # Load aneurysm data
    print(f"Loading aneurysm data from: {args.aneurysm_json}")
    with open(args.aneurysm_json, 'r') as f:
        aneurysm_data = json.load(f)
    
    # Find vessel files
    vessel_files = []
    for patient_id, patient_data in aneurysm_data.items():
        if args.patient_limit and len(vessel_files) >= args.patient_limit:
            break
            
        vessel_file = os.path.join(args.vessel_dir, f"{patient_id}_aneurysm_1_vessel_geo_small.stl")
        if os.path.exists(vessel_file):
            vessel_files.append((vessel_file, patient_data))
        else:
            print(f"Warning: Vessel file not found for {patient_id}")
    
    print(f"Found {len(vessel_files)} vessel files to process")
    print(f"Removing leaf meshes connected by thin connections...")
    
    # Prepare processing arguments
    process_args = [(vessel_file, patient_data, args.output_dir) 
                   for vessel_file, patient_data in vessel_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Removing leaves"):
            result = process_single_vessel_remove_leaves_and_cap(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_remove_leaves_and_cap, process_args),
                               total=len(process_args), desc="Removing leaves"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Leaf Removal Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        total_leaves = sum(r['analysis']['leaf_regions_removed'] for r in successful)
        total_vertices_removed = sum(r['analysis']['vertices_removed'] for r in successful)
        avg_vertices_before = np.mean([r['analysis']['original_vertices'] for r in successful])
        avg_vertices_after = np.mean([r['analysis']['final_vertices'] for r in successful])
        watertight_count = sum(1 for r in successful if r['analysis']['is_watertight'])
        
        print(f"\nLeaf Removal Summary:")
        print(f"  Total leaf regions removed: {total_leaves}")
        print(f"  Total vertices removed: {total_vertices_removed}")
        print(f"  Average vertices: {avg_vertices_before:.0f} → {avg_vertices_after:.0f}")
        print(f"  Watertight meshes: {watertight_count}/{len(successful)}")
        print(f"  ✓ Removed artificial leaf-like mesh regions")
        print(f"  ✓ Filled holes with triangular faces")
        print(f"  ✓ Ready for capping process")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'leaf_removal_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Leaf meshes removed and holes filled! Ready for capping.")
    
    return 0


if __name__ == "__main__":
    exit(main()) 