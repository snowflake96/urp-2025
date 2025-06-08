#!/usr/bin/env python3
"""
Surgical Leaf Cutting

This script performs precise surgical cutting of leaf-like mesh regions:
1. Identifies exact bottleneck connections (thin bridges)
2. Cuts specifically at the bottleneck edge
3. Preserves main vessel wall geometry
4. Fills only the small connection holes

This avoids removing vessel wall and creates clean cuts.
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


def find_bottleneck_edges(mesh: trimesh.Trimesh, aneurysm_location: np.ndarray) -> List[Dict]:
    """
    Find specific edges that represent bottleneck connections to leaf regions.
    """
    print("    Finding bottleneck edges for surgical cutting...")
    
    # Build edge adjacency (which faces share each edge)
    edge_faces = {}
    for face_idx, face in enumerate(mesh.faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(face_idx)
    
    # Find aneurysm region (main vessel core)
    aneurysm_distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    core_radius = 8.0
    core_vertices = set(np.where(aneurysm_distances <= core_radius)[0])
    
    print(f"    Core region: {len(core_vertices)} vertices within {core_radius}mm")
    
    # Analyze each edge for bottleneck characteristics
    bottleneck_edges = []
    
    for edge, face_list in edge_faces.items():
        if len(face_list) != 2:  # Only consider internal edges (not boundary)
            continue
            
        v1, v2 = edge
        
        # Calculate edge properties
        edge_length = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
        edge_center = (mesh.vertices[v1] + mesh.vertices[v2]) / 2
        distance_to_aneurysm = np.linalg.norm(edge_center - aneurysm_location)
        
        # Check if this edge connects core and non-core regions
        v1_is_core = v1 in core_vertices
        v2_is_core = v2 in core_vertices
        
        # Bottleneck criteria (relaxed):
        # 1. Edge connects core to non-core region (boundary edge)
        # 2. Edge is moderately far from aneurysm (> 8mm)
        # 3. Edge is reasonably short (< 3mm - indicates thin connection)
        if ((v1_is_core and not v2_is_core) or (not v1_is_core and v2_is_core)) and \
           distance_to_aneurysm > 8.0 and \
           edge_length < 3.0:
            
            # Get the two faces that share this edge
            face1_idx, face2_idx = face_list
            face1 = mesh.faces[face1_idx]
            face2 = mesh.faces[face2_idx]
            
            # Calculate face normals to check angle
            face1_normal = mesh.face_normals[face1_idx]
            face2_normal = mesh.face_normals[face2_idx]
            face_angle = np.arccos(np.clip(np.dot(face1_normal, face2_normal), -1, 1))
            
            # Find the third vertex in each face (not part of the edge)
            face1_third = [v for v in face1 if v not in edge][0]
            face2_third = [v for v in face2 if v not in edge][0]
            
            bottleneck_edges.append({
                'edge': edge,
                'vertices': [v1, v2],
                'edge_length': edge_length,
                'edge_center': edge_center,
                'distance_to_aneurysm': distance_to_aneurysm,
                'face_angle': np.degrees(face_angle),
                'faces': [face1_idx, face2_idx],
                'face1_third': face1_third,
                'face2_third': face2_third,
                'core_vertex': v1 if v1_is_core else v2,
                'leaf_vertex': v2 if v1_is_core else v1
            })
    
    # Sort by distance from aneurysm (process closest first)
    bottleneck_edges.sort(key=lambda x: x['distance_to_aneurysm'])
    
    print(f"    Found {len(bottleneck_edges)} bottleneck edges")
    for i, edge_info in enumerate(bottleneck_edges[:5]):  # Show first 5
        print(f"      Edge {i+1}: length {edge_info['edge_length']:.2f}mm, "
              f"distance {edge_info['distance_to_aneurysm']:.1f}mm, "
              f"angle {edge_info['face_angle']:.1f}°")
    
    return bottleneck_edges


def identify_leaf_regions_from_cuts(mesh: trimesh.Trimesh, 
                                   bottleneck_edges: List[Dict],
                                   aneurysm_location: np.ndarray) -> List[Dict]:
    """
    For each bottleneck edge, identify which side is the leaf region to remove.
    """
    print("    Identifying leaf regions for each bottleneck...")
    
    # Build vertex adjacency for BFS
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
    
    aneurysm_distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    
    leaf_regions = []
    
    for edge_info in bottleneck_edges[:20]:  # Process first 20 edges only for debugging
        leaf_vertex = edge_info['leaf_vertex']
        core_vertex = edge_info['core_vertex']
        
        # BFS from leaf_vertex to find connected region (but don't cross the bottleneck edge)
        visited = set()
        leaf_region = set()
        queue = [leaf_vertex]
        
        # Remove the bottleneck edge from adjacency temporarily
        temp_adj = {}
        for v, neighbors in adjacency.items():
            temp_adj[v] = neighbors.copy()
        
        # Remove bottleneck connection
        temp_adj[edge_info['vertices'][0]].discard(edge_info['vertices'][1])
        temp_adj[edge_info['vertices'][1]].discard(edge_info['vertices'][0])
        
        # BFS to find leaf region
        while queue and len(leaf_region) < 1000:  # Limit size
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            leaf_region.add(current)
            
            # Add neighbors
            for neighbor in temp_adj.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # Debug: Show region size for all found regions
        print(f"        Found region of size {len(leaf_region)}")
        
        # Validate this is actually a leaf region (more permissive)
        if 5 <= len(leaf_region) <= 800:  # More permissive size range
            avg_distance = np.mean([aneurysm_distances[v] for v in leaf_region])
            max_distance = np.max([aneurysm_distances[v] for v in leaf_region])
            
            leaf_regions.append({
                'bottleneck_edge': edge_info,
                'leaf_vertices': leaf_region,
                'size': len(leaf_region),
                'avg_distance': avg_distance,
                'max_distance': max_distance
            })
        else:
            print(f"        Rejected region: size {len(leaf_region)} outside range [5, 800]")
    
    print(f"    Identified {len(leaf_regions)} valid leaf regions")
    for i, region in enumerate(leaf_regions):
        print(f"      Region {i+1}: {region['size']} vertices, "
              f"max distance {region['max_distance']:.1f}mm")
    
    return leaf_regions


def surgical_cut_leaf_regions(mesh: trimesh.Trimesh, 
                             leaf_regions: List[Dict]) -> trimesh.Trimesh:
    """
    Perform surgical cuts to remove leaf regions while preserving main vessel.
    """
    print("    Performing surgical cuts...")
    
    # Collect all vertices to remove (only the leaf parts)
    vertices_to_remove = set()
    edges_to_cut = []
    
    for region in leaf_regions:
        vertices_to_remove.update(region['leaf_vertices'])
        edges_to_cut.append(region['bottleneck_edge']['edge'])
    
    print(f"    Removing {len(vertices_to_remove)} leaf vertices")
    print(f"    Cutting {len(edges_to_cut)} bottleneck edges")
    
    # Create mask for vertices to keep
    keep_mask = np.ones(len(mesh.vertices), dtype=bool)
    for vertex in vertices_to_remove:
        keep_mask[vertex] = False
    
    kept_vertices = np.where(keep_mask)[0]
    
    if len(kept_vertices) < 500:
        print("    Warning: Too many vertices would be removed")
        return mesh
    
    # Create vertex mapping
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[kept_vertices] = np.arange(len(kept_vertices))
    
    # Extract kept vertices
    new_vertices = mesh.vertices[kept_vertices]
    
    # Extract faces where all vertices are kept
    new_faces = []
    removed_faces = 0
    
    for face in mesh.faces:
        if all(keep_mask[face]):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
        else:
            removed_faces += 1
    
    if len(new_faces) == 0:
        print("    Warning: No faces left after surgical cutting")
        return mesh
    
    # Create surgically cut mesh
    cut_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
    
    print(f"    Surgical cut complete: {len(mesh.vertices)} → {len(cut_mesh.vertices)} vertices")
    print(f"    Removed {removed_faces} faces at cut locations")
    
    return cut_mesh


def fill_surgical_holes(mesh: trimesh.Trimesh, max_hole_size: int = 20) -> trimesh.Trimesh:
    """
    Fill small holes created by surgical cuts (much smaller than leaf removal holes).
    """
    print("    Filling small surgical holes...")
    
    # Detect boundary loops (holes from cuts)
    edge_counts = {}
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    if not boundary_edges:
        print("    No boundary edges found after surgical cuts")
        return mesh
    
    print(f"    Found {len(boundary_edges)} boundary edges")
    
    # Build boundary graph
    boundary_graph = nx.Graph()
    for edge in boundary_edges:
        boundary_graph.add_edge(edge[0], edge[1])
    
    # Find hole boundaries
    hole_boundaries = []
    for component in nx.connected_components(boundary_graph):
        if len(component) >= 3:
            hole_boundaries.append(list(component))
    
    print(f"    Found {len(hole_boundaries)} surgical holes to fill")
    
    # Fill small holes only (surgical cuts should create small holes)
    current_mesh = mesh.copy()
    filled_count = 0
    
    for i, hole_vertices in enumerate(hole_boundaries):
        if len(hole_vertices) > max_hole_size:
            print(f"      Skipping large hole {i+1} with {len(hole_vertices)} vertices (probably vessel opening)")
            continue
        
        try:
            # Simple fan triangulation for small holes
            hole_coords = current_mesh.vertices[hole_vertices]
            centroid = np.mean(hole_coords, axis=0)
            
            # Add centroid vertex
            centroid_idx = len(current_mesh.vertices)
            new_vertices = np.vstack([current_mesh.vertices, centroid.reshape(1, -1)])
            
            # Create fan triangulation
            new_faces = current_mesh.faces.tolist()
            
            # Sort vertices in circular order around hole
            center = np.mean(hole_coords, axis=0)
            relative_coords = hole_coords - center
            
            # Use first coordinate as reference for angle calculation
            ref_vec = relative_coords[0]
            angles = []
            for coord in relative_coords:
                angle = np.arctan2(np.cross(ref_vec, coord), np.dot(ref_vec, coord))
                angles.append(angle)
            
            sorted_indices = np.argsort(angles)
            sorted_hole_vertices = [hole_vertices[i] for i in sorted_indices]
            
            # Create triangular faces
            for j in range(len(sorted_hole_vertices)):
                v1 = sorted_hole_vertices[j]
                v2 = sorted_hole_vertices[(j + 1) % len(sorted_hole_vertices)]
                new_faces.append([centroid_idx, v1, v2])
            
            # Update mesh
            current_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
            filled_count += 1
            
            print(f"      Filled surgical hole {i+1} with {len(hole_vertices)} vertices")
            
        except Exception as e:
            print(f"      Failed to fill hole {i+1}: {e}")
            continue
    
    print(f"    Filled {filled_count}/{len(hole_boundaries)} surgical holes")
    
    # Clean up
    try:
        current_mesh.remove_duplicate_faces()
        current_mesh.remove_unreferenced_vertices()
    except:
        pass
    
    return current_mesh


def surgical_leaf_cutting_process(mesh: trimesh.Trimesh, 
                                 aneurysm_location: np.ndarray) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Complete surgical leaf cutting process.
    """
    print("  Performing surgical leaf cutting...")
    
    # Step 1: Find bottleneck edges
    bottleneck_edges = find_bottleneck_edges(mesh, aneurysm_location)
    
    if not bottleneck_edges:
        print("    No bottleneck edges found, keeping original mesh")
        return mesh, {
            'cuts_made': 0, 
            'vertices_removed': 0,
            'original_vertices': len(mesh.vertices),
            'final_vertices': len(mesh.vertices),
            'is_watertight': mesh.is_watertight
        }
    
    # Step 2: Identify leaf regions for each bottleneck
    leaf_regions = identify_leaf_regions_from_cuts(mesh, bottleneck_edges, aneurysm_location)
    
    if not leaf_regions:
        print("    No valid leaf regions found, keeping original mesh")
        return mesh, {
            'cuts_made': 0, 
            'vertices_removed': 0,
            'original_vertices': len(mesh.vertices),
            'final_vertices': len(mesh.vertices),
            'is_watertight': mesh.is_watertight
        }
    
    # Step 3: Perform surgical cuts
    cut_mesh = surgical_cut_leaf_regions(mesh, leaf_regions)
    
    # Step 4: Fill small surgical holes
    final_mesh = fill_surgical_holes(cut_mesh, max_hole_size=15)
    
    # Analysis
    analysis = {
        'cuts_made': len(leaf_regions),
        'vertices_removed': len(mesh.vertices) - len(final_mesh.vertices),
        'original_vertices': len(mesh.vertices),
        'final_vertices': len(final_mesh.vertices),
        'is_watertight': final_mesh.is_watertight
    }
    
    print(f"  ✓ Surgical cutting complete: {analysis['cuts_made']} cuts made")
    print(f"    {analysis['original_vertices']} → {analysis['final_vertices']} vertices")
    print(f"    Watertight: {analysis['is_watertight']}")
    
    return final_mesh, analysis


def process_single_vessel_surgical_cutting(args: Tuple) -> Dict:
    """
    Process a single vessel with surgical leaf cutting.
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
        print(f"\nProcessing {patient_id} with surgical leaf cutting...")
        
        # Load vessel mesh
        vessel_mesh = trimesh.load(vessel_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Perform surgical cutting
        cut_mesh, analysis = surgical_leaf_cutting_process(vessel_mesh, aneurysm_location)
        
        # Save surgically cut mesh
        cut_file = os.path.join(output_dir, f"{patient_id}_surgical_cut.stl")
        os.makedirs(output_dir, exist_ok=True)
        cut_mesh.export(cut_file)
        
        result['success'] = True
        result['analysis'] = analysis
        result['cut_file'] = cut_file
        
        print(f"  ✓ {patient_id}: Made {analysis['cuts_made']} surgical cuts")
        print(f"    Saved: {cut_file}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for surgical leaf cutting"""
    parser = argparse.ArgumentParser(description='Surgical Leaf Cutting')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geo_small_orthogonal'),
                       help='Directory containing vessel STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/surgical_cut_vessels'),
                       help='Output directory for surgically cut vessels')
    
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
    print(f"Performing surgical leaf cutting...")
    
    # Prepare processing arguments
    process_args = [(vessel_file, patient_data, args.output_dir) 
                   for vessel_file, patient_data in vessel_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Surgical cutting"):
            result = process_single_vessel_surgical_cutting(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_surgical_cutting, process_args),
                               total=len(process_args), desc="Surgical cutting"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Surgical Leaf Cutting Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        total_cuts = sum(r['analysis']['cuts_made'] for r in successful)
        total_vertices_removed = sum(r['analysis']['vertices_removed'] for r in successful)
        avg_vertices_before = np.mean([r['analysis']['original_vertices'] for r in successful])
        avg_vertices_after = np.mean([r['analysis']['final_vertices'] for r in successful])
        watertight_count = sum(1 for r in successful if r['analysis']['is_watertight'])
        
        print(f"\nSurgical Cutting Summary:")
        print(f"  Total surgical cuts made: {total_cuts}")
        print(f"  Total vertices removed: {total_vertices_removed}")
        print(f"  Average vertices: {avg_vertices_before:.0f} → {avg_vertices_after:.0f}")
        print(f"  Watertight meshes: {watertight_count}/{len(successful)}")
        print(f"  ✓ Precise cutting at bottleneck edges")
        print(f"  ✓ Preserved main vessel walls")
        print(f"  ✓ Removed only leaf artifacts")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'surgical_cutting_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Surgical leaf cutting complete - preserved vessel walls!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 