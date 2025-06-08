#!/usr/bin/env python3
"""
Volumetric Random Walk Aneurysm Vessel Extraction

This script performs volumetric random walk analysis starting from aneurysm locations
to extract connected vessel regions by walking INSIDE the vessel mesh volume.
"""

import json
import os
import numpy as np
import trimesh
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import time


def find_inside_start_point(mesh: trimesh.Trimesh, target_point: np.ndarray, search_radius: float = 10.0) -> np.ndarray:
    """
    Find a point inside the mesh near the target point.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    target_point : np.ndarray
        Target aneurysm point (may be outside mesh)
    search_radius : float
        Radius to search for inside points
    
    Returns:
    --------
    np.ndarray : Point inside the mesh, or closest point if none found inside
    """
    # First check if target point is already inside
    if mesh.contains([target_point])[0]:
        return target_point
    
    # Generate candidate points around the target
    num_candidates = 100
    angles = np.random.uniform(0, 2*np.pi, num_candidates)
    elevations = np.random.uniform(-np.pi/2, np.pi/2, num_candidates)
    radii = np.random.uniform(0.1, search_radius, num_candidates)
    
    # Convert to cartesian coordinates
    candidates = []
    for i in range(num_candidates):
        x = target_point[0] + radii[i] * np.cos(elevations[i]) * np.cos(angles[i])
        y = target_point[1] + radii[i] * np.cos(elevations[i]) * np.sin(angles[i])
        z = target_point[2] + radii[i] * np.sin(elevations[i])
        candidates.append([x, y, z])
    
    candidates = np.array(candidates)
    
    # Check which candidates are inside the mesh
    inside_mask = mesh.contains(candidates)
    inside_candidates = candidates[inside_mask]
    
    if len(inside_candidates) > 0:
        # Return the inside point closest to target
        distances = np.linalg.norm(inside_candidates - target_point, axis=1)
        closest_idx = np.argmin(distances)
        return inside_candidates[closest_idx]
    
    # If no inside points found, return closest surface point
    closest_point, _, _ = mesh.nearest.on_surface([target_point])
    return closest_point[0]


def volumetric_random_walk(mesh: trimesh.Trimesh,
                          start_point: np.ndarray,
                          max_steps: int = 5000,
                          step_size: float = 1.0,
                          step_probability: float = 0.8,
                          distance_threshold: float = 50.0) -> Set[tuple]:
    """
    Perform volumetric random walk inside the vessel mesh.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    start_point : np.ndarray
        Starting point (should be inside mesh)
    max_steps : int
        Maximum number of walk steps
    step_size : float
        Size of each step
    step_probability : float
        Probability of continuing walk
    distance_threshold : float
        Maximum distance from start point
    
    Returns:
    --------
    Set[tuple] : Set of points visited during walk (as tuples for hashing)
    """
    walked_points = set()
    current_point = start_point.copy()
    
    # Add start point
    walked_points.add(tuple(current_point))
    
    # Multiple random walks from start point
    num_walks = 20
    for walk_id in range(num_walks):
        current_point = start_point.copy()
        
        for step in range(max_steps // num_walks):
            # Check distance constraint
            distance = np.linalg.norm(current_point - start_point)
            if distance > distance_threshold:
                break
            
            # Probability of stopping
            if np.random.random() > step_probability:
                break
            
            # Generate random step direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)  # Normalize
            
            # Take step
            candidate_point = current_point + step_size * direction
            
            # Check if new point is inside mesh
            if mesh.contains([candidate_point])[0]:
                current_point = candidate_point
                walked_points.add(tuple(current_point))
            else:
                # If outside, try smaller steps or different directions
                for attempt in range(5):
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    smaller_step = step_size * (0.5 ** attempt)
                    candidate_point = current_point + smaller_step * direction
                    
                    if mesh.contains([candidate_point])[0]:
                        current_point = candidate_point
                        walked_points.add(tuple(current_point))
                        break
                else:
                    # If all attempts fail, stop this walk
                    break
    
    return walked_points


def extract_vessel_region_from_walk(mesh: trimesh.Trimesh,
                                   walked_points: Set[tuple],
                                   expansion_radius: float = 5.0) -> trimesh.Trimesh:
    """
    Extract vessel mesh region based on volumetric random walk points.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original vessel mesh
    walked_points : Set[tuple]
        Points visited during random walk
    expansion_radius : float
        Radius for expanding around walked points
    
    Returns:
    --------
    trimesh.Trimesh : Extracted vessel region
    """
    if not walked_points:
        return trimesh.Trimesh()
    
    # Convert walked points to array
    walk_points = np.array(list(walked_points))
    
    # Find mesh vertices near walked points
    distances = cdist(mesh.vertices, walk_points)
    min_distances = distances.min(axis=1)
    
    # Select vertices within expansion radius of walked points
    selected_vertex_mask = min_distances <= expansion_radius
    selected_vertex_indices = np.where(selected_vertex_mask)[0]
    
    if len(selected_vertex_indices) == 0:
        return trimesh.Trimesh()
    
    # Find faces that have all vertices selected
    vertex_set = set(selected_vertex_indices)
    face_mask = np.all(np.isin(mesh.faces, selected_vertex_indices), axis=1)
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


def process_single_aneurysm_volumetric(args: Tuple) -> Dict:
    """
    Process a single aneurysm for volumetric vessel extraction
    
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
        
        # Process each aneurysm
        for aneurysm in aneurysm_data['aneurysms']:
            aneurysm_id = aneurysm['aneurysm_id']
            aneurysm_point = np.array(aneurysm['mesh_vertex_coords'])
            
            print(f"  Extracting {aneurysm_id} from point {aneurysm_point}...")
            
            # Step 1: Find inside start point
            inside_start_point = find_inside_start_point(
                mesh=mesh,
                target_point=aneurysm_point,
                search_radius=extraction_params.get('search_radius', 10.0)
            )
            
            print(f"    Start point moved from {aneurysm_point} to {inside_start_point}")
            
            # Step 2: Perform volumetric random walk inside the vessel
            walked_points = volumetric_random_walk(
                mesh=mesh,
                start_point=inside_start_point,
                max_steps=extraction_params.get('max_steps', 5000),
                step_size=extraction_params.get('step_size', 1.0),
                step_probability=extraction_params.get('step_probability', 0.8),
                distance_threshold=extraction_params.get('distance_threshold', 50.0)
            )
            
            if not walked_points:
                print(f"    Warning: No valid walk points for {aneurysm_id}")
                continue
            
            print(f"    Volumetric walk visited {len(walked_points)} points inside vessel")
            
            # Step 3: Extract vessel region based on walked points
            extracted_mesh = extract_vessel_region_from_walk(
                mesh=mesh,
                walked_points=walked_points,
                expansion_radius=extraction_params.get('expansion_radius', 5.0)
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
                'original_point': aneurysm_point.tolist(),
                'inside_start_point': inside_start_point.tolist(),
                'walked_points_count': len(walked_points),
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
        description='Extract aneurysm-connected vessel regions using volumetric random walk'
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
        default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_volumetric'),
        help='Output directory for extracted vessel meshes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=15000,
        help='Maximum random walk steps (default: 15000)'
    )
    parser.add_argument(
        '--step-size',
        type=float,
        default=1.0,
        help='Size of each random walk step (default: 1.0)'
    )
    parser.add_argument(
        '--step-probability',
        type=float,
        default=0.85,
        help='Probability of continuing random walk (default: 0.85)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=40.0,
        help='Maximum distance from aneurysm center (default: 40.0)'
    )
    parser.add_argument(
        '--expansion-radius',
        type=float,
        default=5.0,
        help='Radius for expanding around walked points (default: 5.0)'
    )
    parser.add_argument(
        '--search-radius',
        type=float,
        default=10.0,
        help='Radius for searching inside start point (default: 10.0)'
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
        'max_steps': args.max_steps,
        'step_size': args.step_size,
        'step_probability': args.step_probability,
        'distance_threshold': args.distance_threshold,
        'expansion_radius': args.expansion_radius,
        'search_radius': args.search_radius
    }
    
    print("\nVolumetric Random Walk Parameters:")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Step probability: {args.step_probability}")
    print(f"  - Distance threshold: {args.distance_threshold}")
    print(f"  - Expansion radius: {args.expansion_radius}")
    print(f"  - Search radius: {args.search_radius}")
    
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
    
    print(f"\nProcessing {len(process_args)} patients with volumetric random walk")
    
    # Process patients
    start_time = time.time()
    if args.workers == 1:
        # Sequential processing
        results = []
        for process_arg in tqdm(process_args, desc="Extracting vessels"):
            result = process_single_aneurysm_volumetric(process_arg)
            results.append(result)
    else:
        # Parallel processing
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_aneurysm_volumetric, process_args), 
                               total=len(process_args), desc="Extracting vessels"))
    
    # Summary
    total_time = time.time() - start_time
    successful_patients = sum(1 for r in results if r['success'])
    total_aneurysms = sum(len(r['extracted_aneurysms']) for r in results if r['success'])
    
    print(f"\n" + "="*60)
    print(f"Volumetric vessel extraction complete in {total_time:.1f} seconds")
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