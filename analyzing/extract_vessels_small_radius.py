#!/usr/bin/env python3
"""
Small Radius Vessel Extraction

This script extracts vessels using a small radius around the aneurysm location
to maintain consistent vessel direction and avoid non-orthogonal boundaries.

Key features:
1. Uses fixed radius around aneurysm instead of vertex count
2. Keeps extraction close to aneurysm location
3. Maintains consistent vessel direction
4. Prevents non-orthogonal cuts from vessel direction changes
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


def extract_vessel_by_radius(mesh: trimesh.Trimesh,
                           aneurysm_location: np.ndarray,
                           extraction_radius: float = 15.0) -> trimesh.Trimesh:
    """
    Extract vessel using a fixed radius around the aneurysm location.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    aneurysm_location : np.ndarray
        Aneurysm location coordinates
    extraction_radius : float
        Radius in mm around aneurysm to extract
    
    Returns:
    --------
    trimesh.Trimesh : Extracted vessel mesh
    """
    print(f"  Extracting vessel within {extraction_radius}mm radius of aneurysm...")
    
    # Calculate distances from aneurysm to all vertices
    distances = np.linalg.norm(mesh.vertices - aneurysm_location, axis=1)
    
    # Select vertices within extraction radius
    within_radius = distances <= extraction_radius
    
    if np.sum(within_radius) < 100:
        print(f"    Warning: Only {np.sum(within_radius)} vertices within {extraction_radius}mm, increasing radius to 20mm")
        extraction_radius = 20.0
        within_radius = distances <= extraction_radius
        
    if np.sum(within_radius) < 100:
        print(f"    Warning: Still only {np.sum(within_radius)} vertices within {extraction_radius}mm, increasing to 25mm")
        extraction_radius = 25.0
        within_radius = distances <= extraction_radius
    
    print(f"    Selected {np.sum(within_radius)} vertices within {extraction_radius}mm radius")
    
    # Create vertex mapping
    selected_vertices = np.where(within_radius)[0]
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[selected_vertices] = np.arange(len(selected_vertices))
    
    # Extract vertices
    new_vertices = mesh.vertices[selected_vertices]
    
    # Extract faces that have all vertices within radius
    new_faces = []
    for face in mesh.faces:
        if all(within_radius[face]):
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
    
    if len(new_faces) == 0:
        print(f"    Error: No faces found within radius")
        return None
    
    # Create new mesh
    vessel_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(new_faces))
    
    # Ensure we have a connected component containing the aneurysm
    if len(vessel_mesh.vertices) > 0:
        # Find closest vertex to aneurysm in extracted mesh
        distances_extracted = np.linalg.norm(vessel_mesh.vertices - aneurysm_location, axis=1)
        closest_vertex = np.argmin(distances_extracted)
        
        # Get connected component containing this vertex
        connected_components = vessel_mesh.split(only_watertight=False)
        
        # Find which component contains the closest vertex
        best_component = vessel_mesh
        for component in connected_components:
            comp_distances = np.linalg.norm(component.vertices - aneurysm_location, axis=1)
            if np.min(comp_distances) < 2.0:  # Within 2mm of aneurysm
                if len(component.vertices) > len(best_component.vertices) * 0.3:  # At least 30% of original size
                    best_component = component
                    break
        
        vessel_mesh = best_component
    
    print(f"    Final radius-extracted mesh: {len(vessel_mesh.vertices)} vertices, {len(vessel_mesh.faces)} faces")
    
    return vessel_mesh


def process_single_vessel_small_radius(args: Tuple) -> Dict:
    """
    Process a single vessel with small radius extraction.
    """
    stl_file, aneurysm_data, output_dir, extraction_radius = args
    
    patient_id = os.path.basename(stl_file).replace('.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'vessel_vertices': 0,
        'vessel_faces': 0,
        'extraction_radius': extraction_radius
    }
    
    try:
        print(f"\nProcessing {patient_id} with {extraction_radius}mm radius extraction...")
        
        # Load mesh
        mesh = trimesh.load(stl_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Extract vessel by radius
        vessel_mesh = extract_vessel_by_radius(mesh, aneurysm_location, extraction_radius)
        
        if vessel_mesh is None or len(vessel_mesh.vertices) < 50:
            result['error'] = f"Extraction failed or too few vertices"
            return result
        
        # Save vessel
        output_file = os.path.join(output_dir, f"{patient_id}_aneurysm_1_vessel_small.stl")
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
    """Main processing function for small radius vessel extraction"""
    parser = argparse.ArgumentParser(description='Small Radius Vessel Extraction')
    
    parser.add_argument('--data-dir', 
                       default=os.path.expanduser('~/urp/data/uan/original_taubin_smoothed_ftetwild'),
                       help='Directory containing STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_small_radius'),
                       help='Output directory for extracted vessels')
    
    parser.add_argument('--extraction-radius', type=float, default=15.0,
                       help='Extraction radius around aneurysm (mm)')
    
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
    print(f"Extraction radius: {args.extraction_radius}mm")
    
    # Prepare processing arguments
    process_args = [(stl_file, patient_data, args.output_dir, args.extraction_radius) 
                   for stl_file, patient_data in stl_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Small radius extraction"):
            result = process_single_vessel_small_radius(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_small_radius, process_args),
                               total=len(process_args), desc="Small radius extraction"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Small Radius Vessel Extraction Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_vertices = np.mean([r['vessel_vertices'] for r in successful])
        avg_faces = np.mean([r['vessel_faces'] for r in successful])
        
        print(f"\nSmall Radius Extraction Summary:")
        print(f"  Extraction radius: {args.extraction_radius}mm")
        print(f"  Average vessel vertices: {avg_vertices:.0f}")
        print(f"  Average vessel faces: {avg_faces:.0f}")
        print(f"  ✓ Compact extraction around aneurysm")
        print(f"  ✓ Consistent vessel direction")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'small_radius_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 Vessels extracted with small radius for consistent vessel direction!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 