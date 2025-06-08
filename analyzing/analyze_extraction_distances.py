#!/usr/bin/env python3
"""
Analyze Extraction Distances

This script analyzes the geodesic + small + orthogonal vessel extractions
to determine how far each extraction crops from the aneurysm location.
Saves the data as JSON for all patients.
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


def analyze_single_patient_extraction(args: Tuple) -> Dict:
    """
    Analyze extraction distance for a single patient.
    """
    vessel_file, aneurysm_data, patient_id = args
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'extraction_analysis': None
    }
    
    try:
        # Load vessel mesh
        if not os.path.exists(vessel_file):
            result['error'] = f"Vessel file not found: {vessel_file}"
            return result
            
        vessel_mesh = trimesh.load(vessel_file)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Calculate distances from aneurysm to all vertices
        distances = np.linalg.norm(vessel_mesh.vertices - aneurysm_location, axis=1)
        
        # Find vessel extent and shape
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        
        # Calculate percentile distances
        percentile_50 = np.percentile(distances, 50)
        percentile_75 = np.percentile(distances, 75)
        percentile_90 = np.percentile(distances, 90)
        percentile_95 = np.percentile(distances, 95)
        
        # Analyze vessel shape using PCA
        pca = PCA(n_components=3)
        pca.fit(vessel_mesh.vertices)
        principal_components = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # Calculate vessel dimensions along principal axes
        centered_vertices = vessel_mesh.vertices - np.mean(vessel_mesh.vertices, axis=0)
        projections = np.dot(centered_vertices, principal_components.T)
        
        axis_lengths = []
        for i in range(3):
            axis_range = np.max(projections[:, i]) - np.min(projections[:, i])
            axis_lengths.append(axis_range)
        
        # Find vertices at maximum distances (extreme points)
        max_dist_vertex_idx = np.argmax(distances)
        max_dist_vertex_coords = vessel_mesh.vertices[max_dist_vertex_idx].tolist()
        
        # Count vertices in distance bands
        distance_bands = {
            '0-5mm': int(np.sum(distances <= 5.0)),
            '5-10mm': int(np.sum((distances > 5.0) & (distances <= 10.0))),
            '10-15mm': int(np.sum((distances > 10.0) & (distances <= 15.0))),
            '15-20mm': int(np.sum((distances > 15.0) & (distances <= 20.0))),
            '20-25mm': int(np.sum((distances > 20.0) & (distances <= 25.0))),
            'over_25mm': int(np.sum(distances > 25.0))
        }
        
        # Calculate mesh properties
        mesh_volume = vessel_mesh.volume if vessel_mesh.is_volume else None
        mesh_area = vessel_mesh.area
        is_watertight = vessel_mesh.is_watertight
        
        # Analyze extraction efficiency
        total_vertices = len(vessel_mesh.vertices)
        total_faces = len(vessel_mesh.faces)
        
        # Effective extraction radius (radius containing 90% of vertices)
        effective_radius = percentile_90
        
        extraction_analysis = {
            # Basic mesh info
            'total_vertices': int(total_vertices),
            'total_faces': int(total_faces),
            'mesh_volume_mm3': float(mesh_volume) if mesh_volume is not None else None,
            'mesh_area_mm2': float(mesh_area),
            'is_watertight': bool(is_watertight),
            
            # Distance statistics
            'distances_from_aneurysm': {
                'min_distance_mm': float(min_distance),
                'max_distance_mm': float(max_distance),
                'avg_distance_mm': float(avg_distance),
                'median_distance_mm': float(median_distance),
                'std_distance_mm': float(std_distance),
                'effective_radius_mm': float(effective_radius)
            },
            
            # Distance percentiles
            'distance_percentiles': {
                'p50_mm': float(percentile_50),
                'p75_mm': float(percentile_75),
                'p90_mm': float(percentile_90),
                'p95_mm': float(percentile_95)
            },
            
            # Vertex distribution by distance
            'distance_bands': distance_bands,
            
            # Vessel shape analysis
            'vessel_shape': {
                'principal_axis_lengths_mm': [float(x) for x in axis_lengths],
                'explained_variance_ratio': [float(x) for x in explained_variance_ratio],
                'max_extent_mm': float(max(axis_lengths)),
                'min_extent_mm': float(min(axis_lengths)),
                'aspect_ratio': float(max(axis_lengths) / min(axis_lengths)) if min(axis_lengths) > 0 else None
            },
            
            # Extreme points
            'extreme_points': {
                'max_distance_vertex_coords': max_dist_vertex_coords,
                'max_distance_mm': float(max_distance)
            },
            
            # Aneurysm reference
            'aneurysm_location': aneurysm_location.tolist(),
            
            # Extraction method info
            'extraction_method': 'geodesic_small_orthogonal',
            'extraction_parameters': {
                'target_vertices': 'adaptive',
                'max_distance_constraint': '16-18mm',
                'orthogonal_constraints': True,
                'vessel_direction_analysis': True
            }
        }
        
        result['success'] = True
        result['extraction_analysis'] = extraction_analysis
        
        print(f"  ✓ {patient_id}: max distance {max_distance:.1f}mm, effective radius {effective_radius:.1f}mm, {total_vertices} vertices")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main function to analyze extraction distances for all patients"""
    parser = argparse.ArgumentParser(description='Analyze Extraction Distances')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geo_small_orthogonal'),
                       help='Directory containing geodesic+small+orthogonal vessel files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-json',
                       default='../vessel_extraction_distances.json',
                       help='Output JSON file with extraction distance data')
    
    parser.add_argument('--workers', '-j', type=int, default=16,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Load aneurysm data
    print(f"Loading aneurysm data from: {args.aneurysm_json}")
    with open(args.aneurysm_json, 'r') as f:
        aneurysm_data = json.load(f)
    
    print(f"Analyzing extraction distances for geodesic+small+orthogonal vessels...")
    print(f"Vessel directory: {args.vessel_dir}")
    
    # Find vessel files
    vessel_files = []
    for patient_id, patient_data in aneurysm_data.items():
        vessel_file = os.path.join(args.vessel_dir, f"{patient_id}_aneurysm_1_vessel_geo_small.stl")
        if os.path.exists(vessel_file):
            vessel_files.append((vessel_file, patient_data, patient_id))
        else:
            print(f"Warning: Vessel file not found for {patient_id}")
    
    print(f"Found {len(vessel_files)} vessel files to analyze")
    
    # Prepare processing arguments
    process_args = vessel_files
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Analyzing distances"):
            result = analyze_single_patient_extraction(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(analyze_single_patient_extraction, process_args),
                               total=len(process_args), desc="Analyzing distances"))
    
    # Process results
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Extraction Distance Analysis Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    # Generate summary statistics
    if successful:
        max_distances = [r['extraction_analysis']['distances_from_aneurysm']['max_distance_mm'] for r in successful]
        effective_radii = [r['extraction_analysis']['distances_from_aneurysm']['effective_radius_mm'] for r in successful]
        total_vertices = [r['extraction_analysis']['total_vertices'] for r in successful]
        
        print(f"\nExtraction Distance Summary:")
        print(f"  Average max distance: {np.mean(max_distances):.1f} ± {np.std(max_distances):.1f} mm")
        print(f"  Range max distance: {np.min(max_distances):.1f} - {np.max(max_distances):.1f} mm")
        print(f"  Average effective radius (p90): {np.mean(effective_radii):.1f} ± {np.std(effective_radii):.1f} mm")
        print(f"  Average vertices per vessel: {np.mean(total_vertices):.0f} ± {np.std(total_vertices):.0f}")
        
        # Count vessels in different size categories
        small_vessels = sum(1 for d in max_distances if d <= 15.0)
        medium_vessels = sum(1 for d in max_distances if 15.0 < d <= 25.0)
        large_vessels = sum(1 for d in max_distances if d > 25.0)
        
        print(f"\nVessel Size Distribution:")
        print(f"  Small vessels (≤15mm): {small_vessels}")
        print(f"  Medium vessels (15-25mm): {medium_vessels}")
        print(f"  Large vessels (>25mm): {large_vessels}")
    
    if failed:
        print(f"\nFailed cases ({len(failed)}):")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Prepare final JSON output
    output_data = {
        'metadata': {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extraction_method': 'geodesic_small_orthogonal',
            'total_patients_analyzed': len(successful),
            'failed_analyses': len(failed),
            'vessel_directory': args.vessel_dir,
            'aneurysm_data_source': args.aneurysm_json
        },
        'summary_statistics': {
            'max_distance_mm': {
                'mean': float(np.mean(max_distances)) if successful else None,
                'std': float(np.std(max_distances)) if successful else None,
                'min': float(np.min(max_distances)) if successful else None,
                'max': float(np.max(max_distances)) if successful else None
            },
            'effective_radius_mm': {
                'mean': float(np.mean(effective_radii)) if successful else None,
                'std': float(np.std(effective_radii)) if successful else None,
                'min': float(np.min(effective_radii)) if successful else None,
                'max': float(np.max(effective_radii)) if successful else None
            },
            'vessel_sizes': {
                'small_vessels_le15mm': small_vessels if successful else 0,
                'medium_vessels_15_25mm': medium_vessels if successful else 0,
                'large_vessels_gt25mm': large_vessels if successful else 0
            }
        } if successful else {},
        'patient_data': {}
    }
    
    # Add individual patient data
    for result in successful:
        patient_id = result['patient_id']
        output_data['patient_data'][patient_id] = result['extraction_analysis']
    
    # Add failed cases
    if failed:
        output_data['failed_analyses'] = {
            result['patient_id']: result['error'] for result in failed
        }
    
    # Save JSON file
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nExtraction distance data saved to: {args.output_json}")
    print(f"🎯 Analysis complete - vessel extraction distances documented for all patients!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 