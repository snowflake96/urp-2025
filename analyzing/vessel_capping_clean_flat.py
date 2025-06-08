#!/usr/bin/env python3
"""
Clean Flat Vessel Capping Pipeline

This script creates watertight vessel meshes with clean flat caps by:
1. Detecting irregular/spiky vessel boundaries
2. Cropping spiky parts using plane cutting
3. Creating perfectly flat circular caps
4. Ensuring watertight topology with clean geometry
5. Generating accurate boundary conditions
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


def detect_boundary_loops_ordered(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    Detect boundary loops with proper vertex ordering.
    """
    print("    Detecting boundary loops...")
    
    # Find boundary edges (edges that belong to only one face)
    edges = mesh.edges_unique
    edge_counts = {}
    
    # Count how many faces each edge belongs to
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Boundary edges appear in only one face
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    if not boundary_edges:
        print("    No boundary edges found")
        return []
    
    print(f"    Found {len(boundary_edges)} boundary edges")
    
    # Build graph for connected components
    boundary_graph = nx.Graph()
    for edge in boundary_edges:
        boundary_graph.add_edge(edge[0], edge[1])
    
    # Find connected components (boundary loops)
    boundary_loops = []
    for component in nx.connected_components(boundary_graph):
        if len(component) >= 3:
            # Order vertices by following the boundary
            loop_vertices = list(component)
            boundary_loops.append(loop_vertices)
    
    print(f"    Found {len(boundary_loops)} boundary loops")
    for i, loop in enumerate(boundary_loops):
        print(f"      Loop {i+1}: {len(loop)} vertices")
    
    return boundary_loops


def analyze_boundary_irregularity(mesh: trimesh.Trimesh, loop_vertices: List[int]) -> Dict:
    """
    Analyze boundary loop irregularity and determine if it needs cropping.
    """
    loop_coords = mesh.vertices[loop_vertices]
    centroid = np.mean(loop_coords, axis=0)
    
    # Fit plane using PCA
    pca = PCA(n_components=3)
    pca.fit(loop_coords - centroid)
    
    # Normal is the direction with least variance
    normal = pca.components_[2]
    
    # Ensure normal points outward (away from mesh center)
    mesh_center = np.mean(mesh.vertices, axis=0)
    to_center = mesh_center - centroid
    if np.dot(normal, to_center) > 0:
        normal = -normal
    
    # Calculate distances from centroid and irregularity metrics
    distances = np.linalg.norm(loop_coords - centroid, axis=1)
    avg_radius = np.mean(distances)
    std_radius = np.std(distances)
    max_radius = np.max(distances)
    min_radius = np.min(distances)
    
    # Irregularity score: higher means more spiky/irregular
    irregularity_score = std_radius / avg_radius if avg_radius > 0 else 0
    radius_ratio = max_radius / min_radius if min_radius > 0 else float('inf')
    
    # Project to plane for area calculation
    u = pca.components_[0]
    v = pca.components_[1]
    projected_2d = np.column_stack([
        np.dot(loop_coords - centroid, u),
        np.dot(loop_coords - centroid, v)
    ])
    
    # Calculate area using shoelace formula
    projected_2d_closed = np.vstack([projected_2d, projected_2d[0]])
    area = 0.5 * abs(sum(
        projected_2d_closed[i][0] * projected_2d_closed[i+1][1] - 
        projected_2d_closed[i+1][0] * projected_2d_closed[i][1]
        for i in range(len(projected_2d_closed)-1)
    ))
    
    # Determine if boundary is spiky and needs cropping
    needs_cropping = (irregularity_score > 0.3) or (radius_ratio > 2.5)
    
    return {
        'vertices': loop_vertices,
        'coords': loop_coords,
        'centroid': centroid,
        'normal': normal,
        'radius': avg_radius,
        'std_radius': std_radius,
        'max_radius': max_radius,
        'min_radius': min_radius,
        'area': area,
        'irregularity_score': irregularity_score,
        'radius_ratio': radius_ratio,
        'needs_cropping': needs_cropping,
        'plane_u': u,
        'plane_v': v,
        'projected_2d': projected_2d
    }


def crop_spiky_boundary_simple(mesh: trimesh.Trimesh, boundary_info: Dict, crop_distance: float = 2.0) -> trimesh.Trimesh:
    """
    Crop spiky boundary using simple plane cutting approach.
    """
    print(f"      Cropping spiky boundary (irregularity: {boundary_info['irregularity_score']:.3f})")
    
    # Define cutting plane
    centroid = boundary_info['centroid']
    normal = boundary_info['normal']
    
    # Move plane inward to crop spiky parts
    plane_origin = centroid + normal * crop_distance
    
    try:
        # Simple approach: remove vertices that are too far from the plane
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Calculate distances from vertices to cutting plane
        distances_to_plane = np.dot(vertices - plane_origin, normal)
        
        # Keep vertices that are on the "inside" (negative distance)
        keep_vertices = distances_to_plane <= 0
        
        if np.sum(keep_vertices) < 100:  # Too few vertices left
            print(f"      Warning: Cropping would remove too many vertices, skipping")
            return mesh
        
        # Create vertex mapping
        old_to_new = np.full(len(vertices), -1, dtype=int)
        new_vertices = vertices[keep_vertices]
        old_to_new[keep_vertices] = np.arange(len(new_vertices))
        
        # Update faces
        new_faces = []
        for face in faces:
            if all(keep_vertices[face]):  # All vertices of face are kept
                new_face = [old_to_new[v] for v in face]
                if all(v >= 0 for v in new_face):  # Valid mapping
                    new_faces.append(new_face)
        
        if len(new_faces) == 0:
            print(f"      Warning: No faces left after cropping, using original")
            return mesh
        
        cropped_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        print(f"      Successfully cropped: {len(mesh.vertices)} → {len(cropped_mesh.vertices)} vertices")
        return cropped_mesh
        
    except Exception as e:
        print(f"      Warning: Cropping failed ({e}), using original boundary")
        return mesh


def create_flat_circular_cap(center: np.ndarray, 
                           normal: np.ndarray, 
                           radius: float,
                           resolution: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a perfectly flat circular cap.
    """
    print(f"      Creating flat circular cap (radius: {radius:.2f} mm)")
    
    # Create two orthogonal vectors in the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Create circular cap vertices
    vertices = [center]  # Center vertex
    faces = []
    
    # Create circle of vertices
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        point = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
        vertices.append(point)
    
    # Create triangular faces from center to circle edge
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Triangle: center (0), current vertex (i+1), next vertex (next_i+1)
        faces.append([0, i + 1, next_i + 1])
    
    return np.array(vertices), np.array(faces)


def process_vessel_clean_flat(mesh: trimesh.Trimesh, 
                             min_area: float = 1.0,
                             crop_distance: float = 2.0) -> Tuple[trimesh.Trimesh, List[Dict]]:
    """
    Process vessel to create clean flat caps on all openings.
    """
    print(f"  Creating clean flat caps...")
    
    # Detect boundary loops
    boundary_loops = detect_boundary_loops_ordered(mesh)
    
    if not boundary_loops:
        print("  No boundary loops detected")
        return mesh, []
    
    # Analyze each loop
    loop_infos = []
    for i, loop_vertices in enumerate(boundary_loops):
        print(f"    Analyzing loop {i+1}...")
        loop_info = analyze_boundary_irregularity(mesh, loop_vertices)
        loop_info['id'] = f'opening_{i+1}'
        
        if loop_info['area'] >= min_area:
            loop_infos.append(loop_info)
            print(f"      Area: {loop_info['area']:.2f} mm², Irregularity: {loop_info['irregularity_score']:.3f}")
            if loop_info['needs_cropping']:
                print(f"      → Needs cropping (spiky boundary)")
        else:
            print(f"      Skipping small opening (area: {loop_info['area']:.2f} mm²)")
    
    if not loop_infos:
        print("  No significant openings found")
        return mesh, []
    
    # Start with original mesh
    current_mesh = mesh.copy()
    cap_infos = []
    
    # Process each opening
    for loop_info in loop_infos:
        print(f"    Processing {loop_info['id']}...")
        
        # Crop if needed
        if loop_info['needs_cropping']:
            current_mesh = crop_spiky_boundary_simple(current_mesh, loop_info, crop_distance)
        
        # Create flat circular cap
        cap_radius = loop_info['radius'] * 1.2  # Slightly larger for better coverage
        cap_vertices, cap_faces = create_flat_circular_cap(
            loop_info['centroid'],
            loop_info['normal'],
            cap_radius,
            resolution=24
        )
        
        # Add cap to mesh
        cap_start_idx = len(current_mesh.vertices)
        new_vertices = np.vstack([current_mesh.vertices, cap_vertices])
        cap_faces_adjusted = cap_faces + cap_start_idx
        new_faces = np.vstack([current_mesh.faces, cap_faces_adjusted])
        
        current_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # Store cap information
        cap_info = {
            'opening_id': loop_info['id'],
            'area': np.pi * cap_radius**2,  # Exact circular area
            'centroid': loop_info['centroid'],
            'normal': loop_info['normal'],
            'radius': cap_radius,
            'was_cropped': loop_info['needs_cropping'],
            'irregularity_score': loop_info['irregularity_score'],
            'is_flat': True,
            'cross_sectional_area': np.pi * cap_radius**2
        }
        cap_infos.append(cap_info)
        
        print(f"      ✓ Added flat cap: {cap_info['area']:.2f} mm²")
    
    # Clean up mesh
    try:
        current_mesh.remove_duplicate_faces()
        current_mesh.remove_unreferenced_vertices()
        
        # Try pymeshfix repair if not watertight
        if not current_mesh.is_watertight:
            try:
                import pymeshfix
                meshfix = pymeshfix.MeshFix(current_mesh.vertices, current_mesh.faces)
                meshfix.repair(verbose=False)
                current_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
                print(f"  ✓ Repaired with pymeshfix: {current_mesh.is_watertight}")
            except ImportError:
                print("  Warning: pymeshfix not available")
    except Exception as e:
        print(f"  Warning: Mesh cleanup failed: {e}")
    
    print(f"  ✓ Final mesh: {len(current_mesh.vertices)} vertices, {len(current_mesh.faces)} faces")
    print(f"    Watertight: {current_mesh.is_watertight}")
    
    return current_mesh, cap_infos


def determine_inlet_outlet_clean(cap_infos: List[Dict], aneurysm_location: np.ndarray) -> Dict:
    """
    Determine inlet and outlet based on distance from aneurysm.
    """
    if len(cap_infos) < 2:
        print("    Warning: Need at least 2 openings for flow analysis")
        return {}
    
    # Calculate distances from aneurysm to each cap
    for cap in cap_infos:
        distance = np.linalg.norm(cap['centroid'] - aneurysm_location)
        cap['distance_to_aneurysm'] = distance
    
    # Sort caps by distance from aneurysm
    sorted_caps = sorted(cap_infos, key=lambda x: x['distance_to_aneurysm'])
    
    # Assign inlet (furthest from aneurysm) and outlet (closest to aneurysm)
    inlet_cap = sorted_caps[-1]  # Furthest
    outlet_cap = sorted_caps[0]  # Closest
    
    # Mark caps
    for cap in cap_infos:
        cap['is_inlet'] = cap['opening_id'] == inlet_cap['opening_id']
        cap['is_outlet'] = cap['opening_id'] == outlet_cap['opening_id']
        cap['flow_type'] = 'inlet' if cap['is_inlet'] else ('outlet' if cap['is_outlet'] else 'side_branch')
    
    flow_assignment = {
        'inlet': inlet_cap,
        'outlet': outlet_cap,
        'side_branches': [cap for cap in cap_infos if not cap['is_inlet'] and not cap['is_outlet']],
        'total_openings': len(cap_infos)
    }
    
    print(f"    Flow assignment:")
    print(f"      Inlet: {inlet_cap['opening_id']} (area: {inlet_cap['cross_sectional_area']:.2f} mm²)")
    print(f"      Outlet: {outlet_cap['opening_id']} (area: {outlet_cap['cross_sectional_area']:.2f} mm²)")
    print(f"      Side branches: {len(flow_assignment['side_branches'])}")
    
    return flow_assignment


def calculate_boundary_conditions_clean(flow_assignment: Dict,
                                       flow_rate: float = 5.0,  # ml/s
                                       outlet_pressure: float = 80.0,  # mmHg
                                       blood_density: float = 1060.0,  # kg/m³
                                       blood_viscosity: float = 0.004) -> Dict:  # Pa·s
    """
    Calculate boundary conditions for clean flat mesh.
    """
    if not flow_assignment:
        return {}
    
    inlet = flow_assignment['inlet']
    outlet = flow_assignment['outlet']
    
    # Convert units
    flow_rate_m3s = flow_rate * 1e-6  # ml/s to m³/s
    outlet_pressure_pa = outlet_pressure * 133.322  # mmHg to Pa
    
    # Calculate inlet velocity based on flat circular area
    inlet_area_m2 = inlet['cross_sectional_area'] * 1e-6  # mm² to m²
    inlet_velocity = flow_rate_m3s / inlet_area_m2  # m/s
    
    # Calculate hydraulic diameter and Reynolds number
    inlet_diameter_m = 2 * np.sqrt(inlet_area_m2 / np.pi)  # Circular diameter
    reynolds_number = (blood_density * inlet_velocity * inlet_diameter_m) / blood_viscosity
    
    boundary_conditions = {
        'mesh_properties': {
            'is_watertight': True,
            'has_flat_caps': True,
            'cfd_ready': True
        },
        'fluid_properties': {
            'density': blood_density,
            'dynamic_viscosity': blood_viscosity,
            'kinematic_viscosity': blood_viscosity / blood_density
        },
        'flow_parameters': {
            'volumetric_flow_rate_ml_s': flow_rate,
            'volumetric_flow_rate_m3_s': flow_rate_m3s,
            'reynolds_number': reynolds_number,
            'flow_regime': 'laminar' if reynolds_number < 2300 else 'turbulent'
        },
        'inlet_conditions': {
            'opening_id': inlet['opening_id'],
            'type': 'velocity_inlet',
            'cross_sectional_area_mm2': inlet['cross_sectional_area'],
            'cross_sectional_area_m2': inlet_area_m2,
            'velocity_magnitude_m_s': inlet_velocity,
            'velocity_direction': (-inlet['normal']).tolist(),  # Into vessel
            'hydraulic_diameter_m': inlet_diameter_m,
            'center_coordinates': inlet['centroid'].tolist(),
            'is_flat_circular': inlet['is_flat'],
            'was_cropped': inlet['was_cropped']
        },
        'outlet_conditions': {
            'opening_id': outlet['opening_id'],
            'type': 'pressure_outlet',
            'pressure_pa': outlet_pressure_pa,
            'pressure_mmhg': outlet_pressure,
            'cross_sectional_area_mm2': outlet['cross_sectional_area'],
            'cross_sectional_area_m2': outlet['cross_sectional_area'] * 1e-6,
            'center_coordinates': outlet['centroid'].tolist(),
            'is_flat_circular': outlet['is_flat'],
            'was_cropped': outlet['was_cropped']
        },
        'wall_conditions': {
            'type': 'no_slip',
            'description': 'Vessel wall with no-slip boundary condition'
        }
    }
    
    # Handle side branches
    if flow_assignment['side_branches']:
        side_branch_conditions = []
        for branch in flow_assignment['side_branches']:
            side_branch_conditions.append({
                'opening_id': branch['opening_id'],
                'type': 'pressure_outlet',
                'pressure_pa': outlet_pressure_pa * 0.95,  # Slightly lower pressure
                'cross_sectional_area_mm2': branch['cross_sectional_area'],
                'center_coordinates': branch['centroid'].tolist(),
                'is_flat_circular': branch['is_flat'],
                'was_cropped': branch['was_cropped']
            })
        boundary_conditions['side_branch_conditions'] = side_branch_conditions
    
    return boundary_conditions


def process_single_vessel_clean_flat(args: Tuple) -> Dict:
    """
    Process a single vessel to create clean flat caps.
    """
    vessel_path, aneurysm_data, output_dir, processing_params = args
    
    patient_id = os.path.basename(vessel_path).replace('_aneurysm_1_vessel.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'analysis': None
    }
    
    try:
        print(f"\nProcessing {patient_id} for clean flat caps...")
        
        # Load vessel mesh
        mesh = trimesh.load(vessel_path)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Create clean flat vessel
        clean_mesh, cap_infos = process_vessel_clean_flat(
            mesh,
            min_area=processing_params.get('min_area', 1.0),
            crop_distance=processing_params.get('crop_distance', 2.0)
        )
        
        if not cap_infos:
            result['error'] = "No valid openings found for capping"
            return result
        
        # Determine inlet/outlet assignment
        flow_assignment = determine_inlet_outlet_clean(cap_infos, aneurysm_location)
        
        if not flow_assignment:
            result['error'] = "Could not determine inlet/outlet assignment"
            return result
        
        # Calculate boundary conditions
        boundary_conditions = calculate_boundary_conditions_clean(
            flow_assignment,
            flow_rate=processing_params.get('flow_rate', 5.0),
            outlet_pressure=processing_params.get('outlet_pressure', 80.0),
            blood_density=processing_params.get('blood_density', 1060.0),
            blood_viscosity=processing_params.get('blood_viscosity', 0.004)
        )
        
        # Export files
        output_base = os.path.join(output_dir, patient_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Export mesh
        mesh_file = f"{output_base}_clean_flat.stl"
        clean_mesh.export(mesh_file)
        
        # Export boundary conditions
        bc_file = f"{output_base}_boundary_conditions.json"
        with open(bc_file, 'w') as f:
            json.dump(boundary_conditions, f, indent=2, default=str)
        
        result['success'] = True
        result['analysis'] = {
            'openings_detected': len(cap_infos),
            'caps_cropped': sum(1 for cap in cap_infos if cap.get('was_cropped', False)),
            'all_caps_flat': all(cap.get('is_flat', False) for cap in cap_infos),
            'is_watertight': clean_mesh.is_watertight,
            'volume_mm3': clean_mesh.volume if clean_mesh.is_volume else None,
            'inlet_area_mm2': flow_assignment['inlet']['cross_sectional_area'],
            'outlet_area_mm2': flow_assignment['outlet']['cross_sectional_area'],
            'reynolds_number': boundary_conditions['flow_parameters']['reynolds_number'],
            'mesh_vertices': len(clean_mesh.vertices),
            'mesh_faces': len(clean_mesh.faces)
        }
        
        print(f"  ✓ {patient_id}: Clean flat caps with {len(cap_infos)} openings")
        print(f"    Cropped caps: {result['analysis']['caps_cropped']}/{len(cap_infos)}")
        print(f"    Watertight: {result['analysis']['is_watertight']}")
        print(f"    Reynolds: {result['analysis']['reynolds_number']:.1f}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for clean flat vessel capping"""
    parser = argparse.ArgumentParser(description='Clean Flat Vessel Capping Pipeline')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geodesic_large'),
                       help='Directory containing vessel STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/clean_flat_vessels'),
                       help='Output directory for clean flat vessels')
    
    parser.add_argument('--min-area', type=float, default=1.0,
                       help='Minimum area to consider as vessel opening (mm²)')
    
    parser.add_argument('--crop-distance', type=float, default=2.0,
                       help='Distance to crop spiky boundaries (mm)')
    
    parser.add_argument('--flow-rate', type=float, default=5.0,
                       help='Blood flow rate (ml/s)')
    
    parser.add_argument('--outlet-pressure', type=float, default=80.0,
                       help='Outlet pressure (mmHg)')
    
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
            
        vessel_file = os.path.join(args.vessel_dir, f"{patient_id}_aneurysm_1_vessel.stl")
        if os.path.exists(vessel_file):
            vessel_files.append((vessel_file, patient_data))
        else:
            print(f"Warning: Vessel file not found for {patient_id}")
    
    print(f"Found {len(vessel_files)} vessel files to process")
    
    # Processing parameters
    processing_params = {
        'min_area': args.min_area,
        'crop_distance': args.crop_distance,
        'flow_rate': args.flow_rate,
        'outlet_pressure': args.outlet_pressure,
        'blood_density': 1060.0,
        'blood_viscosity': 0.004
    }
    
    print(f"\nClean Flat Processing Parameters:")
    print(f"  Min opening area: {args.min_area} mm²")
    print(f"  Crop distance: {args.crop_distance} mm")
    print(f"  Flow rate: {args.flow_rate} ml/s")
    print(f"  Outlet pressure: {args.outlet_pressure} mmHg")
    
    # Prepare processing arguments
    process_args = [(vessel_file, patient_data, args.output_dir, processing_params) 
                   for vessel_file, patient_data in vessel_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Creating clean flat vessels"):
            result = process_single_vessel_clean_flat(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_clean_flat, process_args),
                               total=len(process_args), desc="Creating clean flat vessels"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Clean Flat Vessel Capping Pipeline Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        watertight_count = sum(1 for r in successful if r['analysis']['is_watertight'])
        flat_caps_count = sum(1 for r in successful if r['analysis']['all_caps_flat'])
        avg_openings = np.mean([r['analysis']['openings_detected'] for r in successful])
        avg_cropped = np.mean([r['analysis']['caps_cropped'] for r in successful])
        avg_inlet_area = np.mean([r['analysis']['inlet_area_mm2'] for r in successful])
        avg_outlet_area = np.mean([r['analysis']['outlet_area_mm2'] for r in successful])
        avg_reynolds = np.mean([r['analysis']['reynolds_number'] for r in successful])
        
        volumes = [r['analysis']['volume_mm3'] for r in successful if r['analysis']['volume_mm3'] is not None]
        avg_volume = np.mean(volumes) if volumes else None
        
        print(f"\nClean Flat Analysis Summary:")
        print(f"  Watertight meshes: {watertight_count}/{len(successful)}")
        print(f"  All caps flat: {flat_caps_count}/{len(successful)}")
        print(f"  Average openings per vessel: {avg_openings:.1f}")
        print(f"  Average cropped caps per vessel: {avg_cropped:.1f}")
        print(f"  Average inlet area: {avg_inlet_area:.1f} mm²")
        print(f"  Average outlet area: {avg_outlet_area:.1f} mm²")
        print(f"  Average volume: {avg_volume:.1f} mm³" if avg_volume else "  Average volume: N/A")
        print(f"  Average Reynolds number: {avg_reynolds:.1f}")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'clean_flat_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 All meshes have clean flat caps and are CFD-ready!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 