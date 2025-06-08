#!/usr/bin/env python3
"""
Advanced Vessel Capping Pipeline

This script properly processes vessel meshes by:
1. Detecting all vessel openings/holes
2. Identifying and cutting spiky/irregular edges  
3. Creating clean planar cuts
4. Filling openings with flat caps
5. Computing accurate cross-sectional areas
6. Generating precise boundary conditions for CFD/FEA
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
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import networkx as nx


def detect_boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    Detect boundary loops (holes) in the mesh using edge analysis.
    
    Returns:
    --------
    List[List[int]] : List of boundary loops, each as list of vertex indices
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
    
    # Build adjacency graph of boundary vertices
    boundary_graph = nx.Graph()
    for edge in boundary_edges:
        boundary_graph.add_edge(edge[0], edge[1])
    
    # Find connected components (boundary loops)
    boundary_loops = []
    for component in nx.connected_components(boundary_graph):
        if len(component) >= 3:  # Need at least 3 vertices for a loop
            # Order vertices to form a proper loop
            loop_vertices = list(component)
            if len(loop_vertices) > 3:
                # Find ordering by traversing the graph
                ordered_loop = []
                current = loop_vertices[0]
                visited = {current}
                ordered_loop.append(current)
                
                while len(ordered_loop) < len(loop_vertices):
                    neighbors = [n for n in boundary_graph.neighbors(current) if n not in visited]
                    if neighbors:
                        current = neighbors[0]
                        visited.add(current)
                        ordered_loop.append(current)
                    else:
                        break
                
                if len(ordered_loop) >= 3:
                    boundary_loops.append(ordered_loop)
            else:
                boundary_loops.append(loop_vertices)
    
    print(f"    Found {len(boundary_loops)} boundary loops")
    for i, loop in enumerate(boundary_loops):
        print(f"      Loop {i+1}: {len(loop)} vertices")
    
    return boundary_loops


def analyze_loop_geometry(mesh: trimesh.Trimesh, loop_vertices: List[int]) -> Dict:
    """
    Analyze the geometry of a boundary loop to determine cutting plane.
    
    Returns:
    --------
    Dict : Loop geometric properties
    """
    loop_coords = mesh.vertices[loop_vertices]
    centroid = np.mean(loop_coords, axis=0)
    
    # Fit plane using PCA
    pca = PCA(n_components=3)
    pca.fit(loop_coords - centroid)
    
    # Normal is the direction with least variance
    normal = pca.components_[2]
    
    # Ensure normal points outward (heuristic: away from mesh center)
    mesh_center = np.mean(mesh.vertices, axis=0)
    to_center = mesh_center - centroid
    if np.dot(normal, to_center) > 0:
        normal = -normal
    
    # Calculate geometric properties
    distances = np.linalg.norm(loop_coords - centroid, axis=1)
    avg_radius = np.mean(distances)
    max_radius = np.max(distances)
    std_radius = np.std(distances)
    
    # Calculate area using convex hull projection
    try:
        # Project points onto the fitted plane
        u = pca.components_[0]
        v = pca.components_[1]
        projected_2d = np.column_stack([
            np.dot(loop_coords - centroid, u),
            np.dot(loop_coords - centroid, v)
        ])
        
        hull = ConvexHull(projected_2d)
        area = hull.volume  # In 2D, volume is area
    except:
        # Fallback: approximate as circle
        area = np.pi * avg_radius**2
    
    # Detect irregularity (spikiness)
    irregularity_score = std_radius / avg_radius if avg_radius > 0 else 0
    is_irregular = irregularity_score > 0.3  # Threshold for irregular boundary
    
    return {
        'vertices': loop_vertices,
        'coords': loop_coords,
        'centroid': centroid,
        'normal': normal,
        'avg_radius': avg_radius,
        'max_radius': max_radius,
        'std_radius': std_radius,
        'area': area,
        'irregularity_score': irregularity_score,
        'is_irregular': is_irregular,
        'plane_basis': (pca.components_[0], pca.components_[1], normal)
    }


def cut_irregular_boundary(mesh: trimesh.Trimesh, loop_info: Dict, 
                          cut_distance: float = 5.0) -> trimesh.Trimesh:
    """
    Cut irregular boundary to create clean opening.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original mesh
    loop_info : Dict
        Loop geometry information
    cut_distance : float
        Distance to cut inward from boundary
    
    Returns:
    --------
    trimesh.Trimesh : Mesh with clean cut
    """
    if not loop_info['is_irregular']:
        print(f"    Loop is regular, no cutting needed")
        return mesh
    
    print(f"    Cutting irregular boundary (irregularity: {loop_info['irregularity_score']:.3f})")
    
    centroid = loop_info['centroid']
    normal = loop_info['normal']
    
    # Define cutting plane slightly inward from the boundary
    cutting_plane_origin = centroid - normal * cut_distance
    cutting_plane_normal = normal
    
    # Cut mesh with plane
    try:
        # Use trimesh's slice_plane method
        cut_mesh = mesh.slice_plane(cutting_plane_origin, cutting_plane_normal)
        
        if cut_mesh is None or len(cut_mesh.vertices) == 0:
            print("    Warning: Cutting failed, returning original mesh")
            return mesh
        
        print(f"    Cut mesh: {len(cut_mesh.vertices)} vertices (was {len(mesh.vertices)})")
        return cut_mesh
        
    except Exception as e:
        print(f"    Warning: Cutting failed ({e}), returning original mesh")
        return mesh


def create_planar_cap(loop_info: Dict, mesh_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a clean planar cap for a boundary loop.
    
    Parameters:
    -----------
    loop_info : Dict
        Loop geometry information
    mesh_scale : float
        Scale factor for cap resolution
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : Cap vertices and faces
    """
    centroid = loop_info['centroid']
    normal = loop_info['normal']
    u, v, n = loop_info['plane_basis']
    avg_radius = loop_info['avg_radius']
    
    # Create circular cap with appropriate resolution
    n_radial = max(16, int(avg_radius * mesh_scale * 2))
    n_angular = max(16, int(avg_radius * mesh_scale))
    
    # Generate cap vertices in polar coordinates
    cap_vertices = [centroid]  # Center vertex
    cap_faces = []
    
    # Create concentric rings
    radii = np.linspace(0.1 * avg_radius, avg_radius, n_radial)
    angles = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    vertex_idx = 1
    prev_ring_start = 0
    prev_ring_size = 1
    
    for r_idx, radius in enumerate(radii):
        ring_start = vertex_idx
        
        # Add vertices for this ring
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Convert from 2D polar to 3D world coordinates
            point_3d = centroid + x * u + y * v
            cap_vertices.append(point_3d)
            vertex_idx += 1
        
        ring_size = len(angles)
        
        # Create faces connecting to previous ring
        if r_idx == 0:
            # Connect center to first ring
            for i in range(ring_size):
                next_i = (i + 1) % ring_size
                cap_faces.append([0, ring_start + i, ring_start + next_i])
        else:
            # Connect current ring to previous ring
            for i in range(ring_size):
                curr_v1 = ring_start + i
                curr_v2 = ring_start + (i + 1) % ring_size
                prev_v1 = prev_ring_start + i
                prev_v2 = prev_ring_start + (i + 1) % ring_size
                
                # Create two triangles
                cap_faces.append([prev_v1, curr_v1, curr_v2])
                cap_faces.append([prev_v1, curr_v2, prev_v2])
        
        prev_ring_start = ring_start
        prev_ring_size = ring_size
    
    return np.array(cap_vertices), np.array(cap_faces)


def process_vessel_openings(mesh: trimesh.Trimesh, 
                           cut_distance: float = 5.0,
                           min_area: float = 1.0) -> Tuple[trimesh.Trimesh, List[Dict]]:
    """
    Process all vessel openings: detect, cut irregular edges, and cap.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input vessel mesh
    cut_distance : float
        Distance to cut inward from irregular boundaries
    min_area : float
        Minimum area to consider as vessel opening
    
    Returns:
    --------
    Tuple[trimesh.Trimesh, List[Dict]] : Processed mesh and opening information
    """
    print(f"  Processing vessel openings...")
    
    # Detect boundary loops
    boundary_loops = detect_boundary_loops(mesh)
    
    if not boundary_loops:
        print("  No boundary loops detected")
        return mesh, []
    
    # Analyze each loop
    loop_infos = []
    for i, loop_vertices in enumerate(boundary_loops):
        print(f"    Analyzing loop {i+1}...")
        loop_info = analyze_loop_geometry(mesh, loop_vertices)
        loop_info['id'] = f'opening_{i+1}'
        
        if loop_info['area'] >= min_area:
            loop_infos.append(loop_info)
            print(f"      Area: {loop_info['area']:.2f}, "
                  f"Avg radius: {loop_info['avg_radius']:.2f}, "
                  f"Irregular: {loop_info['is_irregular']}")
        else:
            print(f"      Skipping small opening (area: {loop_info['area']:.2f})")
    
    if not loop_infos:
        print("  No significant openings found")
        return mesh, []
    
    # Process mesh by cutting irregular boundaries
    processed_mesh = mesh.copy()
    for loop_info in loop_infos:
        if loop_info['is_irregular']:
            processed_mesh = cut_irregular_boundary(processed_mesh, loop_info, cut_distance)
    
    # Re-detect boundaries after cutting
    print("  Re-detecting boundaries after cutting...")
    new_boundary_loops = detect_boundary_loops(processed_mesh)
    
    # Create caps for the clean openings
    final_vertices = processed_mesh.vertices.copy()
    final_faces = processed_mesh.faces.copy()
    
    cap_infos = []
    
    for i, loop_vertices in enumerate(new_boundary_loops):
        # Re-analyze the cleaned loop
        clean_loop_info = analyze_loop_geometry(processed_mesh, loop_vertices)
        clean_loop_info['id'] = f'opening_{i+1}'
        
        if clean_loop_info['area'] >= min_area:
            print(f"    Creating cap for opening {i+1}...")
            
            # Create planar cap
            cap_vertices, cap_faces = create_planar_cap(clean_loop_info, processed_mesh.scale / 100)
            
            # Adjust face indices for the combined mesh
            cap_faces_adjusted = cap_faces + len(final_vertices)
            
            # Add cap to mesh
            final_vertices = np.vstack([final_vertices, cap_vertices])
            final_faces = np.vstack([final_faces, cap_faces_adjusted])
            
            # Store cap information
            cap_info = {
                'opening_id': clean_loop_info['id'],
                'original_area': clean_loop_info['area'],
                'cap_center': clean_loop_info['centroid'],
                'cap_normal': clean_loop_info['normal'],
                'cap_radius': clean_loop_info['avg_radius'],
                'cap_vertices': list(range(len(final_vertices) - len(cap_vertices), len(final_vertices))),
                'is_inlet_candidate': True,  # Will be determined later based on flow analysis
                'cross_sectional_area': np.pi * clean_loop_info['avg_radius']**2  # Circular approximation
            }
            cap_infos.append(cap_info)
            
            print(f"      Cap created: area={cap_info['cross_sectional_area']:.2f}, "
                  f"radius={cap_info['cap_radius']:.2f}")
    
    # Create final mesh
    final_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
    
    # Clean up mesh
    final_mesh.remove_duplicate_faces()
    final_mesh.remove_unreferenced_vertices()
    
    # Store cap information in metadata
    final_mesh.metadata['caps'] = cap_infos
    
    print(f"  Final mesh: {len(final_mesh.vertices)} vertices, {len(final_mesh.faces)} faces")
    print(f"  Created {len(cap_infos)} caps")
    
    return final_mesh, cap_infos


def determine_inlet_outlet(cap_infos: List[Dict], aneurysm_location: np.ndarray) -> Dict:
    """
    Determine inlet and outlet based on distance from aneurysm and cap properties.
    
    Parameters:
    -----------
    cap_infos : List[Dict]
        List of cap information
    aneurysm_location : np.ndarray
        3D coordinates of aneurysm location
    
    Returns:
    --------
    Dict : Inlet/outlet assignments and flow properties
    """
    if len(cap_infos) < 2:
        print("    Warning: Need at least 2 openings for flow analysis")
        return {}
    
    # Calculate distances from aneurysm to each cap
    distances = []
    for cap in cap_infos:
        distance = np.linalg.norm(cap['cap_center'] - aneurysm_location)
        distances.append(distance)
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
    print(f"      Inlet: {inlet_cap['opening_id']} (area: {inlet_cap['cross_sectional_area']:.2f})")
    print(f"      Outlet: {outlet_cap['opening_id']} (area: {outlet_cap['cross_sectional_area']:.2f})")
    print(f"      Side branches: {len(flow_assignment['side_branches'])}")
    
    return flow_assignment


def calculate_boundary_conditions(flow_assignment: Dict,
                                flow_rate: float = 5.0,  # ml/s
                                outlet_pressure: float = 80.0,  # mmHg
                                blood_density: float = 1060.0,  # kg/m³
                                blood_viscosity: float = 0.004) -> Dict:  # Pa·s
    """
    Calculate accurate boundary conditions based on cross-sectional areas.
    """
    if not flow_assignment:
        return {}
    
    inlet = flow_assignment['inlet']
    outlet = flow_assignment['outlet']
    
    # Convert units
    flow_rate_m3s = flow_rate * 1e-6  # ml/s to m³/s
    outlet_pressure_pa = outlet_pressure * 133.322  # mmHg to Pa
    
    # Calculate inlet velocity based on actual cross-sectional area
    inlet_area_m2 = inlet['cross_sectional_area'] * 1e-6  # mm² to m²
    inlet_velocity = flow_rate_m3s / inlet_area_m2  # m/s
    
    # Calculate hydraulic diameter and Reynolds number
    inlet_diameter_m = 2 * np.sqrt(inlet_area_m2 / np.pi)  # Equivalent diameter
    reynolds_number = (blood_density * inlet_velocity * inlet_diameter_m) / blood_viscosity
    
    boundary_conditions = {
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
            'velocity_direction': -inlet['cap_normal'],  # Into vessel
            'hydraulic_diameter_m': inlet_diameter_m,
            'center_coordinates': inlet['cap_center'].tolist(),
            'cap_vertices': inlet['cap_vertices']
        },
        'outlet_conditions': {
            'opening_id': outlet['opening_id'],
            'type': 'pressure_outlet',
            'pressure_pa': outlet_pressure_pa,
            'pressure_mmhg': outlet_pressure,
            'cross_sectional_area_mm2': outlet['cross_sectional_area'],
            'cross_sectional_area_m2': outlet['cross_sectional_area'] * 1e-6,
            'center_coordinates': outlet['cap_center'].tolist(),
            'cap_vertices': outlet['cap_vertices']
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
                'center_coordinates': branch['cap_center'].tolist(),
                'cap_vertices': branch['cap_vertices']
            })
        boundary_conditions['side_branch_conditions'] = side_branch_conditions
    
    return boundary_conditions


def export_analysis_files(mesh: trimesh.Trimesh,
                         boundary_conditions: Dict,
                         output_base_path: str,
                         formats: List[str] = ['stl', 'ply', 'obj']) -> Dict:
    """
    Export mesh and analysis files.
    """
    output_dir = os.path.dirname(output_base_path)
    base_name = os.path.basename(output_base_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    export_files = {}
    
    # Export mesh in multiple formats
    for fmt in formats:
        mesh_file = os.path.join(output_dir, f"{base_name}_capped_clean.{fmt}")
        mesh.export(mesh_file)
        export_files[f'mesh_{fmt}'] = mesh_file
    
    # Export boundary conditions
    bc_file = os.path.join(output_dir, f"{base_name}_boundary_conditions.json")
    with open(bc_file, 'w') as f:
        json.dump(boundary_conditions, f, indent=2, default=str)
    export_files['boundary_conditions'] = bc_file
    
    # Export analysis summary
    analysis_summary = {
        'mesh_info': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'surface_area_mm2': mesh.area,
            'is_watertight': mesh.is_watertight,
            'bounding_box_mm': mesh.bounds.tolist(),
            'scale_mm': mesh.scale
        },
        'flow_analysis': boundary_conditions,
        'cfd_recommendations': {
            'software': ['ANSYS Fluent', 'OpenFOAM', 'STAR-CCM+', 'COMSOL Multiphysics'],
            'mesh_size_mm': mesh.scale / 50,  # Recommended element size
            'time_step_s': 1e-4,  # Recommended time step
            'solver_settings': {
                'pressure_velocity_coupling': 'SIMPLE or PISO',
                'turbulence_model': 'k-omega SST' if boundary_conditions.get('flow_parameters', {}).get('reynolds_number', 0) > 2300 else 'Laminar',
                'wall_treatment': 'Enhanced wall treatment'
            }
        }
    }
    
    summary_file = os.path.join(output_dir, f"{base_name}_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    export_files['analysis_summary'] = summary_file
    
    return export_files


def process_single_vessel(args: Tuple) -> Dict:
    """
    Process a single vessel through the complete capping pipeline.
    """
    vessel_path, aneurysm_data, output_dir, processing_params = args
    
    patient_id = os.path.basename(vessel_path).replace('_aneurysm_1_vessel.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'files': None,
        'analysis': None
    }
    
    try:
        print(f"\nProcessing {patient_id}...")
        
        # Load vessel mesh
        mesh = trimesh.load(vessel_path)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Process vessel openings
        capped_mesh, cap_infos = process_vessel_openings(
            mesh,
            cut_distance=processing_params.get('cut_distance', 5.0),
            min_area=processing_params.get('min_area', 1.0)
        )
        
        if not cap_infos:
            result['error'] = "No valid openings found after processing"
            return result
        
        # Determine inlet/outlet assignment
        flow_assignment = determine_inlet_outlet(cap_infos, aneurysm_location)
        
        if not flow_assignment:
            result['error'] = "Could not determine inlet/outlet assignment"
            return result
        
        # Calculate boundary conditions
        boundary_conditions = calculate_boundary_conditions(
            flow_assignment,
            flow_rate=processing_params.get('flow_rate', 5.0),
            outlet_pressure=processing_params.get('outlet_pressure', 80.0),
            blood_density=processing_params.get('blood_density', 1060.0),
            blood_viscosity=processing_params.get('blood_viscosity', 0.004)
        )
        
        # Export files
        output_base = os.path.join(output_dir, patient_id)
        export_files = export_analysis_files(
            capped_mesh,
            boundary_conditions,
            output_base,
            formats=processing_params.get('formats', ['stl', 'ply', 'obj'])
        )
        
        result['success'] = True
        result['files'] = export_files
        result['analysis'] = {
            'openings_detected': len(cap_infos),
            'flow_assignment': flow_assignment,
            'inlet_area_mm2': flow_assignment['inlet']['cross_sectional_area'],
            'outlet_area_mm2': flow_assignment['outlet']['cross_sectional_area'],
            'reynolds_number': boundary_conditions['flow_parameters']['reynolds_number'],
            'mesh_vertices': len(capped_mesh.vertices),
            'mesh_faces': len(capped_mesh.faces)
        }
        
        print(f"  ✓ {patient_id}: {len(cap_infos)} openings processed")
        print(f"    Inlet area: {result['analysis']['inlet_area_mm2']:.2f} mm²")
        print(f"    Outlet area: {result['analysis']['outlet_area_mm2']:.2f} mm²")
        print(f"    Reynolds: {result['analysis']['reynolds_number']:.1f}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description='Advanced Vessel Capping Pipeline')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geodesic_large'),
                       help='Directory containing vessel STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../processing/all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/capped_vessels_clean'),
                       help='Output directory for processed vessels')
    
    parser.add_argument('--cut-distance', type=float, default=5.0,
                       help='Distance to cut inward from irregular boundaries (mm)')
    
    parser.add_argument('--min-area', type=float, default=1.0,
                       help='Minimum area to consider as vessel opening (mm²)')
    
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
        'cut_distance': args.cut_distance,
        'min_area': args.min_area,
        'flow_rate': args.flow_rate,
        'outlet_pressure': args.outlet_pressure,
        'blood_density': 1060.0,
        'blood_viscosity': 0.004,
        'formats': ['stl', 'ply', 'obj']
    }
    
    print(f"\nProcessing Parameters:")
    print(f"  Cut distance: {args.cut_distance} mm")
    print(f"  Min opening area: {args.min_area} mm²")
    print(f"  Flow rate: {args.flow_rate} ml/s")
    print(f"  Outlet pressure: {args.outlet_pressure} mmHg")
    
    # Prepare processing arguments
    process_args = [(vessel_file, patient_data, args.output_dir, processing_params) 
                   for vessel_file, patient_data in vessel_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        results = []
        for process_arg in tqdm(process_args, desc="Processing vessels"):
            result = process_single_vessel(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel, process_args),
                               total=len(process_args), desc="Processing vessels"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Advanced Vessel Capping Pipeline Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_openings = np.mean([r['analysis']['openings_detected'] for r in successful])
        avg_inlet_area = np.mean([r['analysis']['inlet_area_mm2'] for r in successful])
        avg_outlet_area = np.mean([r['analysis']['outlet_area_mm2'] for r in successful])
        avg_reynolds = np.mean([r['analysis']['reynolds_number'] for r in successful])
        
        print(f"\nAnalysis Summary:")
        print(f"  Average openings per vessel: {avg_openings:.1f}")
        print(f"  Average inlet area: {avg_inlet_area:.1f} mm²")
        print(f"  Average outlet area: {avg_outlet_area:.1f} mm²")
        print(f"  Average Reynolds number: {avg_reynolds:.1f}")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'processing_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 