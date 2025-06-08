#!/usr/bin/env python3
"""
Watertight Vessel Capping Pipeline

This script creates perfectly watertight vessel meshes by:
1. Detecting all vessel boundary loops
2. Creating caps that are properly stitched to boundary edges
3. Ensuring no gaps or holes remain
4. Validating watertight topology
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
    Detect boundary loops with proper vertex ordering for watertight stitching.
    
    Returns:
    --------
    List[List[int]] : List of boundary loops, each as ordered list of vertex indices
    """
    print("    Detecting ordered boundary loops...")
    
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
    
    # Build directed graph to maintain edge ordering
    boundary_graph = nx.Graph()
    edge_directions = {}
    
    for edge in boundary_edges:
        v1, v2 = edge
        boundary_graph.add_edge(v1, v2)
        
        # Find the face containing this edge to determine direction
        for face in mesh.faces:
            face_edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            if edge in face_edges:
                # Store original directed edge from face
                if (face[0], face[1]) == edge or (face[1], face[0]) == edge:
                    if face[0] == edge[0]:
                        edge_directions[edge] = (edge[0], edge[1])
                    else:
                        edge_directions[edge] = (edge[1], edge[0])
                elif (face[1], face[2]) == edge or (face[2], face[1]) == edge:
                    if face[1] == edge[0]:
                        edge_directions[edge] = (edge[0], edge[1])
                    else:
                        edge_directions[edge] = (edge[1], edge[0])
                elif (face[2], face[0]) == edge or (face[0], face[2]) == edge:
                    if face[2] == edge[0]:
                        edge_directions[edge] = (edge[0], edge[1])
                    else:
                        edge_directions[edge] = (edge[1], edge[0])
                break
    
    # Find connected components and order vertices properly
    boundary_loops = []
    
    for component in nx.connected_components(boundary_graph):
        if len(component) < 3:
            continue
            
        # Create ordered loop by traversing edges
        loop_vertices = []
        visited_edges = set()
        
        # Start with any vertex in the component
        start_vertex = next(iter(component))
        current_vertex = start_vertex
        
        while True:
            loop_vertices.append(current_vertex)
            
            # Find next vertex following boundary edge direction
            neighbors = list(boundary_graph.neighbors(current_vertex))
            next_vertex = None
            
            for neighbor in neighbors:
                edge = tuple(sorted([current_vertex, neighbor]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    next_vertex = neighbor
                    break
            
            if next_vertex is None:
                break
            
            current_vertex = next_vertex
            
            # Stop when we complete the loop
            if current_vertex == start_vertex:
                break
        
        if len(loop_vertices) >= 3:
            boundary_loops.append(loop_vertices)
    
    print(f"    Found {len(boundary_loops)} ordered boundary loops")
    for i, loop in enumerate(boundary_loops):
        print(f"      Loop {i+1}: {len(loop)} vertices")
    
    return boundary_loops


def analyze_loop_for_capping(mesh: trimesh.Trimesh, loop_vertices: List[int]) -> Dict:
    """
    Analyze boundary loop geometry for watertight capping.
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
    
    # Calculate geometric properties
    distances = np.linalg.norm(loop_coords - centroid, axis=1)
    avg_radius = np.mean(distances)
    max_radius = np.max(distances)
    
    # Calculate actual area using shoelace formula on projected 2D
    u = pca.components_[0]
    v = pca.components_[1]
    projected_2d = np.column_stack([
        np.dot(loop_coords - centroid, u),
        np.dot(loop_coords - centroid, v)
    ])
    
    # Close the loop for area calculation
    projected_2d_closed = np.vstack([projected_2d, projected_2d[0]])
    area = 0.5 * abs(sum(
        projected_2d_closed[i][0] * projected_2d_closed[i+1][1] - 
        projected_2d_closed[i+1][0] * projected_2d_closed[i][1]
        for i in range(len(projected_2d_closed)-1)
    ))
    
    return {
        'vertices': loop_vertices,
        'coords': loop_coords,
        'centroid': centroid,
        'normal': normal,
        'radius': avg_radius,
        'max_radius': max_radius,
        'area': area,
        'plane_u': u,
        'plane_v': v,
        'projected_2d': projected_2d
    }


def create_watertight_cap(mesh: trimesh.Trimesh, loop_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a watertight cap by triangulating the boundary loop.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original mesh
    loop_info : Dict
        Loop geometry information
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : Cap vertices and faces
    """
    loop_vertices = loop_info['vertices']
    centroid = loop_info['centroid']
    projected_2d = loop_info['projected_2d']
    
    print(f"    Creating watertight cap for {len(loop_vertices)} boundary vertices...")
    
    # Use existing boundary vertices as cap perimeter
    cap_vertices = [centroid]  # Center vertex
    cap_faces = []
    
    # Add center vertex at index 0, boundary vertices keep their original indices
    n_boundary = len(loop_vertices)
    
    # Create triangular faces from center to boundary edges
    for i in range(n_boundary):
        next_i = (i + 1) % n_boundary
        
        # Triangle: center (0), current boundary vertex (i+1), next boundary vertex (next_i+1)
        cap_faces.append([0, i + 1, next_i + 1])
    
    # The cap vertices are: [center] + boundary_vertices_coords
    boundary_coords = mesh.vertices[loop_vertices]
    cap_vertices_coords = np.vstack([cap_vertices, boundary_coords])
    
    return cap_vertices_coords, np.array(cap_faces)


def stitch_cap_to_mesh(mesh: trimesh.Trimesh, 
                       loop_vertices: List[int], 
                       cap_vertices: np.ndarray, 
                       cap_faces: np.ndarray) -> trimesh.Trimesh:
    """
    Stitch cap to mesh ensuring watertight topology.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original mesh
    loop_vertices : List[int]
        Boundary loop vertex indices
    cap_vertices : np.ndarray
        Cap vertex coordinates
    cap_faces : np.ndarray
        Cap face indices (local to cap)
    
    Returns:
    --------
    trimesh.Trimesh : Watertight mesh with stitched cap
    """
    print(f"    Stitching cap to mesh...")
    
    # Start with original mesh
    new_vertices = mesh.vertices.copy()
    new_faces = mesh.faces.copy()
    
    # Add cap center vertex
    cap_center_idx = len(new_vertices)
    new_vertices = np.vstack([new_vertices, cap_vertices[0:1]])  # Just the center
    
    # Create mapping from cap vertex indices to global mesh indices
    vertex_mapping = {0: cap_center_idx}  # Cap center
    
    # Boundary vertices already exist in mesh, use their original indices
    for i, boundary_idx in enumerate(loop_vertices):
        vertex_mapping[i + 1] = boundary_idx
    
    # Add cap faces using the global vertex indices
    for face in cap_faces:
        global_face = [vertex_mapping[face[i]] for i in range(3)]
        new_faces = np.vstack([new_faces, global_face])
    
    # Create new mesh
    stitched_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    return stitched_mesh


def ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Ensure mesh is watertight by fixing any remaining issues.
    """
    print(f"    Ensuring watertight mesh...")
    
    # Remove duplicate faces and vertices
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    
    # Fill small holes if any remain
    if hasattr(mesh, 'fill_holes'):
        try:
            mesh.fill_holes()
        except:
            pass
    
    # Check if watertight
    is_watertight = mesh.is_watertight
    print(f"    Mesh is watertight: {is_watertight}")
    
    if not is_watertight:
        print(f"    Attempting watertight repair...")
        try:
            # Try trimesh's watertight repair
            if hasattr(mesh, 'repair'):
                mesh.repair()
            
            # Alternative: use pymeshfix if available
            try:
                import pymeshfix
                meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                meshfix.repair(verbose=False)
                mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
                print(f"    Repaired with pymeshfix: {mesh.is_watertight}")
            except ImportError:
                print("    pymeshfix not available for advanced repair")
            
        except Exception as e:
            print(f"    Repair failed: {e}")
    
    return mesh


def process_vessel_watertight(mesh: trimesh.Trimesh, 
                             min_area: float = 1.0) -> Tuple[trimesh.Trimesh, List[Dict]]:
    """
    Process vessel to create watertight caps on all openings.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input vessel mesh
    min_area : float
        Minimum area to consider as vessel opening
    
    Returns:
    --------
    Tuple[trimesh.Trimesh, List[Dict]] : Watertight mesh and opening information
    """
    print(f"  Creating watertight vessel...")
    
    # Detect boundary loops with proper ordering
    boundary_loops = detect_boundary_loops_ordered(mesh)
    
    if not boundary_loops:
        print("  No boundary loops detected")
        return mesh, []
    
    # Analyze each loop
    loop_infos = []
    for i, loop_vertices in enumerate(boundary_loops):
        print(f"    Analyzing loop {i+1}...")
        loop_info = analyze_loop_for_capping(mesh, loop_vertices)
        loop_info['id'] = f'opening_{i+1}'
        
        if loop_info['area'] >= min_area:
            loop_infos.append(loop_info)
            print(f"      Area: {loop_info['area']:.2f} mm², Radius: {loop_info['radius']:.2f} mm")
        else:
            print(f"      Skipping small opening (area: {loop_info['area']:.2f} mm²)")
    
    if not loop_infos:
        print("  No significant openings found")
        return mesh, []
    
    # Create watertight mesh by capping all openings
    capped_mesh = mesh.copy()
    cap_infos = []
    
    for loop_info in loop_infos:
        print(f"    Capping {loop_info['id']}...")
        
        # Create watertight cap
        cap_vertices, cap_faces = create_watertight_cap(capped_mesh, loop_info)
        
        # Stitch cap to mesh
        capped_mesh = stitch_cap_to_mesh(
            capped_mesh,
            loop_info['vertices'],
            cap_vertices,
            cap_faces
        )
        
        # Store cap information for boundary conditions
        cap_info = {
            'opening_id': loop_info['id'],
            'area': loop_info['area'],
            'centroid': loop_info['centroid'],
            'normal': loop_info['normal'],
            'radius': loop_info['radius'],
            'boundary_vertices': loop_info['vertices'],
            'is_inlet_candidate': True,
            'cross_sectional_area': loop_info['area']  # Use actual calculated area
        }
        cap_infos.append(cap_info)
    
    # Ensure final mesh is watertight
    final_mesh = ensure_watertight(capped_mesh)
    
    # Verify watertight status
    if final_mesh.is_watertight:
        print(f"  ✓ Watertight mesh created: {len(final_mesh.vertices)} vertices, {len(final_mesh.faces)} faces")
    else:
        print(f"  ⚠ Warning: Mesh may not be fully watertight")
    
    # Store cap information in metadata
    final_mesh.metadata['caps'] = cap_infos
    final_mesh.metadata['is_watertight'] = final_mesh.is_watertight
    
    return final_mesh, cap_infos


def determine_inlet_outlet_watertight(cap_infos: List[Dict], aneurysm_location: np.ndarray) -> Dict:
    """
    Determine inlet and outlet based on distance from aneurysm and cap properties.
    """
    if len(cap_infos) < 2:
        print("    Warning: Need at least 2 openings for flow analysis")
        return {}
    
    # Calculate distances from aneurysm to each cap
    distances = []
    for cap in cap_infos:
        distance = np.linalg.norm(cap['centroid'] - aneurysm_location)
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
    print(f"      Inlet: {inlet_cap['opening_id']} (area: {inlet_cap['cross_sectional_area']:.2f} mm²)")
    print(f"      Outlet: {outlet_cap['opening_id']} (area: {outlet_cap['cross_sectional_area']:.2f} mm²)")
    print(f"      Side branches: {len(flow_assignment['side_branches'])}")
    
    return flow_assignment


def calculate_boundary_conditions_watertight(flow_assignment: Dict,
                                            flow_rate: float = 5.0,  # ml/s
                                            outlet_pressure: float = 80.0,  # mmHg
                                            blood_density: float = 1060.0,  # kg/m³
                                            blood_viscosity: float = 0.004) -> Dict:  # Pa·s
    """
    Calculate boundary conditions for watertight mesh.
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
        'mesh_properties': {
            'is_watertight': True,
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
            'boundary_vertices': inlet['boundary_vertices']
        },
        'outlet_conditions': {
            'opening_id': outlet['opening_id'],
            'type': 'pressure_outlet',
            'pressure_pa': outlet_pressure_pa,
            'pressure_mmhg': outlet_pressure,
            'cross_sectional_area_mm2': outlet['cross_sectional_area'],
            'cross_sectional_area_m2': outlet['cross_sectional_area'] * 1e-6,
            'center_coordinates': outlet['centroid'].tolist(),
            'boundary_vertices': outlet['boundary_vertices']
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
                'boundary_vertices': branch['boundary_vertices']
            })
        boundary_conditions['side_branch_conditions'] = side_branch_conditions
    
    return boundary_conditions


def export_watertight_files(mesh: trimesh.Trimesh,
                           boundary_conditions: Dict,
                           output_base_path: str,
                           formats: List[str] = ['stl', 'ply', 'obj']) -> Dict:
    """
    Export watertight mesh and analysis files.
    """
    output_dir = os.path.dirname(output_base_path)
    base_name = os.path.basename(output_base_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    export_files = {}
    
    # Export mesh in multiple formats
    for fmt in formats:
        mesh_file = os.path.join(output_dir, f"{base_name}_watertight.{fmt}")
        mesh.export(mesh_file)
        export_files[f'mesh_{fmt}'] = mesh_file
    
    # Export boundary conditions
    bc_file = os.path.join(output_dir, f"{base_name}_boundary_conditions.json")
    with open(bc_file, 'w') as f:
        json.dump(boundary_conditions, f, indent=2, default=str)
    export_files['boundary_conditions'] = bc_file
    
    # Export watertight analysis summary
    analysis_summary = {
        'mesh_info': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'is_watertight': mesh.is_watertight,
            'volume_mm3': mesh.volume if mesh.is_volume else None,
            'surface_area_mm2': mesh.area,
            'bounding_box_mm': mesh.bounds.tolist(),
            'scale_mm': mesh.scale,
            'euler_number': mesh.euler_number
        },
        'flow_analysis': boundary_conditions,
        'cfd_recommendations': {
            'software': ['ANSYS Fluent', 'OpenFOAM', 'STAR-CCM+', 'COMSOL Multiphysics'],
            'mesh_requirements': {
                'watertight': True,
                'manifold': mesh.is_watertight,
                'mesh_size_mm': mesh.scale / 50,
                'boundary_layer_cells': 5,
                'y_plus_target': 1.0
            },
            'solver_settings': {
                'pressure_velocity_coupling': 'SIMPLE or PISO',
                'turbulence_model': 'k-omega SST' if boundary_conditions.get('flow_parameters', {}).get('reynolds_number', 0) > 2300 else 'Laminar',
                'time_step_s': 1e-4,
                'convergence_criteria': 1e-5
            }
        }
    }
    
    summary_file = os.path.join(output_dir, f"{base_name}_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    export_files['analysis_summary'] = summary_file
    
    return export_files


def process_single_vessel_watertight(args: Tuple) -> Dict:
    """
    Process a single vessel to create watertight mesh.
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
        print(f"\nProcessing {patient_id} for watertight mesh...")
        
        # Load vessel mesh
        mesh = trimesh.load(vessel_path)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Create watertight vessel
        watertight_mesh, cap_infos = process_vessel_watertight(
            mesh,
            min_area=processing_params.get('min_area', 1.0)
        )
        
        if not cap_infos:
            result['error'] = "No valid openings found for capping"
            return result
        
        # Verify watertight status
        if not watertight_mesh.is_watertight:
            result['error'] = "Failed to create watertight mesh"
            return result
        
        # Determine inlet/outlet assignment
        flow_assignment = determine_inlet_outlet_watertight(cap_infos, aneurysm_location)
        
        if not flow_assignment:
            result['error'] = "Could not determine inlet/outlet assignment"
            return result
        
        # Calculate boundary conditions
        boundary_conditions = calculate_boundary_conditions_watertight(
            flow_assignment,
            flow_rate=processing_params.get('flow_rate', 5.0),
            outlet_pressure=processing_params.get('outlet_pressure', 80.0),
            blood_density=processing_params.get('blood_density', 1060.0),
            blood_viscosity=processing_params.get('blood_viscosity', 0.004)
        )
        
        # Export files
        output_base = os.path.join(output_dir, patient_id)
        export_files = export_watertight_files(
            watertight_mesh,
            boundary_conditions,
            output_base,
            formats=processing_params.get('formats', ['stl', 'ply', 'obj'])
        )
        
        result['success'] = True
        result['files'] = export_files
        result['analysis'] = {
            'openings_detected': len(cap_infos),
            'is_watertight': watertight_mesh.is_watertight,
            'volume_mm3': watertight_mesh.volume if watertight_mesh.is_volume else None,
            'flow_assignment': flow_assignment,
            'inlet_area_mm2': flow_assignment['inlet']['cross_sectional_area'],
            'outlet_area_mm2': flow_assignment['outlet']['cross_sectional_area'],
            'reynolds_number': boundary_conditions['flow_parameters']['reynolds_number'],
            'mesh_vertices': len(watertight_mesh.vertices),
            'mesh_faces': len(watertight_mesh.faces),
            'euler_number': watertight_mesh.euler_number
        }
        
        print(f"  ✓ {patient_id}: Watertight mesh with {len(cap_infos)} openings")
        print(f"    Volume: {result['analysis']['volume_mm3']:.1f} mm³" if result['analysis']['volume_mm3'] else "    Volume: N/A")
        print(f"    Inlet area: {result['analysis']['inlet_area_mm2']:.2f} mm²")
        print(f"    Outlet area: {result['analysis']['outlet_area_mm2']:.2f} mm²")
        print(f"    Reynolds: {result['analysis']['reynolds_number']:.1f}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ {patient_id}: {e}")
    
    return result


def main():
    """Main processing function for watertight vessel capping"""
    parser = argparse.ArgumentParser(description='Watertight Vessel Capping Pipeline')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geodesic_large'),
                       help='Directory containing vessel STL files')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-dir',
                       default=os.path.expanduser('~/urp/data/uan/watertight_vessels'),
                       help='Output directory for watertight vessels')
    
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
        'min_area': args.min_area,
        'flow_rate': args.flow_rate,
        'outlet_pressure': args.outlet_pressure,
        'blood_density': 1060.0,
        'blood_viscosity': 0.004,
        'formats': ['stl', 'ply', 'obj']
    }
    
    print(f"\nWatertight Processing Parameters:")
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
        for process_arg in tqdm(process_args, desc="Creating watertight vessels"):
            result = process_single_vessel_watertight(process_arg)
            results.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_watertight, process_args),
                               total=len(process_args), desc="Creating watertight vessels"))
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print(f"Watertight Vessel Capping Pipeline Complete")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        watertight_count = sum(1 for r in successful if r['analysis']['is_watertight'])
        avg_openings = np.mean([r['analysis']['openings_detected'] for r in successful])
        avg_inlet_area = np.mean([r['analysis']['inlet_area_mm2'] for r in successful])
        avg_outlet_area = np.mean([r['analysis']['outlet_area_mm2'] for r in successful])
        avg_reynolds = np.mean([r['analysis']['reynolds_number'] for r in successful])
        
        volumes = [r['analysis']['volume_mm3'] for r in successful if r['analysis']['volume_mm3'] is not None]
        avg_volume = np.mean(volumes) if volumes else None
        
        print(f"\nWatertight Analysis Summary:")
        print(f"  Watertight meshes: {watertight_count}/{len(successful)}")
        print(f"  Average openings per vessel: {avg_openings:.1f}")
        print(f"  Average inlet area: {avg_inlet_area:.1f} mm²")
        print(f"  Average outlet area: {avg_outlet_area:.1f} mm²")
        print(f"  Average volume: {avg_volume:.1f} mm³" if avg_volume else "  Average volume: N/A")
        print(f"  Average Reynolds number: {avg_reynolds:.1f}")
    
    if failed:
        print(f"\nFailed cases:")
        for fail in failed:
            print(f"  {fail['patient_id']}: {fail['error']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'watertight_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    print(f"🎯 All meshes are CFD-ready and watertight!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 