#!/usr/bin/env python3
"""
Stress Analysis Preprocessing Pipeline for Blood Vessels

This script prepares extracted vessel meshes for hemodynamic stress analysis by:
1. Detecting inlet/outlet boundaries
2. Creating flat caps on openings
3. Generating boundary conditions for CFD/FEA
4. Exporting in analysis-ready formats
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
import networkx as nx
from sklearn.decomposition import PCA


def detect_vessel_openings(mesh: trimesh.Trimesh, 
                          min_opening_area: float = 5.0,
                          curvature_threshold: float = 0.1) -> List[Dict]:
    """
    Detect vessel inlet/outlet openings based on boundary edges and local geometry.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Vessel mesh
    min_opening_area : float
        Minimum area to consider as vessel opening
    curvature_threshold : float
        Threshold for identifying flat regions
    
    Returns:
    --------
    List[Dict] : List of detected openings with geometric properties
    """
    print("    Detecting vessel openings...")
    
    # Find boundary edges (edges that belong to only one face)
    edges = mesh.edges_unique
    edge_face_count = np.bincount(mesh.edges.flatten(), minlength=len(mesh.vertices))
    
    # Get boundary vertices (vertices with fewer face connections)
    boundary_vertices = set()
    for edge in edges:
        v1, v2 = edge
        if edge_face_count[v1] < 6 or edge_face_count[v2] < 6:  # Threshold for boundary detection
            boundary_vertices.add(v1)
            boundary_vertices.add(v2)
    
    if len(boundary_vertices) < 3:
        print("    Warning: No boundary vertices found")
        return []
    
    # Cluster boundary vertices into separate openings
    boundary_coords = mesh.vertices[list(boundary_vertices)]
    
    # Use connected components on boundary vertices
    boundary_list = list(boundary_vertices)
    boundary_graph = nx.Graph()
    boundary_graph.add_nodes_from(range(len(boundary_list)))
    
    # Connect boundary vertices that are close to each other
    distance_threshold = np.percentile(mesh.scale, 10)  # 10% of mesh scale
    for i in range(len(boundary_list)):
        for j in range(i + 1, len(boundary_list)):
            v1, v2 = boundary_list[i], boundary_list[j]
            dist = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
            if dist < distance_threshold:
                boundary_graph.add_edge(i, j)
    
    # Find connected components (separate openings)
    opening_clusters = list(nx.connected_components(boundary_graph))
    
    openings = []
    for i, cluster in enumerate(opening_clusters):
        if len(cluster) < 3:  # Need at least 3 vertices for an opening
            continue
            
        cluster_vertices = [boundary_list[idx] for idx in cluster]
        cluster_coords = mesh.vertices[cluster_vertices]
        
        # Compute opening properties
        centroid = np.mean(cluster_coords, axis=0)
        
        # Fit plane to opening using PCA
        pca = PCA(n_components=3)
        pca.fit(cluster_coords - centroid)
        normal = pca.components_[2]  # Normal is the component with smallest variance
        
        # Compute approximate area (convex hull area)
        try:
            from scipy.spatial import ConvexHull
            hull_2d = ConvexHull(cluster_coords[:, :2])  # Project to 2D for area computation
            area = hull_2d.volume  # In 2D, volume is area
        except:
            # Fallback: estimate area as circle
            max_dist = np.max(cdist([centroid], cluster_coords)[0])
            area = np.pi * max_dist**2
        
        if area >= min_opening_area:
            opening = {
                'id': f'opening_{i+1}',
                'vertices': cluster_vertices,
                'centroid': centroid,
                'normal': normal,
                'area': area,
                'diameter': 2 * np.sqrt(area / np.pi),  # Equivalent circular diameter
                'vertex_count': len(cluster_vertices)
            }
            openings.append(opening)
    
    print(f"    Found {len(openings)} vessel openings")
    for opening in openings:
        print(f"      - {opening['id']}: {opening['vertex_count']} vertices, "
              f"area={opening['area']:.2f}, diameter={opening['diameter']:.2f}")
    
    return openings


def create_flat_cap(mesh: trimesh.Trimesh, 
                   opening: Dict,
                   extension_distance: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a flat circular cap for a vessel opening.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original vessel mesh
    opening : Dict
        Opening information from detect_vessel_openings
    extension_distance : float
        Distance to extend the vessel before capping
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : Cap vertices and faces
    """
    centroid = opening['centroid']
    normal = opening['normal']
    diameter = opening['diameter']
    
    # Extend the vessel slightly before capping
    extended_centroid = centroid + normal * extension_distance
    
    # Create circular cap
    radius = diameter / 2
    n_points = max(16, int(diameter * 2))  # More points for larger openings
    
    # Create circle in the plane perpendicular to normal
    u = np.array([1, 0, 0])
    if abs(np.dot(normal, u)) > 0.9:  # If normal is parallel to x-axis
        u = np.array([0, 1, 0])
    
    # Create orthonormal basis in the cap plane
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Generate cap vertices
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    cap_vertices = []
    
    # Center vertex
    cap_vertices.append(extended_centroid)
    
    # Circle vertices
    for angle in angles:
        point = extended_centroid + radius * (np.cos(angle) * u + np.sin(angle) * v)
        cap_vertices.append(point)
    
    cap_vertices = np.array(cap_vertices)
    
    # Create triangular faces (fan triangulation)
    cap_faces = []
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Triangle: center, current point, next point
        cap_faces.append([0, i + 1, next_i + 1])
    
    cap_faces = np.array(cap_faces)
    
    return cap_vertices, cap_faces


def cap_vessel_openings(mesh: trimesh.Trimesh, 
                       openings: List[Dict],
                       extension_distance: float = 2.0) -> trimesh.Trimesh:
    """
    Add flat caps to all vessel openings.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Original vessel mesh
    openings : List[Dict]
        List of openings from detect_vessel_openings
    extension_distance : float
        Distance to extend vessel before capping
    
    Returns:
    --------
    trimesh.Trimesh : Mesh with added caps
    """
    print(f"    Adding flat caps to {len(openings)} openings...")
    
    # Start with original mesh
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    cap_info = []
    
    for opening in openings:
        # Create cap
        cap_vertices, cap_faces = create_flat_cap(mesh, opening, extension_distance)
        
        # Adjust face indices to account for existing vertices
        cap_faces_adjusted = cap_faces + len(vertices)
        
        # Add to mesh
        vertices = np.vstack([vertices, cap_vertices])
        faces = np.vstack([faces, cap_faces_adjusted])
        
        # Store cap information for boundary conditions
        cap_info.append({
            'opening_id': opening['id'],
            'cap_center_vertex': len(vertices) - len(cap_vertices),  # Index of cap center
            'cap_vertices': list(range(len(vertices) - len(cap_vertices), len(vertices))),
            'normal': opening['normal'],
            'area': opening['area'],
            'diameter': opening['diameter']
        })
    
    # Create new mesh
    capped_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Clean up mesh
    capped_mesh.remove_duplicate_faces()
    capped_mesh.remove_unreferenced_vertices()
    
    # Store cap information as mesh metadata
    capped_mesh.metadata['caps'] = cap_info
    
    print(f"    Capped mesh: {len(capped_mesh.vertices)} vertices, {len(capped_mesh.faces)} faces")
    
    return capped_mesh


def generate_boundary_conditions(mesh: trimesh.Trimesh, 
                                aneurysm_location: np.ndarray,
                                flow_rate: float = 5.0,  # ml/s
                                pressure_outlet: float = 80.0,  # mmHg
                                blood_viscosity: float = 0.004,  # Pa·s
                                blood_density: float = 1060.0) -> Dict: 
    """
    Generate boundary conditions for hemodynamic analysis.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Capped vessel mesh
    aneurysm_location : np.ndarray
        Location of aneurysm center
    flow_rate : float
        Inlet flow rate in ml/s
    pressure_outlet : float
        Outlet pressure in mmHg
    blood_viscosity : float
        Dynamic viscosity of blood in Pa·s
    blood_density : float
        Density of blood in kg/m³
    
    Returns:
    --------
    Dict : Boundary condition specifications
    """
    print("    Generating boundary conditions...")
    
    if 'caps' not in mesh.metadata:
        print("    Warning: No cap information found in mesh metadata")
        return {}
    
    caps = mesh.metadata['caps']
    
    if len(caps) < 2:
        print(f"    Warning: Need at least 2 openings for flow analysis, found {len(caps)}")
        return {}
    
    # Determine inlet and outlet based on distance from aneurysm
    cap_distances = []
    for cap in caps:
        cap_center_idx = cap['cap_center_vertex']
        cap_center = mesh.vertices[cap_center_idx]
        distance = np.linalg.norm(cap_center - aneurysm_location)
        cap_distances.append(distance)
    
    # Inlet is usually the opening furthest from aneurysm (upstream)
    # Outlet is closest to aneurysm (downstream)
    inlet_idx = np.argmax(cap_distances)
    outlet_idx = np.argmin(cap_distances)
    
    inlet_cap = caps[inlet_idx]
    outlet_cap = caps[outlet_idx]
    
    # Convert units
    flow_rate_m3s = flow_rate * 1e-6  # ml/s to m³/s
    pressure_outlet_pa = pressure_outlet * 133.322  # mmHg to Pa
    
    # Calculate inlet velocity
    inlet_area_m2 = inlet_cap['area'] * 1e-6  # mm² to m²
    inlet_velocity = flow_rate_m3s / inlet_area_m2  # m/s
    
    # Calculate Reynolds number
    inlet_diameter_m = inlet_cap['diameter'] * 1e-3  # mm to m
    reynolds_number = (blood_density * inlet_velocity * inlet_diameter_m) / blood_viscosity
    
    boundary_conditions = {
        'fluid_properties': {
            'density': blood_density,  # kg/m³
            'dynamic_viscosity': blood_viscosity,  # Pa·s
            'kinematic_viscosity': blood_viscosity / blood_density,  # m²/s
        },
        'flow_conditions': {
            'flow_rate': flow_rate,  # ml/s
            'flow_rate_si': flow_rate_m3s,  # m³/s
            'reynolds_number': reynolds_number,
            'flow_regime': 'laminar' if reynolds_number < 2300 else 'turbulent'
        },
        'inlet': {
            'opening_id': inlet_cap['opening_id'],
            'type': 'velocity_inlet',
            'velocity_magnitude': inlet_velocity,  # m/s
            'velocity_direction': -inlet_cap['normal'],  # Into the vessel
            'area': inlet_area_m2,  # m²
            'diameter': inlet_diameter_m,  # m
            'cap_vertices': inlet_cap['cap_vertices']
        },
        'outlet': {
            'opening_id': outlet_cap['opening_id'],
            'type': 'pressure_outlet',
            'pressure': pressure_outlet_pa,  # Pa
            'area': outlet_cap['area'] * 1e-6,  # m²
            'diameter': outlet_cap['diameter'] * 1e-3,  # m
            'cap_vertices': outlet_cap['cap_vertices']
        },
        'walls': {
            'type': 'no_slip',
            'description': 'Vessel walls with no-slip condition'
        }
    }
    
    # Add any additional outlets as pressure outlets
    if len(caps) > 2:
        additional_outlets = []
        for i, cap in enumerate(caps):
            if i != inlet_idx and i != outlet_idx:
                additional_outlets.append({
                    'opening_id': cap['opening_id'],
                    'type': 'pressure_outlet',
                    'pressure': pressure_outlet_pa * 0.9,  # Slightly lower pressure
                    'area': cap['area'] * 1e-6,  # m²
                    'diameter': cap['diameter'] * 1e-3,  # m
                    'cap_vertices': cap['cap_vertices']
                })
        
        boundary_conditions['additional_outlets'] = additional_outlets
    
    print(f"    Inlet: {inlet_cap['opening_id']} (velocity = {inlet_velocity:.3f} m/s)")
    print(f"    Outlet: {outlet_cap['opening_id']} (pressure = {pressure_outlet_pa:.0f} Pa)")
    print(f"    Reynolds number: {reynolds_number:.1f} ({boundary_conditions['flow_conditions']['flow_regime']})")
    
    return boundary_conditions


def export_for_analysis(mesh: trimesh.Trimesh,
                       boundary_conditions: Dict,
                       output_path: str,
                       formats: List[str] = ['stl', 'ply', 'obj']) -> Dict:
    """
    Export mesh and boundary conditions for various analysis software.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Capped vessel mesh
    boundary_conditions : Dict
        Boundary condition specifications
    output_path : str
        Base output path (without extension)
    formats : List[str]
        List of export formats
    
    Returns:
    --------
    Dict : Export file paths and information
    """
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    export_info = {
        'mesh_files': {},
        'boundary_condition_file': None,
        'analysis_summary': None
    }
    
    # Export mesh in various formats
    for fmt in formats:
        mesh_file = os.path.join(output_dir, f"{base_name}_capped.{fmt}")
        mesh.export(mesh_file)
        export_info['mesh_files'][fmt] = mesh_file
    
    # Export boundary conditions as JSON
    bc_file = os.path.join(output_dir, f"{base_name}_boundary_conditions.json")
    with open(bc_file, 'w') as f:
        json.dump(boundary_conditions, f, indent=2, default=str)
    export_info['boundary_condition_file'] = bc_file
    
    # Create analysis summary
    summary = {
        'mesh_statistics': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'volume': mesh.volume if mesh.is_volume else None,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight,
            'caps': len(mesh.metadata.get('caps', []))
        },
        'boundary_conditions': boundary_conditions,
        'recommended_analysis': {
            'cfd_software': ['ANSYS Fluent', 'OpenFOAM', 'COMSOL'],
            'fea_software': ['ANSYS Mechanical', 'Abaqus', 'CalculiX'],
            'mesh_size_recommendation': f"Minimum element size: {mesh.scale / 100:.4f} mm",
            'time_step_recommendation': f"Max time step: {0.001 / boundary_conditions['flow_conditions']['reynolds_number'] * 1000:.6f} s" if 'flow_conditions' in boundary_conditions else "N/A"
        }
    }
    
    summary_file = os.path.join(output_dir, f"{base_name}_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    export_info['analysis_summary'] = summary_file
    
    return export_info


def process_single_vessel_for_stress_analysis(args: Tuple) -> Dict:
    """
    Process a single vessel for stress analysis preparation.
    """
    vessel_path, aneurysm_data, output_dir, analysis_params = args
    
    patient_id = os.path.basename(vessel_path).replace('_aneurysm_1_vessel.stl', '')
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'analysis_files': None
    }
    
    try:
        print(f"Processing {patient_id} for stress analysis...")
        
        # Load vessel mesh
        mesh = trimesh.load(vessel_path)
        
        # Get aneurysm location
        aneurysm_location = np.array(aneurysm_data['aneurysms'][0]['mesh_vertex_coords'])
        
        # Detect vessel openings
        openings = detect_vessel_openings(
            mesh,
            min_opening_area=analysis_params.get('min_opening_area', 5.0),
            curvature_threshold=analysis_params.get('curvature_threshold', 0.1)
        )
        
        if len(openings) < 1:
            result['error'] = "No vessel openings detected"
            return result
        
        # Create flat caps
        capped_mesh = cap_vessel_openings(
            mesh, 
            openings, 
            extension_distance=analysis_params.get('extension_distance', 2.0)
        )
        
        # Generate boundary conditions
        boundary_conditions = generate_boundary_conditions(
            capped_mesh,
            aneurysm_location,
            flow_rate=analysis_params.get('flow_rate', 5.0),
            pressure_outlet=analysis_params.get('pressure_outlet', 80.0),
            blood_viscosity=analysis_params.get('blood_viscosity', 0.004),
            blood_density=analysis_params.get('blood_density', 1060.0)
        )
        
        # Export for analysis
        output_base = os.path.join(output_dir, patient_id)
        export_info = export_for_analysis(
            capped_mesh,
            boundary_conditions,
            output_base,
            formats=analysis_params.get('export_formats', ['stl', 'ply', 'obj'])
        )
        
        result['success'] = True
        result['analysis_files'] = export_info
        result['openings_detected'] = len(openings)
        result['mesh_stats'] = {
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'capped_vertices': len(capped_mesh.vertices),
            'capped_faces': len(capped_mesh.faces)
        }
        
        print(f"  ✓ {patient_id}: {len(openings)} openings, "
              f"{len(capped_mesh.vertices)} vertices (capped)")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ERROR {patient_id}: {e}")
    
    return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Prepare vessel meshes for hemodynamic stress analysis'
    )
    parser.add_argument(
        '--vessel-dir',
        default=os.path.expanduser('~/urp/data/uan/aneurysm_vessels_geodesic_large'),
        help='Directory containing vessel STL files'
    )
    parser.add_argument(
        '--aneurysm-json',
        default='../all_patients_aneurysms_for_stl.json',
        help='JSON file with aneurysm coordinates'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.expanduser('~/urp/data/uan/stress_analysis'),
        help='Output directory for analysis-ready files'
    )
    parser.add_argument(
        '--flow-rate',
        type=float,
        default=5.0,
        help='Inlet flow rate in ml/s (default: 5.0)'
    )
    parser.add_argument(
        '--pressure-outlet',
        type=float,
        default=80.0,
        help='Outlet pressure in mmHg (default: 80.0)'
    )
    parser.add_argument(
        '--min-opening-area',
        type=float,
        default=5.0,
        help='Minimum area for vessel opening detection (default: 5.0)'
    )
    parser.add_argument(
        '--extension-distance',
        type=float,
        default=2.0,
        help='Distance to extend vessel before capping (default: 2.0)'
    )
    parser.add_argument(
        '--export-formats',
        nargs='+',
        default=['stl', 'ply', 'obj'],
        help='Export formats (default: stl ply obj)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16)'
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
    
    # Prepare analysis parameters
    analysis_params = {
        'flow_rate': args.flow_rate,
        'pressure_outlet': args.pressure_outlet,
        'min_opening_area': args.min_opening_area,
        'extension_distance': args.extension_distance,
        'export_formats': args.export_formats,
        'blood_viscosity': 0.004,  # Pa·s
        'blood_density': 1060.0    # kg/m³
    }
    
    print(f"\nStress Analysis Parameters:")
    print(f"  - Flow rate: {args.flow_rate} ml/s")
    print(f"  - Outlet pressure: {args.pressure_outlet} mmHg")
    print(f"  - Min opening area: {args.min_opening_area} mm²")
    print(f"  - Extension distance: {args.extension_distance} mm")
    print(f"  - Export formats: {', '.join(args.export_formats)}")
    
    # Prepare processing arguments
    process_args = [(vessel_file, patient_data, args.output_dir, analysis_params) 
                   for vessel_file, patient_data in vessel_files]
    
    # Process vessels
    start_time = time.time()
    if args.workers == 1:
        # Sequential processing
        results = []
        for process_arg in tqdm(process_args, desc="Preparing stress analysis"):
            result = process_single_vessel_for_stress_analysis(process_arg)
            results.append(result)
    else:
        # Parallel processing
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(process_single_vessel_for_stress_analysis, process_args), 
                               total=len(process_args), desc="Preparing stress analysis"))
    
    # Summary
    total_time = time.time() - start_time
    successful_vessels = sum(1 for r in results if r['success'])
    total_openings = sum(r.get('openings_detected', 0) for r in results if r['success'])
    
    print(f"\n" + "="*70)
    print(f"Stress analysis preparation complete in {total_time:.1f} seconds")
    print(f"Successful vessels: {successful_vessels}/{len(results)}")
    print(f"Total openings detected: {total_openings}")
    print(f"Output directory: {args.output_dir}")
    
    # Save results summary
    results_file = os.path.join(args.output_dir, 'stress_analysis_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")
    
    # Create processing summary
    if successful_vessels > 0:
        avg_openings = total_openings / successful_vessels
        print(f"\nSummary Statistics:")
        print(f"  - Average openings per vessel: {avg_openings:.1f}")
        print(f"  - Analysis files generated in: {args.output_dir}")
        print(f"  - Ready for CFD/FEA software import")
    
    return 0


if __name__ == "__main__":
    exit(main()) 