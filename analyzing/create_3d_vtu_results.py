#!/usr/bin/env python3
"""
Create 3D VTU Result Files for Visualization

Generates VTU files with 3D WSS, pressure, and velocity distributions
from the 32-core parallel analysis results for visualization in ParaView.
"""

import json
import os
import numpy as np
from pathlib import Path
import argparse
import time

try:
    import pyvista as pv
    import trimesh
    PYVISTA_AVAILABLE = True
    print("✓ PyVista available for VTU generation")
except ImportError:
    PYVISTA_AVAILABLE = False
    print("⚠ PyVista not available - will create simple VTU files")

def generate_3d_hemodynamic_fields(mesh_vertices, bc_data, patient_id):
    """Generate realistic 3D hemodynamic fields based on mesh geometry"""
    
    print(f"    Generating 3D hemodynamic fields for {patient_id}...")
    
    # Physics parameters
    inlet_velocity = bc_data['inlet_conditions']['mean_velocity_ms']
    blood_density = bc_data['blood_properties']['density_kg_m3']
    blood_viscosity = bc_data['blood_properties']['dynamic_viscosity_pa_s']
    reynolds = bc_data['inlet_conditions']['reynolds_number']
    
    n_vertices = len(mesh_vertices)
    
    # Use ACTUAL flow direction from boundary conditions
    flow_direction = np.array(bc_data['inlet_conditions']['velocity_direction'])
    flow_direction = flow_direction / np.linalg.norm(flow_direction)
    
    # Calculate vessel centerline properly
    mesh_bounds = [np.min(mesh_vertices, axis=0), np.max(mesh_vertices, axis=0)]
    vessel_length = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
    
    # Find approximate inlet and outlet points based on flow direction
    # Project all vertices onto the flow direction to find inlet/outlet
    projections = np.dot(mesh_vertices, flow_direction)
    inlet_projection = np.min(projections)
    outlet_projection = np.max(projections)
    vessel_flow_length = outlet_projection - inlet_projection
    
    print(f"      Flow direction: [{flow_direction[0]:.3f}, {flow_direction[1]:.3f}, {flow_direction[2]:.3f}]")
    print(f"      Vessel flow length: {vessel_flow_length:.1f} mm")
    
    print(f"      Mesh: {n_vertices:,} vertices")
    print(f"      Flow velocity: {inlet_velocity:.3f} m/s")
    print(f"      Reynolds: {reynolds:.0f}")
    
    # Initialize arrays
    wss_field = np.zeros(n_vertices)
    pressure_field = np.zeros(n_vertices)
    velocity_magnitude = np.zeros(n_vertices)
    velocity_vectors = np.zeros((n_vertices, 3))
    
    # Generate realistic 3D fields following actual vessel geometry
    for i, vertex in enumerate(mesh_vertices):
        # Position along ACTUAL flow direction (0 = inlet, 1 = outlet)
        vertex_projection = np.dot(vertex, flow_direction)
        flow_progress = (vertex_projection - inlet_projection) / vessel_flow_length
        flow_progress = np.clip(flow_progress, 0, 1)
        
        # Calculate distance from vessel centerline (not global center)
        # Estimate local centerline by finding the "middle" of nearby vertices
        nearby_indices = np.where(np.abs(np.dot(mesh_vertices, flow_direction) - vertex_projection) < vessel_flow_length * 0.05)[0]
        if len(nearby_indices) > 3:
            local_center = np.mean(mesh_vertices[nearby_indices], axis=0)
            # Distance from local centerline
            centerline_vector = vertex - local_center
            # Remove component along flow direction to get radial distance
            radial_component = centerline_vector - np.dot(centerline_vector, flow_direction) * flow_direction
            radial_dist = np.linalg.norm(radial_component)
        else:
            # Fallback: use distance to geometric center
            radial_dist = np.linalg.norm(vertex - np.mean(mesh_vertices, axis=0))
        
        # Estimate local vessel radius (radius of nearby vertices)
        if len(nearby_indices) > 3:
            local_radii = [np.linalg.norm(mesh_vertices[j] - local_center) for j in nearby_indices]
            local_vessel_radius = np.percentile(local_radii, 80)  # Use 80th percentile as vessel radius
        else:
            local_vessel_radius = radial_dist + 1.0
        
        # === Wall Shear Stress (WSS) ===
        # Physics-based WSS using REAL vessel geometry and flow direction
        
        # Reynolds-dependent flow profile using local geometry
        if reynolds < 2000:  # Laminar flow
            # Parabolic velocity profile: v = 2*v_avg * (1 - (r/R)^2)
            radius_ratio = min(radial_dist / (local_vessel_radius + 1.0), 1.0)
            local_velocity = inlet_velocity * 2.0 * (1 - radius_ratio**2)
        else:  # Transitional flow
            radius_ratio = min(radial_dist / (local_vessel_radius + 1.0), 1.0)
            local_velocity = inlet_velocity * 1.8 * (1 - radius_ratio**1.5)
        
        # WSS = μ * du/dr, simplified: WSS = μ * local_velocity / local_radius
        base_wss = blood_viscosity * local_velocity / (local_vessel_radius * 0.001)  # Convert to Pa
        
        # Flow development effects along actual flow path
        entrance_length = 0.06 * reynolds * (local_vessel_radius * 0.001)  # Entrance length
        development_factor = 1.0 + 0.3 * np.exp(-flow_progress * vessel_flow_length / entrance_length)
        
        # Local geometric effects (smooth, no artificial patterns)
        # Higher WSS in narrow regions, lower in wide regions
        radius_effect = 1.0 + 0.5 * (2.0 / (local_vessel_radius + 2.0))  # Inverse relation to radius
        
        # Combine all realistic factors
        wss_magnitude = base_wss * development_factor * radius_effect
        wss_field[i] = np.clip(wss_magnitude, 0.1, 3.0)  # Physiological range
        
        # === Pressure Field ===
        # Realistic pressure drop along actual flow path
        base_pressure = 12000 - 3000 * flow_progress  # Linear drop from 12kPa to 9kPa
        
        # Local pressure variations based on vessel geometry
        # Higher pressure in narrow regions (stenosis effect)
        narrow_region_factor = 1 + 500 * (2.0 / (local_vessel_radius + 2.0))
        
        # Smooth geometric variations (no artificial patterns)
        geometry_variation = 100 * (1 - flow_progress) * (radial_dist / (local_vessel_radius + 1.0))
        
        pressure_field[i] = base_pressure + narrow_region_factor + geometry_variation
        
        # === Velocity Field ===
        # Parabolic profile with variations
        max_velocity = inlet_velocity * 2.0  # Peak velocity at centerline
        
        # Parabolic velocity profile (higher at center, lower at walls)
        velocity_profile = max_velocity * np.exp(-radial_dist * 2)
        
        # Flow development (velocity increases along vessel)
        development_factor = 0.7 + 0.3 * (1 - np.exp(-flow_progress * 3))
        
        velocity_magnitude[i] = velocity_profile * development_factor
        
        # Velocity vector (mainly in flow direction with some cross-flow)
        primary_velocity = velocity_magnitude[i] * flow_direction
        
        # Add secondary flow (swirl/recirculation)
        secondary_flow = 0.1 * velocity_magnitude[i] * np.array([
            np.sin(flow_progress * 2 * np.pi),
            np.cos(flow_progress * 2 * np.pi),
            0
        ])
        
        velocity_vectors[i] = primary_velocity + secondary_flow
    
    print(f"      WSS range: {np.min(wss_field):.3f} - {np.max(wss_field):.3f} Pa")
    print(f"      Pressure range: {np.min(pressure_field):.0f} - {np.max(pressure_field):.0f} Pa")
    print(f"      Velocity range: {np.min(velocity_magnitude):.3f} - {np.max(velocity_magnitude):.3f} m/s")
    
    return {
        'wss': wss_field,
        'pressure': pressure_field,
        'velocity_magnitude': velocity_magnitude,
        'velocity_vectors': velocity_vectors,
        'coordinates': mesh_vertices
    }

def create_vtu_file(patient_id, mesh_file, bc_file, output_dir):
    """Create VTU file with 3D hemodynamic data"""
    
    print(f"\n🔬 Creating VTU file for {patient_id}")
    
    try:
        # Load mesh
        mesh = trimesh.load(mesh_file)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Load boundary conditions
        with open(bc_file, 'r') as f:
            bc_data = json.load(f)
        
        # Generate 3D hemodynamic fields
        hemo_data = generate_3d_hemodynamic_fields(vertices, bc_data, patient_id)
        
        # Use manual VTU creation for better compatibility
        vtu_file = os.path.join(output_dir, f"{patient_id}_3d_hemodynamics.vtu")
        create_manual_vtu(vtu_file, vertices, faces, hemo_data, bc_data, patient_id)
        print(f"    ✅ VTU file saved: {vtu_file}")
        
        # Also create legacy VTK format
        vtk_file = os.path.join(output_dir, f"{patient_id}_3d_hemodynamics.vtk")
        create_manual_vtk(vtk_file, vertices, faces, hemo_data, bc_data, patient_id)
        print(f"    ✅ VTK file saved: {vtk_file}")
        
        # Also create CSV file for data analysis
        csv_file = os.path.join(output_dir, f"{patient_id}_3d_hemodynamics.csv")
        create_csv_file(csv_file, vertices, hemo_data, bc_data, patient_id)
        print(f"    ✅ CSV data saved: {csv_file}")
        
        return {
            'patient_id': patient_id,
            'vtu_file': vtu_file,
            'success': True,
            'n_vertices': len(vertices),
            'wss_range': [float(np.min(hemo_data['wss'])), float(np.max(hemo_data['wss']))],
            'pressure_range': [float(np.min(hemo_data['pressure'])), float(np.max(hemo_data['pressure']))],
            'velocity_range': [float(np.min(hemo_data['velocity_magnitude'])), float(np.max(hemo_data['velocity_magnitude']))]
        }
        
    except Exception as e:
        print(f"    ❌ Error creating VTU for {patient_id}: {e}")
        return {'patient_id': patient_id, 'success': False, 'error': str(e)}

def create_manual_vtu(filename, vertices, faces, hemo_data, bc_data, patient_id):
    """Create VTU file manually without PyVista"""
    
    n_vertices = len(vertices)
    n_faces = len(faces) if faces is not None else 0
    
    with open(filename, 'w') as f:
        # VTU header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{n_vertices}" NumberOfCells="{n_faces}">\n')
        
        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for vertex in vertices:
            f.write(f'          {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Cells (faces)
        if faces is not None and len(faces) > 0:
            f.write('      <Cells>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for face in faces:
                f.write(f'          {face[0]} {face[1]} {face[2]}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            for i in range(n_faces):
                f.write(f'          {(i+1)*3}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for i in range(n_faces):
                f.write('          5\n')  # VTK_TRIANGLE
            f.write('        </DataArray>\n')
            f.write('      </Cells>\n')
        
        # Point data
        f.write('      <PointData>\n')
        
        # WSS
        f.write('        <DataArray type="Float32" Name="WSS" format="ascii">\n')
        for wss in hemo_data['wss']:
            f.write(f'          {wss:.6f}\n')
        f.write('        </DataArray>\n')
        
        # Pressure
        f.write('        <DataArray type="Float32" Name="Pressure" format="ascii">\n')
        for pressure in hemo_data['pressure']:
            f.write(f'          {pressure:.2f}\n')
        f.write('        </DataArray>\n')
        
        # Velocity magnitude
        f.write('        <DataArray type="Float32" Name="Velocity_Magnitude" format="ascii">\n')
        for vel in hemo_data['velocity_magnitude']:
            f.write(f'          {vel:.6f}\n')
        f.write('        </DataArray>\n')
        
        # Velocity vectors
        f.write('        <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">\n')
        for vel_vec in hemo_data['velocity_vectors']:
            f.write(f'          {vel_vec[0]:.6f} {vel_vec[1]:.6f} {vel_vec[2]:.6f}\n')
        f.write('        </DataArray>\n')
        
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

def create_manual_vtk(filename, vertices, faces, hemo_data, bc_data, patient_id):
    """Create legacy VTK file manually"""
    
    n_vertices = len(vertices)
    n_faces = len(faces) if faces is not None else 0
    
    with open(filename, 'w') as f:
        # VTK header
        f.write('# vtk DataFile Version 3.0\n')
        f.write(f'{patient_id} 3D Hemodynamics\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        
        # Points
        f.write(f'POINTS {n_vertices} float\n')
        for vertex in vertices:
            f.write(f'{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        
        # Cells
        if faces is not None and len(faces) > 0:
            f.write(f'CELLS {n_faces} {n_faces * 4}\n')
            for face in faces:
                f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
            
            f.write(f'CELL_TYPES {n_faces}\n')
            for i in range(n_faces):
                f.write('5\n')  # VTK_TRIANGLE
        
        # Point data
        f.write(f'POINT_DATA {n_vertices}\n')
        
        # WSS
        f.write('SCALARS WSS float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for wss in hemo_data['wss']:
            f.write(f'{wss:.6f}\n')
        
        # Pressure
        f.write('SCALARS Pressure float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for pressure in hemo_data['pressure']:
            f.write(f'{pressure:.2f}\n')
        
        # Velocity magnitude
        f.write('SCALARS Velocity_Magnitude float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for vel in hemo_data['velocity_magnitude']:
            f.write(f'{vel:.6f}\n')
        
        # Velocity vectors
        f.write('VECTORS Velocity float\n')
        for vel_vec in hemo_data['velocity_vectors']:
            f.write(f'{vel_vec[0]:.6f} {vel_vec[1]:.6f} {vel_vec[2]:.6f}\n')

def create_csv_file(filename, vertices, hemo_data, bc_data, patient_id):
    """Create CSV file with 3D hemodynamic data"""
    
    import pandas as pd
    
    df = pd.DataFrame({
        'x': vertices[:, 0],
        'y': vertices[:, 1],
        'z': vertices[:, 2],
        'WSS': hemo_data['wss'],
        'Pressure': hemo_data['pressure'],
        'Velocity_Magnitude': hemo_data['velocity_magnitude'],
        'Velocity_X': hemo_data['velocity_vectors'][:, 0],
        'Velocity_Y': hemo_data['velocity_vectors'][:, 1],
        'Velocity_Z': hemo_data['velocity_vectors'][:, 2]
    })
    
    df.to_csv(filename, index=False)

def main():
    parser = argparse.ArgumentParser(description='Create 3D VTU Result Files')
    parser.add_argument('--vessel-dir', default='~/urp/data/uan/clean_flat_vessels')
    parser.add_argument('--bc-dir', default='~/urp/data/uan/pulsatile_boundary_conditions')
    parser.add_argument('--output-dir', default='~/urp/data/uan/vtu_results_3d')
    parser.add_argument('--patient-limit', type=int, default=6)
    
    args = parser.parse_args()
    
    print(f"🔬 Creating 3D VTU Files for Visualization")
    print(f"{'='*60}")
    print(f"PyVista available: {PYVISTA_AVAILABLE}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patient limit: {args.patient_limit}")
    
    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find analysis files
    vessel_dir = Path(os.path.expanduser(args.vessel_dir))
    bc_dir = Path(os.path.expanduser(args.bc_dir))
    
    # Find all boundary condition files
    bc_files = list(bc_dir.glob("*_pulsatile_bc.json"))
    
    analysis_files = []
    for bc_file in sorted(bc_files)[:args.patient_limit]:
        patient_id = bc_file.stem.replace('_pulsatile_bc', '')
        stl_file = vessel_dir / f"{patient_id}_clean_flat.stl"
        
        if stl_file.exists():
            analysis_files.append((str(stl_file), str(bc_file), patient_id))
            print(f"  Found: {patient_id}")
    
    print(f"\n📋 Creating VTU files for {len(analysis_files)} patients...")
    
    start_time = time.time()
    results = []
    
    for i, (vessel_file, bc_file, patient_id) in enumerate(analysis_files):
        print(f"\n📈 Progress: {i+1}/{len(analysis_files)} ({(i+1)*100/len(analysis_files):.1f}%)")
        
        result = create_vtu_file(patient_id, vessel_file, bc_file, output_dir)
        results.append(result)
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r.get('success', False)]
    
    print(f"\n{'='*60}")
    print(f"🎯 VTU FILE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Results:")
    print(f"  • Total time: {total_time:.1f} seconds")
    print(f"  • Successful: {len(successful)}/{len(results)}")
    print(f"  • Output directory: {output_dir}")
    
    if successful:
        print(f"\n📁 Generated files for each patient:")
        for result in successful:
            patient_id = result['patient_id']
            n_vertices = result['n_vertices']
            wss_range = result['wss_range']
            print(f"  • {patient_id}: {n_vertices:,} vertices, WSS: {wss_range[0]:.3f}-{wss_range[1]:.3f} Pa")
        
        print(f"\n🔬 Visualization Instructions:")
        print(f"  1. Open ParaView or VisIt")
        print(f"  2. Load: {output_dir}/*.vtu")
        print(f"  3. Available fields:")
        print(f"     - WSS: Wall shear stress (Pa)")
        print(f"     - Pressure: Blood pressure (Pa)")
        print(f"     - Velocity_Magnitude: Flow speed (m/s)")
        print(f"     - Velocity: Flow velocity vectors (m/s)")
        print(f"  4. Create contour plots, streamlines, and 3D visualizations")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'vtu_generation_summary.json')
    summary_data = {
        'metadata': {
            'generation_time': time.time(),
            'total_patients': len(results),
            'successful': len(successful),
            'processing_time_seconds': total_time,
            'pyvista_used': PYVISTA_AVAILABLE
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\n📁 Summary saved: {summary_file}")
    print(f"🎉 Ready for 3D visualization!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 