#!/usr/bin/env python3
"""
Direct STL to VTP Converter
Author: Jiwoo Lee

This script directly converts STL files to VTP format using PyVista,
providing immediate visualization files while the full PyFluent pipeline
is being developed.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    print(f"✅ PyVista loaded successfully")
except ImportError as e:
    PYVISTA_AVAILABLE = False
    print(f"❌ PyVista not available: {e}")
    sys.exit(1)

def load_boundary_conditions(bc_file: Path) -> Dict:
    """Load boundary conditions from JSON file."""
    try:
        with open(bc_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {bc_file}: {e}")
        return {}

def add_synthetic_flow_data(mesh: pv.PolyData, bc_data: Dict) -> pv.PolyData:
    """Add synthetic flow data to the mesh based on boundary conditions."""
    
    # Get fluid properties
    fluid_props = bc_data.get('fluid_properties', {})
    inlet_data = bc_data.get('inlet_conditions', {})
    outlet_data = bc_data.get('outlet_conditions', {})
    
    # Basic flow parameters
    inlet_velocity = inlet_data.get('velocity_magnitude_m_s', 0.5)  # m/s
    outlet_pressure = outlet_data.get('pressure_pa', 10665.76)  # Pa
    
    n_points = mesh.n_points
    
    # Generate synthetic pressure field (linear gradient from inlet to outlet)
    # This is a simple approximation for visualization
    z_coords = mesh.points[:, 2]  # Assuming flow is roughly in Z direction
    z_min, z_max = z_coords.min(), z_coords.max()
    
    if z_max > z_min:
        # Normalize Z coordinates
        z_norm = (z_coords - z_min) / (z_max - z_min)
        
        # Create pressure gradient (higher at inlet, lower at outlet)
        pressure = outlet_pressure + (15000 - outlet_pressure) * (1 - z_norm)
    else:
        # Uniform pressure if no Z variation
        pressure = np.full(n_points, outlet_pressure)
    
    # Generate synthetic velocity field
    # Simple approximation: higher velocity in the center, lower near walls
    center = mesh.center
    distances = np.linalg.norm(mesh.points - center, axis=1)
    max_distance = distances.max()
    
    if max_distance > 0:
        # Parabolic velocity profile approximation
        velocity_magnitude = inlet_velocity * (1 - (distances / max_distance) ** 2)
        velocity_magnitude = np.maximum(velocity_magnitude, 0.1)  # Minimum velocity
    else:
        velocity_magnitude = np.full(n_points, inlet_velocity)
    
    # Generate velocity vector field (simplified)
    # Assume flow direction is roughly along the vessel centerline
    velocity_vectors = np.zeros((n_points, 3))
    
    # Simple flow direction (from inlet to outlet, roughly in Z direction)
    flow_direction = np.array([0.1, 0.1, 1.0])  # Slight XY component for realism
    flow_direction = flow_direction / np.linalg.norm(flow_direction)
    
    for i in range(n_points):
        velocity_vectors[i] = velocity_magnitude[i] * flow_direction
    
    # Calculate wall shear stress (simplified approximation)
    # Higher near walls, lower in center
    if max_distance > 0:
        wall_factor = distances / max_distance
        wall_shear_stress = 5.0 * wall_factor  # Approximate WSS in Pa
    else:
        wall_shear_stress = np.full(n_points, 2.5)
    
    # Add all data to mesh
    mesh['pressure'] = pressure
    mesh['velocity_magnitude'] = velocity_magnitude
    mesh['velocity'] = velocity_vectors
    mesh['wall_shear_stress'] = wall_shear_stress
    
    # Add scalar fields for analysis
    mesh['reynolds_number'] = np.full(n_points, bc_data.get('flow_parameters', {}).get('reynolds_number', 400))
    mesh['density'] = np.full(n_points, fluid_props.get('density', 1060.0))
    mesh['viscosity'] = np.full(n_points, fluid_props.get('dynamic_viscosity', 0.004))
    
    return mesh

def convert_stl_to_vtp(stl_file: Path, bc_file: Path, output_dir: Path) -> bool:
    """Convert STL file to VTP with synthetic flow data."""
    
    case_name = stl_file.stem.replace("_clean_flat", "")
    
    try:
        print(f"🔄 Processing {case_name}...")
        
        # Load STL mesh
        mesh = pv.read(str(stl_file))
        print(f"  📁 Loaded STL: {mesh.n_points} points, {mesh.n_cells} cells")
        
        # Load boundary conditions
        bc_data = load_boundary_conditions(bc_file)
        if not bc_data:
            print(f"  ⚠️  No boundary conditions, using defaults")
            bc_data = {}
        
        # Add synthetic flow data
        mesh_with_data = add_synthetic_flow_data(mesh, bc_data)
        print(f"  📊 Added flow data fields")
        
        # Save as VTP
        vtp_file = output_dir / f"{case_name}.vtp"
        mesh_with_data.save(str(vtp_file))
        print(f"  ✅ VTP saved: {vtp_file}")
        
        # Generate summary
        summary = {
            'case_name': case_name,
            'stl_file': str(stl_file),
            'bc_file': str(bc_file),
            'vtp_file': str(vtp_file),
            'mesh_stats': {
                'n_points': int(mesh.n_points),
                'n_cells': int(mesh.n_cells),
                'bounds': list(mesh.bounds)
            },
            'flow_data': {
                'pressure_range': [float(mesh_with_data['pressure'].min()), 
                                 float(mesh_with_data['pressure'].max())],
                'velocity_range': [float(mesh_with_data['velocity_magnitude'].min()), 
                                 float(mesh_with_data['velocity_magnitude'].max())],
                'wss_range': [float(mesh_with_data['wall_shear_stress'].min()), 
                            float(mesh_with_data['wall_shear_stress'].max())]
            }
        }
        
        return summary
        
    except Exception as e:
        print(f"  ❌ Failed to process {case_name}: {e}")
        return None

def batch_convert_stl_to_vtp(data_dir: str, output_dir: str, max_cases: int = None):
    """Convert multiple STL files to VTP format."""
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Direct STL to VTP Conversion")
    print(f"Data Directory: {data_path}")
    print(f"Output Directory: {output_path}")
    print("=" * 50)
    
    # Find STL files
    stl_files = list(data_path.glob("*_clean_flat.stl"))
    
    if max_cases:
        stl_files = stl_files[:max_cases]
    
    print(f"Found {len(stl_files)} STL files to process")
    
    results = []
    successful = 0
    
    for stl_file in stl_files:
        # Find corresponding boundary condition file
        base_name = stl_file.stem.replace("_clean_flat", "")
        bc_file = data_path / f"{base_name}_boundary_conditions.json"
        
        if bc_file.exists():
            result = convert_stl_to_vtp(stl_file, bc_file, output_path)
            if result:
                results.append(result)
                successful += 1
        else:
            print(f"⚠️  No BC file for {stl_file.name}")
    
    # Save batch summary
    batch_summary = {
        'conversion_info': {
            'total_files': len(stl_files),
            'successful': successful,
            'failed': len(stl_files) - successful,
            'success_rate': successful / len(stl_files) * 100 if stl_files else 0
        },
        'results': results
    }
    
    summary_file = output_path / "conversion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\n📊 Conversion Summary:")
    print(f"Successful: {successful}/{len(stl_files)} files")
    print(f"VTP files saved in: {output_path}")
    print(f"Summary saved: {summary_file}")
    
    return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct STL to VTP Converter")
    parser.add_argument("--data-dir", default="/home/jiwoo/urp/data/uan/clean_flat_vessels",
                       help="Directory containing STL files")
    parser.add_argument("--output-dir", default="./vtp_results", 
                       help="Output directory for VTP files")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases to process")
    
    args = parser.parse_args()
    
    results = batch_convert_stl_to_vtp(args.data_dir, args.output_dir, args.max_cases)
    
    if results:
        print(f"\n🎉 VTP conversion completed!")
        print(f"Generated {len(results)} VTP files ready for ParaView visualization")
        print(f"\nTo view in ParaView:")
        print(f"1. Open ParaView")
        print(f"2. File → Open → Select *.vtp files")
        print(f"3. Apply to load geometry and data")
        print(f"4. Use 'pressure', 'velocity_magnitude', 'wall_shear_stress' for coloring")
    else:
        print(f"\n❌ VTP conversion failed!")

if __name__ == "__main__":
    main() 