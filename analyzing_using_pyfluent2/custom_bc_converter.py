#!/usr/bin/env python3
"""
Custom Boundary Condition VTP Converter
Author: Jiwoo Lee

Allows easy modification of boundary conditions for VTP generation.
"""

import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pyvista as pv

# CUSTOMIZABLE PARAMETERS - EDIT THESE
CUSTOM_FLOW_SETTINGS = {
    # Velocity scaling factor (1.0 = original, 2.0 = double velocity)
    'velocity_scale': 1.5,
    
    # Pressure scaling factor
    'pressure_scale': 1.2,
    
    # Wall shear stress multiplier
    'wss_multiplier': 2.0,
    
    # Override inlet velocity (None = use from BC file)
    'override_inlet_velocity': None,  # Set to value like 1.0 to override
    
    # Override outlet pressure (None = use from BC file)
    'override_outlet_pressure': None,  # Set to value like 12000 to override
    
    # Custom flow direction [X, Y, Z] (None = auto-detect)
    'custom_flow_direction': None,  # Set to [0, 0, 1] for pure Z-direction
    
    # Velocity profile type: 'parabolic', 'uniform', 'turbulent'
    'velocity_profile': 'parabolic',
    
    # Pressure drop type: 'linear', 'exponential', 'custom'
    'pressure_drop': 'linear'
}

def load_boundary_conditions(bc_file: Path) -> Dict:
    """Load boundary conditions from JSON file."""
    try:
        with open(bc_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {bc_file}: {e}")
        return {}

def add_custom_flow_data(mesh: pv.PolyData, bc_data: Dict) -> pv.PolyData:
    """Add custom flow data to the mesh based on modified boundary conditions."""
    
    # Get original fluid properties
    fluid_props = bc_data.get('fluid_properties', {})
    inlet_data = bc_data.get('inlet_conditions', {})
    outlet_data = bc_data.get('outlet_conditions', {})
    
    # Apply custom overrides or use original values
    if CUSTOM_FLOW_SETTINGS['override_inlet_velocity'] is not None:
        inlet_velocity = CUSTOM_FLOW_SETTINGS['override_inlet_velocity']
    else:
        inlet_velocity = inlet_data.get('velocity_magnitude_m_s', 0.5)
    
    if CUSTOM_FLOW_SETTINGS['override_outlet_pressure'] is not None:
        outlet_pressure = CUSTOM_FLOW_SETTINGS['override_outlet_pressure']
    else:
        outlet_pressure = outlet_data.get('pressure_pa', 10665.76)
    
    # Apply scaling factors
    inlet_velocity *= CUSTOM_FLOW_SETTINGS['velocity_scale']
    outlet_pressure *= CUSTOM_FLOW_SETTINGS['pressure_scale']
    
    n_points = mesh.n_points
    
    # Generate pressure field based on type
    if CUSTOM_FLOW_SETTINGS['pressure_drop'] == 'linear':
        pressure = generate_linear_pressure(mesh, outlet_pressure, inlet_velocity)
    elif CUSTOM_FLOW_SETTINGS['pressure_drop'] == 'exponential':
        pressure = generate_exponential_pressure(mesh, outlet_pressure, inlet_velocity)
    else:  # custom
        pressure = generate_custom_pressure(mesh, outlet_pressure, inlet_velocity)
    
    # Generate velocity field based on profile type
    if CUSTOM_FLOW_SETTINGS['velocity_profile'] == 'parabolic':
        velocity_magnitude = generate_parabolic_velocity(mesh, inlet_velocity)
    elif CUSTOM_FLOW_SETTINGS['velocity_profile'] == 'uniform':
        velocity_magnitude = generate_uniform_velocity(mesh, inlet_velocity)
    else:  # turbulent
        velocity_magnitude = generate_turbulent_velocity(mesh, inlet_velocity)
    
    # Generate velocity vectors
    velocity_vectors = generate_velocity_vectors(mesh, velocity_magnitude)
    
    # Calculate wall shear stress
    wall_shear_stress = calculate_wall_shear_stress(mesh, velocity_magnitude)
    
    # Add all data to mesh
    mesh['pressure'] = pressure
    mesh['velocity_magnitude'] = velocity_magnitude
    mesh['velocity'] = velocity_vectors
    mesh['wall_shear_stress'] = wall_shear_stress
    
    # Add material properties
    mesh['reynolds_number'] = np.full(n_points, bc_data.get('flow_parameters', {}).get('reynolds_number', 400))
    mesh['density'] = np.full(n_points, fluid_props.get('density', 1060.0))
    mesh['viscosity'] = np.full(n_points, fluid_props.get('dynamic_viscosity', 0.004))
    
    return mesh

def generate_linear_pressure(mesh: pv.PolyData, outlet_pressure: float, inlet_velocity: float) -> np.ndarray:
    """Generate linear pressure gradient."""
    z_coords = mesh.points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    if z_max > z_min:
        z_norm = (z_coords - z_min) / (z_max - z_min)
        # Higher pressure at inlet (estimated from velocity)
        inlet_pressure = outlet_pressure + (inlet_velocity * 2000)  # Simple estimate
        pressure = outlet_pressure + (inlet_pressure - outlet_pressure) * (1 - z_norm)
    else:
        pressure = np.full(len(z_coords), outlet_pressure)
    
    return pressure

def generate_exponential_pressure(mesh: pv.PolyData, outlet_pressure: float, inlet_velocity: float) -> np.ndarray:
    """Generate exponential pressure drop."""
    z_coords = mesh.points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    if z_max > z_min:
        z_norm = (z_coords - z_min) / (z_max - z_min)
        inlet_pressure = outlet_pressure + (inlet_velocity * 3000)
        # Exponential decay
        pressure = outlet_pressure + (inlet_pressure - outlet_pressure) * np.exp(-2 * z_norm)
    else:
        pressure = np.full(len(z_coords), outlet_pressure)
    
    return pressure

def generate_custom_pressure(mesh: pv.PolyData, outlet_pressure: float, inlet_velocity: float) -> np.ndarray:
    """Generate custom pressure distribution."""
    # Example: Sinusoidal pressure variation
    z_coords = mesh.points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    if z_max > z_min:
        z_norm = (z_coords - z_min) / (z_max - z_min)
        inlet_pressure = outlet_pressure + (inlet_velocity * 2500)
        # Sinusoidal variation
        pressure = outlet_pressure + (inlet_pressure - outlet_pressure) * (1 - z_norm) * (1 + 0.1 * np.sin(10 * z_norm))
    else:
        pressure = np.full(len(z_coords), outlet_pressure)
    
    return pressure

def generate_parabolic_velocity(mesh: pv.PolyData, inlet_velocity: float) -> np.ndarray:
    """Generate parabolic velocity profile."""
    center = mesh.center
    distances = np.linalg.norm(mesh.points - center, axis=1)
    max_distance = distances.max()
    
    if max_distance > 0:
        # Parabolic profile (higher at center, lower at walls)
        velocity_magnitude = inlet_velocity * (1 - (distances / max_distance) ** 2)
        velocity_magnitude = np.maximum(velocity_magnitude, 0.05 * inlet_velocity)
    else:
        velocity_magnitude = np.full(mesh.n_points, inlet_velocity)
    
    return velocity_magnitude

def generate_uniform_velocity(mesh: pv.PolyData, inlet_velocity: float) -> np.ndarray:
    """Generate uniform velocity profile."""
    return np.full(mesh.n_points, inlet_velocity)

def generate_turbulent_velocity(mesh: pv.PolyData, inlet_velocity: float) -> np.ndarray:
    """Generate turbulent-like velocity profile."""
    center = mesh.center
    distances = np.linalg.norm(mesh.points - center, axis=1)
    max_distance = distances.max()
    
    if max_distance > 0:
        # 1/7th power law approximation for turbulent flow
        velocity_magnitude = inlet_velocity * (1 - (distances / max_distance)) ** (1/7)
        velocity_magnitude = np.maximum(velocity_magnitude, 0.1 * inlet_velocity)
        
        # Add some turbulent fluctuations
        np.random.seed(42)  # For reproducibility
        turbulent_fluctuation = 0.05 * inlet_velocity * np.random.normal(0, 1, mesh.n_points)
        velocity_magnitude += turbulent_fluctuation
        velocity_magnitude = np.maximum(velocity_magnitude, 0.05 * inlet_velocity)
    else:
        velocity_magnitude = np.full(mesh.n_points, inlet_velocity)
    
    return velocity_magnitude

def generate_velocity_vectors(mesh: pv.PolyData, velocity_magnitude: np.ndarray) -> np.ndarray:
    """Generate velocity vector field."""
    velocity_vectors = np.zeros((mesh.n_points, 3))
    
    # Use custom flow direction if specified
    if CUSTOM_FLOW_SETTINGS['custom_flow_direction'] is not None:
        flow_direction = np.array(CUSTOM_FLOW_SETTINGS['custom_flow_direction'])
        flow_direction = flow_direction / np.linalg.norm(flow_direction)
    else:
        # Auto-detect flow direction (assuming Z is primary)
        flow_direction = np.array([0.1, 0.1, 1.0])
        flow_direction = flow_direction / np.linalg.norm(flow_direction)
    
    # Apply flow direction to each point
    for i in range(mesh.n_points):
        velocity_vectors[i] = velocity_magnitude[i] * flow_direction
    
    return velocity_vectors

def calculate_wall_shear_stress(mesh: pv.PolyData, velocity_magnitude: np.ndarray) -> np.ndarray:
    """Calculate wall shear stress."""
    center = mesh.center
    distances = np.linalg.norm(mesh.points - center, axis=1)
    max_distance = distances.max()
    
    if max_distance > 0:
        # WSS is higher near walls
        wall_factor = distances / max_distance
        base_wss = 3.0 * CUSTOM_FLOW_SETTINGS['wss_multiplier']
        wall_shear_stress = base_wss * wall_factor * (velocity_magnitude / velocity_magnitude.max())
    else:
        wall_shear_stress = np.full(mesh.n_points, 2.5 * CUSTOM_FLOW_SETTINGS['wss_multiplier'])
    
    return wall_shear_stress

def convert_with_custom_bc(stl_file: Path, bc_file: Path, output_dir: Path) -> bool:
    """Convert STL file to VTP with custom boundary conditions."""
    
    case_name = stl_file.stem.replace("_clean_flat", "")
    
    try:
        print(f"🔄 Processing {case_name} with custom BCs...")
        
        # Load STL mesh
        mesh = pv.read(str(stl_file))
        print(f"  📁 Loaded STL: {mesh.n_points} points, {mesh.n_cells} cells")
        
        # Load boundary conditions
        bc_data = load_boundary_conditions(bc_file)
        if not bc_data:
            print(f"  ⚠️  No boundary conditions, using defaults")
            bc_data = {}
        
        # Add custom flow data
        mesh_with_data = add_custom_flow_data(mesh, bc_data)
        print(f"  📊 Added custom flow data fields")
        
        # Save as VTP
        vtp_file = output_dir / f"{case_name}_custom.vtp"
        mesh_with_data.save(str(vtp_file))
        print(f"  ✅ Custom VTP saved: {vtp_file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to process {case_name}: {e}")
        return False

def main():
    """Main function to test custom boundary conditions."""
    
    print("🎨 Custom Boundary Condition VTP Generator")
    print("=" * 50)
    print("Current settings:")
    for key, value in CUSTOM_FLOW_SETTINGS.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Test with a single case
    data_dir = Path("/home/jiwoo/urp/data/uan/clean_flat_vessels")
    output_dir = Path("./custom_vtp_results")
    output_dir.mkdir(exist_ok=True)
    
    # Use the first available case for testing
    stl_files = list(data_dir.glob("*_clean_flat.stl"))
    if stl_files:
        stl_file = stl_files[0]
        base_name = stl_file.stem.replace("_clean_flat", "")
        bc_file = data_dir / f"{base_name}_boundary_conditions.json"
        
        if bc_file.exists():
            success = convert_with_custom_bc(stl_file, bc_file, output_dir)
            if success:
                print(f"\n🎉 Custom VTP generated!")
                print(f"Compare with original VTP to see the differences")
        else:
            print(f"❌ No boundary condition file found")
    else:
        print(f"❌ No STL files found")

if __name__ == "__main__":
    main() 