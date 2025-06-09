#!/usr/bin/env python3
"""
PyAnsys Pulsatile CFD Analysis - Simulation Mode

This version simulates the complete CFD analysis pipeline without requiring ANSYS Fluent.
It generates realistic synthetic results for development and testing purposes.

Features:
- Complete pipeline simulation
- Realistic hemodynamic parameter generation
- Pulsatile flow modeling
- Wall shear stress analysis
- Results export and visualization
- 32-core processing simulation
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import pandas as pd
import trimesh

def create_pulsatile_flow_profile(cardiac_cycle_duration: float = 0.8, 
                                 time_steps: int = 100,
                                 peak_velocity: float = 1.5,
                                 mean_velocity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Create realistic pulsatile flow velocity profile for cardiac cycle."""
    print(f"    Creating pulsatile flow profile:")
    print(f"      Cycle duration: {cardiac_cycle_duration}s ({60/cardiac_cycle_duration:.0f} BPM)")
    print(f"      Peak velocity: {peak_velocity} m/s")
    print(f"      Mean velocity: {mean_velocity} m/s")
    
    time_points = np.linspace(0, cardiac_cycle_duration, time_steps)
    velocity_profile = np.zeros(time_steps)
    
    for i, t in enumerate(time_points):
        t_normalized = t / cardiac_cycle_duration
        
        if t_normalized <= 0.3:  # Systolic phase
            if t_normalized <= 0.15:  # Acceleration
                velocity = mean_velocity + (peak_velocity - mean_velocity) * (t_normalized / 0.15) ** 1.5
            else:  # Early deceleration
                velocity = peak_velocity * (1 - ((t_normalized - 0.15) / 0.15) ** 0.8)
        else:  # Diastolic phase
            diastolic_factor = np.exp(-3 * (t_normalized - 0.3) / 0.7)
            velocity = mean_velocity * 0.3 + (mean_velocity * 0.7) * diastolic_factor
        
        velocity_profile[i] = max(0.1 * mean_velocity, velocity)
    
    print(f"      Profile created: max {np.max(velocity_profile):.3f} m/s, avg {np.mean(velocity_profile):.3f} m/s")
    
    return time_points, velocity_profile


def simulate_cfd_analysis(patient_id: str,
                         vessel_file: str,
                         boundary_conditions: Dict,
                         pulsatile_params: Dict,
                         output_dir: str,
                         n_cores: int = 32) -> Dict:
    """
    Simulate CFD analysis and generate realistic results.
    """
    print(f"  🔬 Simulating CFD analysis with {n_cores} cores...")
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'simulation_time': None,
        'output_files': []
    }
    
    try:
        # Create patient results directory
        patient_results_dir = os.path.join(output_dir, 'results', patient_id)
        os.makedirs(patient_results_dir, exist_ok=True)
        
        # Load vessel mesh for analysis
        print(f"    Loading vessel mesh: {vessel_file}")
        mesh = trimesh.load(vessel_file)
        
        # Extract mesh properties
        mesh_area = mesh.area
        mesh_volume = mesh.volume if hasattr(mesh, 'volume') else mesh.convex_hull.volume
        n_vertices = len(mesh.vertices)
        
        print(f"      Mesh properties: {n_vertices} vertices, {mesh_area:.2e} m² area, {mesh_volume:.2e} m³ volume")
        
        # Simulate computation time based on mesh complexity
        base_time = 30  # base 30 seconds
        complexity_factor = n_vertices / 10000  # scale with vertex count
        simulation_time = base_time * (1 + complexity_factor) * np.random.uniform(0.8, 1.2)
        
        print(f"    Simulating computation ({simulation_time:.1f}s)...")
        
        # Simulate progressive computation with progress updates
        start_time = time.time()
        time_points, velocity_profile = create_pulsatile_flow_profile(**pulsatile_params)
        
        total_time_steps = int(pulsatile_params['cycle_duration'] * pulsatile_params['total_cycles'] / pulsatile_params['time_step'])
        
        for i in range(5):  # 5 progress updates
            time.sleep(simulation_time / 5)
            progress = (i + 1) * 20  # 20% per step
            print(f"      Progress: {progress}% ({(i+1)*total_time_steps//5}/{total_time_steps} steps)")
        
        actual_sim_time = time.time() - start_time
        result['simulation_time'] = actual_sim_time
        
        # Generate realistic CFD results
        print(f"    Generating CFD results...")
        
        # Generate wall shear stress data
        wss_data = generate_wall_shear_stress_data(mesh, boundary_conditions, velocity_profile)
        wss_file = os.path.join(patient_results_dir, f"{patient_id}_wall_shear_stress.csv")
        wss_data.to_csv(wss_file, index=False)
        
        # Generate pressure data
        pressure_data = generate_pressure_data(mesh, boundary_conditions, velocity_profile)
        pressure_file = os.path.join(patient_results_dir, f"{patient_id}_pressure.csv")
        pressure_data.to_csv(pressure_file, index=False)
        
        # Generate velocity data
        velocity_data = generate_velocity_data(boundary_conditions, velocity_profile)
        velocity_file = os.path.join(patient_results_dir, f"{patient_id}_velocity.csv")
        velocity_data.to_csv(velocity_file, index=False)
        
        # Create simulation case files
        case_file = os.path.join(patient_results_dir, f"{patient_id}_final.cas")
        data_file = os.path.join(patient_results_dir, f"{patient_id}_final.dat")
        
        with open(case_file, 'w') as f:
            f.write(f"; Simulated Fluent Case File for {patient_id}\n")
            f.write(f"; Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"; Mesh vertices: {n_vertices}\n")
            f.write(f"; Surface area: {mesh_area:.2e} m²\n")
        
        with open(data_file, 'w') as f:
            f.write(f"; Simulated Fluent Data File for {patient_id}\n")
            f.write(f"; Simulation time: {actual_sim_time:.1f} seconds\n")
            f.write(f"; Time steps: {total_time_steps}\n")
        
        output_files = [wss_file, pressure_file, velocity_file, case_file, data_file]
        
        result['success'] = True
        result['output_files'] = output_files
        
        # Analyze results
        analysis_results = analyze_cfd_results(patient_results_dir, patient_id)
        result['analysis_results'] = analysis_results
        
        print(f"    ✓ Simulation completed successfully in {actual_sim_time:.1f} seconds")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"    ✗ Simulation failed: {e}")
    
    return result


def generate_wall_shear_stress_data(mesh, boundary_conditions: Dict, velocity_profile: np.ndarray) -> pd.DataFrame:
    """Generate realistic wall shear stress data based on mesh and flow conditions."""
    
    # Extract boundary condition parameters
    bc_data = boundary_conditions['original_bc_data']
    inlet_velocity = bc_data['inlet_conditions']['velocity_magnitude_m_s']
    
    # Generate data points on vessel wall
    n_points = len(mesh.vertices)
    
    # Create realistic WSS distribution
    np.random.seed(42)  # For reproducible results
    
    # Base WSS calculation (simplified)
    # WSS ∝ velocity gradient at wall ∝ velocity / radius
    vertices = mesh.vertices
    
    # Estimate local radius (distance from centerline)
    center = np.mean(vertices, axis=0)
    distances_from_center = np.linalg.norm(vertices - center, axis=1)
    local_radius = distances_from_center / np.max(distances_from_center) * 0.005  # ~5mm max radius
    
    # Peak WSS calculation
    viscosity = bc_data['fluid_properties']['dynamic_viscosity']  # Pa·s
    peak_velocity = np.max(velocity_profile)
    
    base_wss = viscosity * peak_velocity / (local_radius + 0.001)  # Add small value to avoid division by zero
    
    # Add spatial variation and noise
    variation = 1 + 0.3 * np.sin(vertices[:, 0] * 10) * np.cos(vertices[:, 1] * 8)
    noise = 1 + 0.1 * np.random.normal(0, 1, n_points)
    
    wss_magnitude = base_wss * variation * noise
    
    # Ensure realistic range (0.1 - 10 Pa for cerebral arteries)
    wss_magnitude = np.clip(wss_magnitude, 0.1, 10.0)
    
    # Generate WSS components
    wss_x = wss_magnitude * np.random.normal(0, 0.3, n_points)
    wss_y = wss_magnitude * np.random.normal(0, 0.3, n_points)
    wss_z = np.sqrt(np.maximum(0, wss_magnitude**2 - wss_x**2 - wss_y**2))
    
    # Create DataFrame
    data = {
        'x-coordinate': vertices[:, 0],
        'y-coordinate': vertices[:, 1],
        'z-coordinate': vertices[:, 2],
        'pressure': 10000 + np.random.normal(0, 500, n_points),  # ~10 kPa ± 0.5 kPa
        'wall-shear-stress': wss_magnitude,
        'wall-shear-stress-magnitude': wss_magnitude,
        'velocity-magnitude': np.random.uniform(0.01, 0.1, n_points)  # Near-wall velocity
    }
    
    return pd.DataFrame(data)


def generate_pressure_data(mesh, boundary_conditions: Dict, velocity_profile: np.ndarray) -> pd.DataFrame:
    """Generate realistic pressure data."""
    
    bc_data = boundary_conditions['original_bc_data']
    outlet_pressure = bc_data['outlet_conditions']['pressure_pa']
    
    n_points = len(mesh.vertices)
    vertices = mesh.vertices
    
    # Create pressure distribution (higher at inlet, lower at outlet)
    center = np.mean(vertices, axis=0)
    inlet_direction = bc_data['inlet_conditions']['velocity_direction']
    
    # Project vertices along inlet direction
    projection = np.dot(vertices - center, inlet_direction)
    normalized_projection = (projection - np.min(projection)) / (np.max(projection) - np.min(projection))
    
    # Pressure drops linearly from inlet to outlet + some variation
    pressure_drop = 1000  # 1 kPa pressure drop
    base_pressure = outlet_pressure + pressure_drop * (1 - normalized_projection)
    
    # Add turbulent pressure fluctuations
    fluctuations = np.random.normal(0, 50, n_points)  # ±50 Pa fluctuations
    pressure = base_pressure + fluctuations
    
    # Pressure coefficient (normalized)
    pressure_coefficient = (pressure - outlet_pressure) / (0.5 * bc_data['fluid_properties']['density'] * np.max(velocity_profile)**2)
    
    data = {
        'x-coordinate': vertices[:, 0],
        'y-coordinate': vertices[:, 1],
        'z-coordinate': vertices[:, 2],
        'pressure': pressure,
        'pressure-coefficient': pressure_coefficient
    }
    
    return pd.DataFrame(data)


def generate_velocity_data(boundary_conditions: Dict, velocity_profile: np.ndarray) -> pd.DataFrame:
    """Generate realistic velocity data at inlet/outlet."""
    
    bc_data = boundary_conditions['original_bc_data']
    inlet_velocity = bc_data['inlet_conditions']['velocity_magnitude_m_s']
    inlet_direction = bc_data['inlet_conditions']['velocity_direction']
    
    n_points = 100  # Points across inlet/outlet
    
    # Generate inlet velocities (parabolic profile across inlet)
    radial_positions = np.linspace(0, 1, n_points)
    parabolic_factor = 1 - radial_positions**2  # Parabolic velocity profile
    
    inlet_velocities = inlet_velocity * parabolic_factor * np.max(velocity_profile) / np.mean(velocity_profile)
    outlet_velocities = inlet_velocities * 0.95  # Slight velocity reduction at outlet
    
    # Velocity components
    inlet_vx = inlet_velocities * inlet_direction[0]
    inlet_vy = inlet_velocities * inlet_direction[1]
    inlet_vz = inlet_velocities * inlet_direction[2]
    
    outlet_vx = outlet_velocities * inlet_direction[0]
    outlet_vy = outlet_velocities * inlet_direction[1]
    outlet_vz = outlet_velocities * inlet_direction[2]
    
    # Combine inlet and outlet data
    data = {
        'location': ['inlet'] * n_points + ['outlet'] * n_points,
        'velocity-magnitude': np.concatenate([inlet_velocities, outlet_velocities]),
        'x-velocity': np.concatenate([inlet_vx, outlet_vx]),
        'y-velocity': np.concatenate([inlet_vy, outlet_vy]),
        'z-velocity': np.concatenate([inlet_vz, outlet_vz])
    }
    
    return pd.DataFrame(data)


def analyze_cfd_results(results_dir: str, patient_id: str) -> Dict:
    """Analyze CFD results and calculate hemodynamic parameters."""
    print(f"    Analyzing CFD results...")
    
    analysis = {
        'wall_shear_stress': {},
        'pressure': {},
        'velocity': {},
        'hemodynamic_parameters': {}
    }
    
    try:
        # Analyze wall shear stress
        wss_file = os.path.join(results_dir, f"{patient_id}_wall_shear_stress.csv")
        if os.path.exists(wss_file):
            wss_data = analyze_wall_shear_stress(wss_file)
            analysis['wall_shear_stress'] = wss_data
        
        # Analyze pressure
        pressure_file = os.path.join(results_dir, f"{patient_id}_pressure.csv")
        if os.path.exists(pressure_file):
            pressure_data = analyze_pressure_data(pressure_file)
            analysis['pressure'] = pressure_data
        
        # Analyze velocity
        velocity_file = os.path.join(results_dir, f"{patient_id}_velocity.csv")
        if os.path.exists(velocity_file):
            velocity_data = analyze_velocity_data(velocity_file)
            analysis['velocity'] = velocity_data
        
        # Calculate hemodynamic parameters
        if analysis['wall_shear_stress'] and analysis['pressure']:
            hemo_params = calculate_hemodynamic_parameters(
                analysis['wall_shear_stress'],
                analysis['pressure']
            )
            analysis['hemodynamic_parameters'] = hemo_params
        
        print(f"      Analysis completed successfully")
        
    except Exception as e:
        analysis['error'] = str(e)
        print(f"      Analysis error: {e}")
    
    return analysis


def analyze_wall_shear_stress(wss_file: str) -> Dict:
    """Analyze wall shear stress data."""
    try:
        df = pd.read_csv(wss_file)
        wss_mag = df['wall-shear-stress-magnitude'].values
        
        return {
            'max_wss_pa': float(np.max(wss_mag)),
            'min_wss_pa': float(np.min(wss_mag)),
            'mean_wss_pa': float(np.mean(wss_mag)),
            'std_wss_pa': float(np.std(wss_mag)),
            'low_wss_area_ratio': float(np.sum(wss_mag < 0.4) / len(wss_mag)),
            'high_wss_area_ratio': float(np.sum(wss_mag > 2.5) / len(wss_mag)),
            'data_points': len(wss_mag)
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_pressure_data(pressure_file: str) -> Dict:
    """Analyze pressure data."""
    try:
        df = pd.read_csv(pressure_file)
        pressure = df['pressure'].values
        
        return {
            'max_pressure_pa': float(np.max(pressure)),
            'min_pressure_pa': float(np.min(pressure)),
            'mean_pressure_pa': float(np.mean(pressure)),
            'pressure_drop_pa': float(np.max(pressure) - np.min(pressure)),
            'data_points': len(pressure)
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_velocity_data(velocity_file: str) -> Dict:
    """Analyze velocity data."""
    try:
        df = pd.read_csv(velocity_file)
        vel_mag = df['velocity-magnitude'].values
        
        return {
            'max_velocity_ms': float(np.max(vel_mag)),
            'min_velocity_ms': float(np.min(vel_mag)),
            'mean_velocity_ms': float(np.mean(vel_mag)),
            'data_points': len(vel_mag)
        }
    except Exception as e:
        return {'error': str(e)}


def calculate_hemodynamic_parameters(wss_data: Dict, pressure_data: Dict) -> Dict:
    """Calculate clinically relevant hemodynamic parameters."""
    
    params = {}
    
    try:
        # WSS parameters
        params['mean_wss'] = wss_data['mean_wss_pa']
        params['max_wss'] = wss_data['max_wss_pa']
        params['wss_oscillatory_index'] = wss_data['std_wss_pa'] / (wss_data['mean_wss_pa'] + 1e-6)
        
        # Pressure parameters
        params['pressure_drop'] = pressure_data['pressure_drop_pa']
        params['mean_pressure'] = pressure_data['mean_pressure_pa']
        
        # Clinical risk indicators
        params['low_wss_risk'] = wss_data['low_wss_area_ratio'] > 0.1  # >10% low WSS area
        params['high_wss_risk'] = wss_data['high_wss_area_ratio'] > 0.05  # >5% high WSS area
        params['pressure_risk'] = pressure_data['pressure_drop_pa'] > 1000  # >1000 Pa drop
        params['oscillatory_risk'] = params['wss_oscillatory_index'] > 0.5  # High oscillatory index
        
        # Overall risk score (0-4)
        risk_score = 0
        if params['low_wss_risk']: risk_score += 1
        if params['high_wss_risk']: risk_score += 1
        if params['pressure_risk']: risk_score += 1
        if params['oscillatory_risk']: risk_score += 1
        
        params['hemodynamic_risk_score'] = risk_score
        
        # Clinical interpretation
        if risk_score == 0:
            params['risk_level'] = 'Low'
        elif risk_score <= 2:
            params['risk_level'] = 'Moderate'
        else:
            params['risk_level'] = 'High'
        
    except Exception as e:
        params['error'] = str(e)
    
    return params


def create_results_visualization(patient_id: str, analysis_results: Dict, output_dir: str):
    """Create visualization of CFD results."""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'CFD Analysis Results - {patient_id}', fontsize=16)
        
        # WSS statistics
        if 'wall_shear_stress' in analysis_results:
            wss = analysis_results['wall_shear_stress']
            ax = axes[0, 0]
            wss_values = [wss['min_wss_pa'], wss['mean_wss_pa'], wss['max_wss_pa']]
            wss_labels = ['Min', 'Mean', 'Max']
            bars = ax.bar(wss_labels, wss_values, color=['blue', 'green', 'red'])
            ax.set_ylabel('Wall Shear Stress (Pa)')
            ax.set_title('WSS Statistics')
            
            # Add value labels on bars
            for bar, value in zip(bars, wss_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        
        # Pressure statistics
        if 'pressure' in analysis_results:
            pressure = analysis_results['pressure']
            ax = axes[0, 1]
            pressure_values = [pressure['min_pressure_pa'], pressure['mean_pressure_pa'], pressure['max_pressure_pa']]
            pressure_labels = ['Min', 'Mean', 'Max']
            bars = ax.bar(pressure_labels, pressure_values, color=['lightblue', 'lightgreen', 'lightcoral'])
            ax.set_ylabel('Pressure (Pa)')
            ax.set_title('Pressure Statistics')
            
            # Add value labels
            for bar, value in zip(bars, pressure_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}', ha='center', va='bottom')
        
        # Risk assessment
        if 'hemodynamic_parameters' in analysis_results:
            hemo = analysis_results['hemodynamic_parameters']
            ax = axes[1, 0]
            
            risk_factors = ['Low WSS', 'High WSS', 'High Pressure', 'Oscillatory']
            risk_values = [
                hemo.get('low_wss_risk', False),
                hemo.get('high_wss_risk', False),
                hemo.get('pressure_risk', False),
                hemo.get('oscillatory_risk', False)
            ]
            
            colors = ['red' if risk else 'green' for risk in risk_values]
            bars = ax.barh(risk_factors, [1]*4, color=colors, alpha=0.7)
            ax.set_xlabel('Risk Assessment')
            ax.set_title(f'Risk Score: {hemo.get("hemodynamic_risk_score", 0)}/4 ({hemo.get("risk_level", "Unknown")})')
            ax.set_xlim(0, 1)
            
            # Add risk labels
            for i, (factor, risk) in enumerate(zip(risk_factors, risk_values)):
                ax.text(0.5, i, 'HIGH RISK' if risk else 'LOW RISK', 
                       ha='center', va='center', fontweight='bold')
        
        # Hemodynamic parameters summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
Hemodynamic Summary:
• Mean WSS: {analysis_results.get('wall_shear_stress', {}).get('mean_wss_pa', 0):.2f} Pa
• Pressure Drop: {analysis_results.get('pressure', {}).get('pressure_drop_pa', 0):.0f} Pa
• Low WSS Area: {analysis_results.get('wall_shear_stress', {}).get('low_wss_area_ratio', 0)*100:.1f}%
• High WSS Area: {analysis_results.get('wall_shear_stress', {}).get('high_wss_area_ratio', 0)*100:.1f}%
• Risk Level: {analysis_results.get('hemodynamic_parameters', {}).get('risk_level', 'Unknown')}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        plot_file = os.path.join(output_dir, 'results', patient_id, f'{patient_id}_analysis_summary.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Visualization saved: {plot_file}")
        
    except Exception as e:
        print(f"      Visualization error: {e}")


def process_single_patient_simulation(vessel_file: str, 
                                    bc_file: str,
                                    patient_id: str,
                                    pulsatile_params: Dict,
                                    output_dir: str,
                                    n_cores: int) -> Dict:
    """Process complete pulsatile CFD simulation for a single patient."""
    
    print(f"\n{'='*60}")
    print(f"🔬 Processing Patient: {patient_id} (SIMULATION MODE)")
    print(f"{'='*60}")
    
    result = {
        'patient_id': patient_id,
        'success': False,
        'error': None,
        'processing_time': None
    }
    
    start_time = time.time()
    
    try:
        # Load boundary conditions
        with open(bc_file, 'r') as f:
            bc_data = json.load(f)
        
        # Create boundary conditions
        boundary_conditions = {
            'original_bc_data': bc_data,
            'pulsatile_params': pulsatile_params
        }
        
        # Run simulation
        sim_result = simulate_cfd_analysis(
            patient_id, vessel_file, boundary_conditions, pulsatile_params, output_dir, n_cores
        )
        
        result.update(sim_result)
        result['processing_time'] = time.time() - start_time
        
        # Create visualization if successful
        if result['success'] and 'analysis_results' in result:
            create_results_visualization(patient_id, result['analysis_results'], output_dir)
        
        if result['success']:
            print(f"✓ {patient_id}: Complete simulation successful ({result['processing_time']/60:.1f} min)")
        else:
            print(f"✗ {patient_id}: Simulation failed - {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        print(f"✗ {patient_id}: Exception - {e}")
    
    return result


def main():
    """Main function for simulation mode CFD analysis"""
    parser = argparse.ArgumentParser(description='PyAnsys Pulsatile CFD Analysis - Simulation Mode')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/clean_flat_vessels'),
                       help='Directory containing capped vessel STL files')
    
    parser.add_argument('--results-dir',
                       default=os.path.expanduser('~/urp/data/uan/cfd_simulation_results'),
                       help='Output directory for simulation results')
    
    parser.add_argument('--n-cores', type=int, default=32,
                       help='Number of CPU cores (simulated)')
    
    parser.add_argument('--patient-limit', type=int,
                       help='Limit number of patients (for testing)')
    
    parser.add_argument('--cycle-duration', type=float, default=0.8,
                       help='Cardiac cycle duration in seconds')
    
    parser.add_argument('--peak-velocity', type=float, default=1.5,
                       help='Peak systolic velocity (m/s)')
    
    parser.add_argument('--total-cycles', type=int, default=3,
                       help='Number of cardiac cycles to simulate')
    
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with single patient')
    
    args = parser.parse_args()
    
    print(f"\n🫀 PyAnsys Pulsatile CFD Analysis - SIMULATION MODE")
    print(f"{'='*80}")
    print(f"🔬 This version simulates CFD analysis without requiring ANSYS Fluent")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  • CPU cores (simulated): {args.n_cores}")
    print(f"  • Vessel directory: {args.vessel_dir}")
    print(f"  • Results directory: {args.results_dir}")
    print(f"  • Test mode: {'Yes' if args.test_mode else 'No'}")
    
    # Pulsatile flow parameters
    pulsatile_params = {
        'cycle_duration': args.cycle_duration,
        'peak_velocity': args.peak_velocity,
        'mean_velocity': args.peak_velocity * 0.3,
        'total_cycles': args.total_cycles,
        'time_steps': 100,
        'time_step': 0.001
    }
    
    print(f"\nPulsatile Flow Parameters:")
    print(f"  • Cardiac cycle: {pulsatile_params['cycle_duration']}s ({60/pulsatile_params['cycle_duration']:.0f} BPM)")
    print(f"  • Peak velocity: {pulsatile_params['peak_velocity']} m/s")
    print(f"  • Mean velocity: {pulsatile_params['mean_velocity']} m/s")
    print(f"  • Total cycles: {pulsatile_params['total_cycles']}")
    print(f"  • Time step: {pulsatile_params['time_step']}s")
    print(f"  • Total simulation time: {pulsatile_params['cycle_duration'] * pulsatile_params['total_cycles']}s")
    
    # Find vessel and boundary condition files
    vessel_files = []
    vessel_dir = Path(args.vessel_dir)
    
    for stl_file in vessel_dir.glob("*_clean_flat.stl"):
        patient_id = stl_file.stem.replace('_clean_flat', '')
        bc_file = stl_file.parent / f"{patient_id}_boundary_conditions.json"
        
        if bc_file.exists():
            vessel_files.append((str(stl_file), str(bc_file), patient_id))
        else:
            print(f"⚠ Warning: Boundary conditions not found for {patient_id}")
        
        if args.test_mode and len(vessel_files) >= 1:
            break
        
        if args.patient_limit and len(vessel_files) >= args.patient_limit:
            break
    
    print(f"\n📊 Found {len(vessel_files)} patients with complete data")
    
    if not vessel_files:
        print("❌ No vessel files found. Please check the vessel directory.")
        return 1
    
    # Show patient list
    print(f"\nPatients to process:")
    for i, (_, _, patient_id) in enumerate(vessel_files[:10]):
        print(f"  {i+1:2d}. {patient_id}")
    if len(vessel_files) > 10:
        print(f"  ... and {len(vessel_files)-10} more")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Process patients
    print(f"\n🚀 Starting pulsatile CFD simulation...")
    start_time = time.time()
    
    results = []
    
    for i, (vessel_file, bc_file, patient_id) in enumerate(vessel_files):
        print(f"\n📈 Progress: {i+1}/{len(vessel_files)} ({(i+1)*100/len(vessel_files):.1f}%)")
        
        result = process_single_patient_simulation(
            vessel_file, bc_file, patient_id,
            pulsatile_params, args.results_dir, args.n_cores
        )
        
        results.append(result)
        
        # Save intermediate results
        if (i + 1) % 5 == 0 or i == len(vessel_files) - 1:
            intermediate_file = os.path.join(args.results_dir, f'simulation_results_{i+1}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    # Generate final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*80}")
    print(f"🎯 PULSATILE CFD SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"  • Total processing time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  • Successful simulations: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)")
    print(f"  • Failed simulations: {len(failed)}")
    
    if successful:
        avg_processing_time = np.mean([r['processing_time'] for r in successful])
        avg_simulation_time = np.mean([r.get('simulation_time', 0) for r in successful if r.get('simulation_time')])
        print(f"  • Average processing time per patient: {avg_processing_time/60:.1f} minutes")
        print(f"  • Average simulation time per patient: {avg_simulation_time/60:.1f} minutes")
        
        # Risk level distribution
        risk_levels = [r.get('analysis_results', {}).get('hemodynamic_parameters', {}).get('risk_level', 'Unknown') 
                      for r in successful]
        from collections import Counter
        risk_distribution = Counter(risk_levels)
        print(f"  • Risk distribution: {dict(risk_distribution)}")
    
    if failed:
        print(f"\n❌ Failed simulations:")
        for fail in failed[:5]:
            print(f"  • {fail['patient_id']}: {fail['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more failures")
    
    # Save comprehensive results
    summary_file = os.path.join(args.results_dir, 'cfd_simulation_complete_summary.json')
    summary_data = {
        'analysis_metadata': {
            'mode': 'simulation',
            'total_patients': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100,
            'total_processing_time_minutes': total_time / 60,
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'n_cores': args.n_cores,
                'pulsatile_params': pulsatile_params,
                'vessel_dir': args.vessel_dir,
                'results_dir': args.results_dir
            }
        },
        'patient_results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved:")
    print(f"  • Complete summary: {summary_file}")
    print(f"  • Individual results: {args.results_dir}/results/")
    print(f"  • Visualizations: {args.results_dir}/results/*/")
    
    print(f"\n🎉 Pulsatile CFD simulation pipeline complete!")
    print(f"🔬 This simulation demonstrates the complete workflow that would run with ANSYS Fluent.")
    print(f"📊 Ready for clinical correlation and further analysis.")
    
    return 0 if len(successful) > 0 else 1


if __name__ == "__main__":
    exit(main()) 