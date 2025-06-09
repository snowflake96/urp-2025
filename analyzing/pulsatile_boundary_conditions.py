#!/usr/bin/env python3
"""
Pulsatile Boundary Conditions for Aneurysm CFD Analysis

Creates comprehensive pulsatile flow boundary conditions from scratch
for PyAnsys Fluent CFD analysis using the aneurysm conda environment.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
import time
from dataclasses import dataclass

@dataclass
class CardiacCycleParameters:
    """Parameters for cardiac cycle modeling"""
    heart_rate_bpm: float = 75.0
    systolic_duration_ratio: float = 0.35
    peak_velocity_ratio: float = 2.5
    diastolic_decay_rate: float = 3.0
    minimum_flow_ratio: float = 0.15

@dataclass
class BloodProperties:
    """Blood properties for CFD analysis"""
    density: float = 1060.0  # kg/m³
    dynamic_viscosity: float = 0.004  # Pa·s
    specific_heat: float = 3617.0  # J/kg·K
    thermal_conductivity: float = 0.52  # W/m·K

class PulsatileFlowGenerator:
    """Generate physiological pulsatile flow profiles"""
    
    def __init__(self, cardiac_params: CardiacCycleParameters):
        self.cardiac_params = cardiac_params
        self.cycle_duration = 60.0 / cardiac_params.heart_rate_bpm
        
    def generate_velocity_profile(self, mean_velocity: float, time_steps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate physiological velocity waveform"""
        
        time_points = np.linspace(0, self.cycle_duration, time_steps)
        velocity_profile = np.zeros(time_steps)
        
        systolic_duration = self.cycle_duration * self.cardiac_params.systolic_duration_ratio
        peak_velocity = mean_velocity * self.cardiac_params.peak_velocity_ratio
        min_velocity = mean_velocity * self.cardiac_params.minimum_flow_ratio
        
        for i, t in enumerate(time_points):
            if t <= systolic_duration:
                # Systolic phase
                t_norm = t / systolic_duration
                
                if t_norm <= 0.3:  # Acceleration
                    acceleration_factor = 0.5 * (1 + np.tanh(10 * (t_norm - 0.15)))
                    velocity = min_velocity + (peak_velocity - min_velocity) * acceleration_factor
                elif t_norm <= 0.7:  # Peak
                    peak_variation = 1.0 - 0.1 * np.sin(np.pi * (t_norm - 0.3) / 0.4)
                    velocity = peak_velocity * peak_variation
                else:  # Deceleration
                    decel_factor = (1.0 - t_norm) / 0.3
                    velocity = min_velocity + (peak_velocity - min_velocity) * decel_factor**2
            else:
                # Diastolic phase
                t_diastolic = t - systolic_duration
                t_norm_diastolic = t_diastolic / (self.cycle_duration - systolic_duration)
                
                decay_factor = np.exp(-self.cardiac_params.diastolic_decay_rate * t_norm_diastolic)
                oscillation = 1.0 + 0.1 * np.sin(2 * np.pi * t_norm_diastolic) * decay_factor
                
                velocity = min_velocity + (mean_velocity - min_velocity) * decay_factor * oscillation
                
            velocity_profile[i] = max(min_velocity, velocity)
        
        # Normalize to ensure correct mean
        actual_mean = np.mean(velocity_profile)
        velocity_profile = velocity_profile * (mean_velocity / actual_mean)
        
        return time_points, velocity_profile

def create_pulsatile_boundary_conditions(vessel_file: str, original_bc_file: str, output_file: str, heart_rate: float = 75.0) -> Dict:
    """Create comprehensive pulsatile boundary conditions from scratch"""
    
    print(f"Creating pulsatile BC for {Path(vessel_file).stem}...")
    
    # Load original geometry data
    with open(original_bc_file, 'r') as f:
        vessel_geometry = json.load(f)
    
    # Create cardiac parameters
    cardiac_params = CardiacCycleParameters(heart_rate_bpm=heart_rate)
    blood_props = BloodProperties()
    flow_generator = PulsatileFlowGenerator(cardiac_params)
    
    # Extract geometry
    inlet_area = vessel_geometry['inlet_conditions']['cross_sectional_area_m2']
    inlet_direction = vessel_geometry['inlet_conditions']['velocity_direction']
    hydraulic_diameter = vessel_geometry['inlet_conditions']['hydraulic_diameter_m']
    
    # Calculate flow parameters
    mean_flow_rate_ml_s = 5.0  # ml/s
    mean_velocity = (mean_flow_rate_ml_s * 1e-6) / inlet_area  # m/s
    
    # Generate pulsatile profiles
    time_points, velocity_profile = flow_generator.generate_velocity_profile(mean_velocity)
    
    # Calculate Reynolds number
    reynolds_number = (blood_props.density * mean_velocity * hydraulic_diameter) / blood_props.dynamic_viscosity
    
    # Create pressure profile (simplified)
    mean_pressure_pa = 10665.76  # 80 mmHg
    pressure_amplitude = 5333.0  # 40 mmHg
    pressure_profile = mean_pressure_pa + pressure_amplitude * np.sin(2 * np.pi * time_points / flow_generator.cycle_duration)
    
    # Compile boundary conditions
    pulsatile_bc = {
        'metadata': {
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vessel_file': vessel_file,
            'boundary_condition_type': 'pulsatile_comprehensive',
            'heart_rate_bpm': heart_rate,
            'cycle_duration_s': flow_generator.cycle_duration
        },
        
        'blood_properties': {
            'density_kg_m3': blood_props.density,
            'dynamic_viscosity_pa_s': blood_props.dynamic_viscosity,
            'kinematic_viscosity_m2_s': blood_props.dynamic_viscosity / blood_props.density,
            'specific_heat_j_kg_k': blood_props.specific_heat,
            'thermal_conductivity_w_m_k': blood_props.thermal_conductivity
        },
        
        'inlet_conditions': {
            'boundary_type': 'velocity_inlet_pulsatile',
            'time_points': time_points.tolist(),
            'velocity_magnitude': velocity_profile.tolist(),
            'velocity_direction': inlet_direction,
            'mean_velocity_ms': float(mean_velocity),
            'peak_velocity_ms': float(np.max(velocity_profile)),
            'reynolds_number': float(reynolds_number),
            'turbulence_intensity': 0.05,
            'hydraulic_diameter': hydraulic_diameter,
            'cross_sectional_area_m2': inlet_area,
            'temperature': 310.15  # 37°C
        },
        
        'outlet_conditions': {
            'boundary_type': 'pressure_outlet_pulsatile',
            'time_points': time_points.tolist(),
            'gauge_pressure': (pressure_profile - 101325).tolist(),  # Gauge pressure
            'mean_pressure_pa': float(mean_pressure_pa),
            'pressure_amplitude_pa': float(pressure_amplitude),
            'backflow_turbulence_intensity': 0.05,
            'backflow_temperature': 310.15
        },
        
        'wall_conditions': {
            'boundary_type': 'wall',
            'wall_motion': 'stationary',
            'shear_condition': 'no_slip',
            'wall_roughness': 0.0,
            'wall_temperature': 310.15,
            'heat_flux': 0.0
        },
        
        'solver_settings': {
            'solver_type': 'pressure_based',
            'time_formulation': 'unsteady_second_order_implicit',
            'turbulence_model': 'k_omega_sst',
            'time_step_size_s': 0.001,
            'max_iterations_per_time_step': 20,
            'convergence_criteria': {
                'continuity': 1e-4,
                'momentum': 1e-4,
                'turbulence': 1e-4
            }
        },
        
        'original_geometry_data': vessel_geometry
    }
    
    # Save boundary conditions
    with open(output_file, 'w') as f:
        json.dump(pulsatile_bc, f, indent=2, default=str)
    
    print(f"  ✓ Saved: {output_file}")
    print(f"  • Heart rate: {heart_rate} BPM, Cycle: {flow_generator.cycle_duration:.3f}s")
    print(f"  • Mean velocity: {mean_velocity:.3f} m/s, Peak: {np.max(velocity_profile):.3f} m/s")
    print(f"  • Reynolds number: {reynolds_number:.0f}")
    
    return pulsatile_bc

def main():
    parser = argparse.ArgumentParser(description='Create Pulsatile Boundary Conditions')
    parser.add_argument('--vessel-dir', default='~/urp/data/uan/clean_flat_vessels')
    parser.add_argument('--output-dir', default='~/urp/data/uan/pulsatile_boundary_conditions')
    parser.add_argument('--heart-rate', type=float, default=75.0)
    parser.add_argument('--patient-limit', type=int, default=5)
    
    args = parser.parse_args()
    
    print(f"🫀 Creating Pulsatile Boundary Conditions from Scratch")
    print(f"{'='*60}")
    print(f"Heart rate: {args.heart_rate} BPM")
    
    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find vessel files
    vessel_dir = Path(os.path.expanduser(args.vessel_dir))
    vessel_files = []
    
    for stl_file in vessel_dir.glob("*_clean_flat.stl"):
        patient_id = stl_file.stem.replace('_clean_flat', '')
        original_bc_file = stl_file.parent / f"{patient_id}_boundary_conditions.json"
        
        if original_bc_file.exists():
            vessel_files.append((str(stl_file), str(original_bc_file), patient_id))
        
        if len(vessel_files) >= args.patient_limit:
            break
    
    print(f"📊 Found {len(vessel_files)} patients")
    
    # Process patients
    start_time = time.time()
    results = []
    
    for i, (vessel_file, original_bc_file, patient_id) in enumerate(vessel_files):
        print(f"\n📈 Progress: {i+1}/{len(vessel_files)} - {patient_id}")
        
        try:
            output_file = os.path.join(output_dir, f"{patient_id}_pulsatile_bc.json")
            bc_data = create_pulsatile_boundary_conditions(vessel_file, original_bc_file, output_file, args.heart_rate)
            
            results.append({
                'patient_id': patient_id,
                'success': True,
                'output_file': output_file,
                'mean_velocity_ms': bc_data['inlet_conditions']['mean_velocity_ms'],
                'reynolds_number': bc_data['inlet_conditions']['reynolds_number']
            })
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'patient_id': patient_id, 'success': False, 'error': str(e)})
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    
    print(f"\n{'='*60}")
    print(f"🎯 PULSATILE BOUNDARY CONDITIONS COMPLETE")
    print(f"📊 Successful: {len(successful)}/{len(results)}")
    print(f"⏱️ Time: {total_time:.1f} seconds")
    
    if successful:
        velocities = [r['mean_velocity_ms'] for r in successful]
        reynolds = [r['reynolds_number'] for r in successful]
        print(f"🌊 Velocity range: {np.min(velocities):.3f} - {np.max(velocities):.3f} m/s")
        print(f"🔢 Reynolds range: {np.min(reynolds):.0f} - {np.max(reynolds):.0f}")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'pulsatile_bc_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({'metadata': {'heart_rate_bpm': args.heart_rate, 'total_patients': len(results), 'successful': len(successful)}, 'results': results}, f, indent=2, default=str)
    
    print(f"📁 Summary: {summary_file}")
    print(f"🎉 Ready for PyAnsys CFD with 32 cores!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 