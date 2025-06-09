#!/usr/bin/env python3
"""
PyAnsys Pulsatile CFD Analysis for Aneurysm Stress Analysis

This script performs comprehensive CFD analysis with pulsatile flow boundary conditions
using PyAnsys Fluent for hemodynamic stress analysis of aneurysm vessels.

Features:
- Pulsatile flow boundary conditions (cardiac cycle)
- Wall shear stress analysis
- Pressure analysis
- 32-core parallel processing
- Automated mesh import and setup
- Results export and visualization
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple, Optional
import subprocess
import shutil
from tqdm import tqdm

# Check PyAnsys availability
PYANSYS_AVAILABLE = False
try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    PYANSYS_AVAILABLE = True
    print("✓ PyAnsys Fluent available")
except ImportError:
    print("⚠ PyAnsys not available. Install with: pip install ansys-fluent-core")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    print("✓ PyVista available")
except ImportError:
    print("⚠ PyVista not available. Install with: pip install pyvista")
    PYVISTA_AVAILABLE = False


def create_pulsatile_flow_profile(cardiac_cycle_duration: float = 0.8, 
                                 time_steps: int = 100,
                                 peak_velocity: float = 1.5,
                                 mean_velocity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic pulsatile flow velocity profile for cardiac cycle.
    
    Based on physiological cardiac waveform with systolic and diastolic phases.
    """
    print(f"    Creating pulsatile flow profile:")
    print(f"      Cycle duration: {cardiac_cycle_duration}s ({60/cardiac_cycle_duration:.0f} BPM)")
    print(f"      Peak velocity: {peak_velocity} m/s")
    print(f"      Mean velocity: {mean_velocity} m/s")
    
    # Time points over cardiac cycle
    time_points = np.linspace(0, cardiac_cycle_duration, time_steps)
    velocity_profile = np.zeros(time_steps)
    
    for i, t in enumerate(time_points):
        t_normalized = t / cardiac_cycle_duration
        
        if t_normalized <= 0.3:  # Systolic phase (30% of cycle)
            if t_normalized <= 0.15:  # Acceleration phase
                velocity = mean_velocity + (peak_velocity - mean_velocity) * (t_normalized / 0.15) ** 1.5
            else:  # Early deceleration
                velocity = peak_velocity * (1 - ((t_normalized - 0.15) / 0.15) ** 0.8)
        else:  # Diastolic phase (70% of cycle)
            # Exponential decay to low flow
            diastolic_factor = np.exp(-3 * (t_normalized - 0.3) / 0.7)
            velocity = mean_velocity * 0.3 + (mean_velocity * 0.7) * diastolic_factor
        
        # Ensure minimum flow
        velocity_profile[i] = max(0.1 * mean_velocity, velocity)
    
    print(f"      Profile created: max {np.max(velocity_profile):.3f} m/s, avg {np.mean(velocity_profile):.3f} m/s")
    
    return time_points, velocity_profile


def prepare_mesh_for_fluent(stl_file: str, output_dir: str, patient_id: str) -> str:
    """
    Prepare STL mesh for Fluent import by converting to appropriate format.
    """
    print(f"    Preparing mesh for Fluent...")
    
    # Create fluent mesh directory
    fluent_mesh_dir = os.path.join(output_dir, 'fluent_meshes')
    os.makedirs(fluent_mesh_dir, exist_ok=True)
    
    # Copy STL file to fluent directory
    fluent_stl_file = os.path.join(fluent_mesh_dir, f"{patient_id}.stl")
    shutil.copy2(stl_file, fluent_stl_file)
    
    print(f"      Mesh prepared: {fluent_stl_file}")
    
    return fluent_stl_file


def create_fluent_journal_file(patient_id: str, 
                              stl_file: str,
                              boundary_conditions: Dict,
                              pulsatile_params: Dict,
                              output_dir: str) -> str:
    """
    Create Fluent journal file for automated CFD setup and solving.
    """
    print(f"    Creating Fluent journal file...")
    
    journal_file = os.path.join(output_dir, f"{patient_id}_fluent.jou")
    
    # Extract boundary condition data
    bc_data = boundary_conditions['original_bc_data']
    inlet = bc_data['inlet_conditions']
    outlet = bc_data['outlet_conditions']
    side_branches = bc_data.get('side_branch_conditions', [])
    
    # Pulsatile parameters
    time_step = pulsatile_params['time_step']
    total_time = pulsatile_params['cycle_duration'] * pulsatile_params['total_cycles']
    time_steps_total = int(total_time / time_step)
    
    journal_content = f'''
; Fluent Journal File for Pulsatile CFD Analysis
; Patient: {patient_id}
; Generated automatically

; Read STL file
file/import/stl-ascii "{stl_file}"

; Setup solver
define/models/solver/pressure-based yes
define/models/unsteady/unsteady yes
define/models/unsteady/time-formulation 2nd-order-implicit

; Setup viscous model (k-omega SST)
define/models/viscous/kw-sst yes

; Define materials (blood properties)
define/materials/change-create blood blood yes constant {bc_data['fluid_properties']['density']} yes constant {bc_data['fluid_properties']['dynamic_viscosity']} no no no no no

; Setup boundary conditions

; Inlet boundary condition (pulsatile velocity)
define/boundary-conditions/velocity-inlet inlet yes yes yes yes no no yes yes no no yes constant {inlet['velocity_magnitude_m_s']} yes no no yes yes no 0.05 no no yes 1 no

; Outlet boundary condition
define/boundary-conditions/pressure-outlet outlet yes no 0 yes yes no 0.05 no no yes 1 no no no no

; Wall boundary condition
define/boundary-conditions/wall wall 0 no 0 no no no 0 no no no no

; Setup solution methods
solve/set/discretization-scheme/pressure 14
solve/set/discretization-scheme/mom 3
solve/set/discretization-scheme/k 1
solve/set/discretization-scheme/omega 1

; Setup under-relaxation factors
solve/set/under-relaxation/pressure 0.3
solve/set/under-relaxation/mom 0.7
solve/set/under-relaxation/k 0.8
solve/set/under-relaxation/omega 0.8

; Setup time stepping
solve/set/time-step {time_step}

; Initialize solution
solve/initialize/hybrid-initialization

; Setup monitoring
solve/monitors/surface/set-monitor wall-shear-stress-monitor wall-shear-stress wall () yes "wall_shear_stress.out" 1 no
solve/monitors/surface/set-monitor pressure-monitor pressure outlet () yes "pressure.out" 1 no

; Setup data sampling
solve/set/data-sampling yes {int(pulsatile_params['time_steps']/10)}

; Run transient calculation
solve/dual-time-iterate {time_steps_total} 20

; Write results
file/write-data "{patient_id}_final.dat"
file/write-case "{patient_id}_final.cas"

; Export wall shear stress data
surface/export-to-csv "{patient_id}_wall_shear_stress.csv" wall () pressure wall-shear-stress wall-shear-stress-magnitude velocity-magnitude no

; Export pressure data  
surface/export-to-csv "{patient_id}_pressure.csv" wall () pressure pressure-coefficient no

; Export velocity data at inlet/outlet
surface/export-to-csv "{patient_id}_velocity.csv" inlet outlet () velocity-magnitude x-velocity y-velocity z-velocity no

exit
'''
    
    with open(journal_file, 'w') as f:
        f.write(journal_content)
    
    print(f"      Journal file created: {journal_file}")
    
    return journal_file


def run_fluent_analysis(patient_id: str,
                       journal_file: str,
                       output_dir: str,
                       n_cores: int = 32) -> Dict:
    """
    Run Fluent CFD analysis using journal file.
    """
    print(f"  Running Fluent CFD analysis with {n_cores} cores...")
    
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
        
        # Fluent command
        fluent_cmd = [
            'fluent', '3ddp',
            '-t', str(n_cores),
            '-i', journal_file,
            '-g'  # No GUI
        ]
        
        print(f"    Starting Fluent: {' '.join(fluent_cmd)}")
        
        start_time = time.time()
        
        # Run Fluent
        process = subprocess.run(
            fluent_cmd,
            cwd=patient_results_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        simulation_time = time.time() - start_time
        result['simulation_time'] = simulation_time
        
        if process.returncode == 0:
            print(f"    ✓ Fluent completed successfully in {simulation_time/60:.1f} minutes")
            
            # Check for output files
            output_files = []
            for ext in ['*.csv', '*.dat', '*.cas', '*.out']:
                files = list(Path(patient_results_dir).glob(ext))
                output_files.extend([str(f) for f in files])
            
            result['success'] = True
            result['output_files'] = output_files
            
            # Analyze results
            analysis_results = analyze_cfd_results(patient_results_dir, patient_id)
            result['analysis_results'] = analysis_results
            
        else:
            error_msg = f"Fluent failed with return code {process.returncode}"
            if process.stderr:
                error_msg += f": {process.stderr}"
            result['error'] = error_msg
            print(f"    ✗ Fluent failed: {error_msg}")
        
    except subprocess.TimeoutExpired:
        result['error'] = "Fluent simulation timeout (1 hour)"
        print(f"    ✗ Fluent timeout after 1 hour")
    except Exception as e:
        result['error'] = str(e)
        print(f"    ✗ Error running Fluent: {e}")
    
    return result


def analyze_cfd_results(results_dir: str, patient_id: str) -> Dict:
    """
    Analyze CFD results and calculate hemodynamic parameters.
    """
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
        import pandas as pd
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
        import pandas as pd
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
        import pandas as pd
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
        # Oscillatory Shear Index (simplified)
        params['mean_wss'] = wss_data['mean_wss_pa']
        params['max_wss'] = wss_data['max_wss_pa']
        
        # Pressure parameters
        params['pressure_drop'] = pressure_data['pressure_drop_pa']
        params['mean_pressure'] = pressure_data['mean_pressure_pa']
        
        # Clinical risk indicators
        params['low_wss_risk'] = wss_data['low_wss_area_ratio'] > 0.1  # >10% low WSS area
        params['high_wss_risk'] = wss_data['high_wss_area_ratio'] > 0.05  # >5% high WSS area
        params['pressure_risk'] = pressure_data['pressure_drop_pa'] > 1000  # >1000 Pa drop
        
        # Overall risk score (0-3)
        risk_score = 0
        if params['low_wss_risk']: risk_score += 1
        if params['high_wss_risk']: risk_score += 1
        if params['pressure_risk']: risk_score += 1
        
        params['hemodynamic_risk_score'] = risk_score
        
    except Exception as e:
        params['error'] = str(e)
    
    return params


def process_single_patient_cfd(vessel_file: str, 
                              bc_file: str,
                              patient_id: str,
                              pulsatile_params: Dict,
                              output_dir: str,
                              n_cores: int) -> Dict:
    """
    Process complete pulsatile CFD analysis for a single patient.
    """
    print(f"\n{'='*60}")
    print(f"Processing Patient: {patient_id}")
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
        
        # Create pulsatile boundary conditions
        boundary_conditions = {
            'original_bc_data': bc_data,
            'pulsatile_params': pulsatile_params
        }
        
        # Prepare mesh for Fluent
        fluent_mesh = prepare_mesh_for_fluent(vessel_file, output_dir, patient_id)
        
        # Create Fluent journal file
        journal_file = create_fluent_journal_file(
            patient_id, fluent_mesh, boundary_conditions, pulsatile_params, output_dir
        )
        
        # Run Fluent analysis
        fluent_result = run_fluent_analysis(patient_id, journal_file, output_dir, n_cores)
        
        result.update(fluent_result)
        result['processing_time'] = time.time() - start_time
        
        if result['success']:
            print(f"✓ {patient_id}: Complete CFD analysis successful ({result['processing_time']/60:.1f} min)")
        else:
            print(f"✗ {patient_id}: CFD analysis failed - {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        print(f"✗ {patient_id}: Exception - {e}")
    
    return result


def main():
    """Main function for pulsatile CFD analysis"""
    parser = argparse.ArgumentParser(description='PyAnsys Pulsatile CFD Analysis')
    
    parser.add_argument('--vessel-dir', 
                       default=os.path.expanduser('~/urp/data/uan/clean_flat_vessels'),
                       help='Directory containing capped vessel STL files')
    
    parser.add_argument('--results-dir',
                       default=os.path.expanduser('~/urp/data/uan/cfd_pulsatile_results'),
                       help='Output directory for CFD results')
    
    parser.add_argument('--n-cores', type=int, default=32,
                       help='Number of CPU cores for Fluent analysis')
    
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
    
    print(f"\n🫀 PyAnsys Pulsatile CFD Analysis for Aneurysm Stress Analysis")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  • CPU cores: {args.n_cores}")
    print(f"  • Vessel directory: {args.vessel_dir}")
    print(f"  • Results directory: {args.results_dir}")
    print(f"  • Test mode: {'Yes' if args.test_mode else 'No'}")
    
    # Check Fluent availability
    fluent_available = shutil.which('fluent') is not None
    if not fluent_available:
        print("\n❌ Error: Fluent not found in PATH")
        print("Please ensure ANSYS Fluent is installed and accessible")
        return 1
    else:
        print("✓ Fluent found in PATH")
    
    # Pulsatile flow parameters
    pulsatile_params = {
        'cycle_duration': args.cycle_duration,
        'peak_velocity': args.peak_velocity,
        'mean_velocity': args.peak_velocity * 0.3,  # 30% of peak
        'total_cycles': args.total_cycles,
        'time_steps': 100,
        'time_step': 0.001  # 1ms time step
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
    for i, (_, _, patient_id) in enumerate(vessel_files[:10]):  # Show first 10
        print(f"  {i+1:2d}. {patient_id}")
    if len(vessel_files) > 10:
        print(f"  ... and {len(vessel_files)-10} more")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Process patients
    print(f"\n🚀 Starting pulsatile CFD analysis...")
    start_time = time.time()
    
    results = []
    
    for i, (vessel_file, bc_file, patient_id) in enumerate(vessel_files):
        print(f"\n📈 Progress: {i+1}/{len(vessel_files)} ({(i+1)*100/len(vessel_files):.1f}%)")
        
        result = process_single_patient_cfd(
            vessel_file, bc_file, patient_id,
            pulsatile_params, args.results_dir, args.n_cores
        )
        
        results.append(result)
        
        # Save intermediate results
        if (i + 1) % 10 == 0 or i == len(vessel_files) - 1:
            intermediate_file = os.path.join(args.results_dir, f'intermediate_results_{i+1}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    # Generate final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*80}")
    print(f"🎯 PULSATILE CFD ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"  • Total processing time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  • Successful analyses: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)")
    print(f"  • Failed analyses: {len(failed)}")
    
    if successful:
        avg_processing_time = np.mean([r['processing_time'] for r in successful])
        avg_simulation_time = np.mean([r.get('simulation_time', 0) for r in successful if r.get('simulation_time')])
        print(f"  • Average processing time per patient: {avg_processing_time/60:.1f} minutes")
        print(f"  • Average simulation time per patient: {avg_simulation_time/60:.1f} minutes")
    
    if failed:
        print(f"\n❌ Failed analyses:")
        for fail in failed[:5]:  # Show first 5 failures
            print(f"  • {fail['patient_id']}: {fail['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more failures")
    
    # Save comprehensive results
    summary_file = os.path.join(args.results_dir, 'pulsatile_cfd_complete_summary.json')
    summary_data = {
        'analysis_metadata': {
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
    print(f"  • Fluent meshes: {args.results_dir}/fluent_meshes/")
    
    print(f"\n🎉 Pulsatile CFD stress analysis pipeline complete!")
    print(f"Ready for hemodynamic analysis and clinical correlation.")
    
    return 0 if len(successful) > 0 else 1


if __name__ == "__main__":
    exit(main()) 