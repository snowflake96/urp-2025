#!/usr/bin/env python3
"""
Real PyFluent CFD Analysis - Official ansys-fluent-core Implementation

Uses the official PyFluent library for actual Fluent CFD simulations
with proper mesh import, boundary conditions, and solver execution.
"""

import json
import os
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Dict, List
import subprocess

# Official PyFluent imports
try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    PYFLUENT_AVAILABLE = True
    print("✓ Official PyFluent (ansys-fluent-core) available")
except ImportError:
    PYFLUENT_AVAILABLE = False
    print("⚠ PyFluent not available - install with: pip install ansys-fluent-core")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

def setup_pyfluent_session(n_cores: int = 32, precision: str = "double"):
    """Launch PyFluent session with proper configuration"""
    
    print(f"🚀 Launching PyFluent session with {n_cores} cores...")
    
    try:
        # Launch Fluent with PyFluent (using working configuration)
        session = launch_fluent(
            dimension=3,  # Use integer for dimension
            product_version=pyfluent.FluentVersion.v251,  # Explicit version
            precision=precision,
            processor_count=n_cores,
            mode="solver",
            start_transcript=False,
            cleanup_on_exit=True,
            ui_mode="no_gui",  # Updated API
            additional_arguments="-t{}".format(n_cores)
        )
        
        print(f"✅ PyFluent session launched successfully")
        print(f"   Version: {session.get_fluent_version()}")
        print(f"   Cores: {n_cores}")
        print(f"   Precision: {precision}")
        
        return session
        
    except Exception as e:
        print(f"❌ Failed to launch PyFluent: {e}")
        return None

def import_mesh_to_fluent(session, stl_file: str):
    """Import STL mesh into Fluent using PyFluent"""
    
    print(f"📐 Importing mesh: {os.path.basename(stl_file)}")
    
    try:
        # Import STL file
        session.file.import_.cff_files(file_names=[stl_file])
        
        # Check mesh
        session.mesh.check()
        
        # Get mesh info
        mesh_info = session.mesh.get_info()
        print(f"   Nodes: {mesh_info.get('nodes', 'N/A')}")
        print(f"   Faces: {mesh_info.get('faces', 'N/A')}")
        print(f"   Cells: {mesh_info.get('cells', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mesh import failed: {e}")
        return False

def setup_physics_models(session, reynolds: float):
    """Setup physics models based on Reynolds number"""
    
    print(f"⚙️ Setting up physics models (Re = {reynolds:.0f})...")
    
    try:
        # Enable energy equation
        session.setup.models.energy.enabled = True
        
        # Setup viscous model based on Reynolds number
        if reynolds < 2000:
            # Laminar flow
            session.setup.models.viscous.model = "laminar"
            print("   Viscous model: Laminar")
        else:
            # Turbulent flow - k-omega SST
            session.setup.models.viscous.model = "k-omega-sst"
            print("   Viscous model: k-omega SST")
        
        # Setup solver
        session.solution.methods.p_v_coupling.setup.coupled_algorithm = "coupled"
        session.solution.methods.spatial_discretization.pressure = "presto"
        session.solution.methods.spatial_discretization.momentum = "bounded-central-differencing"
        
        # Unsteady solver for pulsatile flow
        session.solution.methods.unsteady_formulation.enabled = True
        session.solution.methods.unsteady_formulation.time_formulation = "second-order-implicit"
        
        print("   ✅ Physics models configured")
        return True
        
    except Exception as e:
        print(f"❌ Physics setup failed: {e}")
        return False

def setup_boundary_conditions(session, bc_data: dict):
    """Setup boundary conditions from pulsatile BC data"""
    
    print(f"🌊 Setting up boundary conditions...")
    
    try:
        # Blood properties
        blood_props = bc_data['blood_properties']
        
        # Create blood material
        session.setup.materials.fluid["blood"] = {
            "density": {"option": "constant", "value": blood_props['density_kg_m3']},
            "viscosity": {"option": "constant", "value": blood_props['dynamic_viscosity_pa_s']},
            "specific_heat": {"option": "constant", "value": blood_props['specific_heat_j_kg_k']},
            "thermal_conductivity": {"option": "constant", "value": blood_props['thermal_conductivity_w_m_k']}
        }
        
        # Assign blood to fluid zone
        fluid_zones = session.setup.boundary_conditions.get_zones(zone_type="fluid")
        if fluid_zones:
            session.setup.boundary_conditions.fluid[fluid_zones[0]].material = "blood"
        
        # Setup inlet boundary condition
        inlet_bc = bc_data['inlet_conditions']
        inlet_zones = session.setup.boundary_conditions.get_zones(zone_type="velocity-inlet")
        
        if inlet_zones:
            inlet = session.setup.boundary_conditions.velocity_inlet[inlet_zones[0]]
            
            # Velocity magnitude
            inlet.momentum.velocity.value = inlet_bc['mean_velocity_ms']
            
            # Velocity direction
            velocity_dir = inlet_bc['velocity_direction']
            inlet.momentum.velocity.direction = velocity_dir
            
            # Turbulence
            inlet.turbulence.turbulent_intensity = inlet_bc['turbulence_intensity']
            
            # Temperature
            inlet.thermal.temperature.value = inlet_bc['temperature']
            
            print(f"   Inlet velocity: {inlet_bc['mean_velocity_ms']:.3f} m/s")
            print(f"   Direction: [{velocity_dir[0]:.3f}, {velocity_dir[1]:.3f}, {velocity_dir[2]:.3f}]")
        
        # Setup outlet boundary condition
        outlet_bc = bc_data['outlet_conditions']
        outlet_zones = session.setup.boundary_conditions.get_zones(zone_type="pressure-outlet")
        
        if outlet_zones:
            outlet = session.setup.boundary_conditions.pressure_outlet[outlet_zones[0]]
            outlet.momentum.gauge_pressure.value = 0  # Reference pressure
            outlet.thermal.backflow_temperature.value = outlet_bc.get('backflow_temperature', 310.15)
            
            print(f"   Outlet: Pressure outlet (0 Pa gauge)")
        
        # Setup wall boundary conditions
        wall_bc = bc_data['wall_conditions']
        wall_zones = session.setup.boundary_conditions.get_zones(zone_type="wall")
        
        for wall_zone in wall_zones:
            wall = session.setup.boundary_conditions.wall[wall_zone]
            wall.momentum.shear_condition = "no-slip"
            wall.thermal.thermal_condition = "temperature"
            wall.thermal.temperature.value = wall_bc.get('wall_temperature', 310.15)
        
        print(f"   Walls: No-slip, T = {wall_bc.get('wall_temperature', 310.15)} K")
        print("   ✅ Boundary conditions set")
        
        return True
        
    except Exception as e:
        print(f"❌ Boundary condition setup failed: {e}")
        return False

def setup_solution_controls(session, bc_data: dict):
    """Setup solution controls for pulsatile flow"""
    
    print(f"🔧 Setting up solution controls...")
    
    try:
        # Time step settings
        time_step = bc_data['solver_settings']['time_step_size_s']
        cycle_duration = bc_data['metadata']['cycle_duration_s']
        
        session.solution.run_calculation.time_step_size = time_step
        session.solution.run_calculation.number_of_time_steps = int(cycle_duration / time_step)
        session.solution.run_calculation.max_iterations_per_time_step = bc_data['solver_settings']['max_iterations_per_time_step']
        
        # Convergence criteria
        convergence = bc_data['solver_settings']['convergence_criteria']
        session.solution.monitor.residual.options.criterion_type = "absolute"
        session.solution.monitor.residual.equations.continuity.absolute_criteria = convergence['continuity']
        session.solution.monitor.residual.equations.x_momentum.absolute_criteria = convergence['momentum']
        session.solution.monitor.residual.equations.y_momentum.absolute_criteria = convergence['momentum']
        session.solution.monitor.residual.equations.z_momentum.absolute_criteria = convergence['momentum']
        
        # Under-relaxation factors for stability
        session.solution.controls.under_relaxation.pressure = 0.3
        session.solution.controls.under_relaxation.momentum = 0.7
        session.solution.controls.under_relaxation.energy = 0.9
        
        print(f"   Time step: {time_step} s")
        print(f"   Steps per cycle: {int(cycle_duration / time_step)}")
        print(f"   Max iterations/step: {bc_data['solver_settings']['max_iterations_per_time_step']}")
        print("   ✅ Solution controls configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Solution controls setup failed: {e}")
        return False

def run_pulsatile_simulation(session, bc_data: dict, n_cycles: int = 3):
    """Run pulsatile CFD simulation"""
    
    print(f"🔄 Running pulsatile simulation ({n_cycles} cardiac cycles)...")
    
    try:
        # Initialize solution
        session.solution.initialization.hybrid_initialize()
        print("   ✅ Solution initialized")
        
        # Calculate total time steps
        time_step = bc_data['solver_settings']['time_step_size_s']
        cycle_duration = bc_data['metadata']['cycle_duration_s']
        steps_per_cycle = int(cycle_duration / time_step)
        total_steps = steps_per_cycle * n_cycles
        
        print(f"   Running {total_steps} time steps...")
        
        # Run transient calculation
        start_time = time.time()
        session.solution.run_calculation.dual_time_iterate(
            number_of_time_steps=total_steps,
            max_iterations_per_time_step=bc_data['solver_settings']['max_iterations_per_time_step']
        )
        
        calc_time = time.time() - start_time
        print(f"   ✅ Simulation completed in {calc_time:.1f} seconds")
        
        return True, calc_time
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False, 0

def extract_results(session, patient_id: str, output_dir: str):
    """Extract CFD results and export data"""
    
    print(f"📊 Extracting results...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export case and data files
        case_file = os.path.join(output_dir, f"{patient_id}_pyfluent.cas")
        data_file = os.path.join(output_dir, f"{patient_id}_pyfluent.dat")
        
        session.file.write_case_data(file_name=case_file.replace('.cas', ''))
        
        # Export results in various formats
        # EnSight format for visualization
        ensight_file = os.path.join(output_dir, f"{patient_id}_results")
        session.file.export.ensight_gold(
            file_name=ensight_file,
            surfaces_list=["wall"],
            variables_list=["pressure", "wall-shear-stress", "velocity-magnitude"]
        )
        
        # Export surface data
        wall_data_file = os.path.join(output_dir, f"{patient_id}_wall_data.csv")
        session.surface.export_to_csv(
            file_name=wall_data_file,
            surfaces_list=["wall"],
            variables_list=["x-coordinate", "y-coordinate", "z-coordinate", 
                          "pressure", "wall-shear-stress", "velocity-magnitude"]
        )
        
        print(f"   ✅ Case file: {case_file}")
        print(f"   ✅ Data file: {data_file}")
        print(f"   ✅ EnSight: {ensight_file}")
        print(f"   ✅ Wall data: {wall_data_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Results extraction failed: {e}")
        return False

def analyze_patient_with_pyfluent(patient_data: tuple) -> dict:
    """Complete PyFluent analysis for one patient"""
    
    vessel_file, bc_file, patient_id, output_dir, n_cores = patient_data
    
    print(f"\n{'='*70}")
    print(f"🔬 PyFluent Analysis: {patient_id}")
    print(f"{'='*70}")
    
    # Set environment variable for Ansys 2025 R1
    os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
    
    # Set environment variable for Ansys 2025 R1
    os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
    
    start_time = time.time()
    result = {'patient_id': patient_id, 'success': False, 'n_cores_used': n_cores}
    
    try:
        # Load boundary conditions
        with open(bc_file, 'r') as f:
            bc_data = json.load(f)
        
        reynolds = bc_data['inlet_conditions']['reynolds_number']
        velocity = bc_data['inlet_conditions']['mean_velocity_ms']
        
        print(f"📋 Patient: {patient_id}")
        print(f"   Reynolds: {reynolds:.0f}")
        print(f"   Velocity: {velocity:.3f} m/s")
        print(f"   Cores: {n_cores}")
        
        if not PYFLUENT_AVAILABLE:
            print("❌ PyFluent not available - install ansys-fluent-core")
            result['error'] = "PyFluent not available"
            return result
        
        # Launch PyFluent session
        session = setup_pyfluent_session(n_cores)
        if not session:
            result['error'] = "Failed to launch PyFluent"
            return result
        
        try:
            # Import mesh
            if not import_mesh_to_fluent(session, vessel_file):
                result['error'] = "Mesh import failed"
                return result
            
            # Setup physics
            if not setup_physics_models(session, reynolds):
                result['error'] = "Physics setup failed"
                return result
            
            # Setup boundary conditions
            if not setup_boundary_conditions(session, bc_data):
                result['error'] = "Boundary conditions setup failed"
                return result
            
            # Setup solution controls
            if not setup_solution_controls(session, bc_data):
                result['error'] = "Solution controls setup failed"
                return result
            
            # Run simulation
            sim_success, calc_time = run_pulsatile_simulation(session, bc_data)
            if not sim_success:
                result['error'] = "Simulation failed"
                return result
            
            # Extract results
            patient_output_dir = os.path.join(output_dir, patient_id)
            if not extract_results(session, patient_id, patient_output_dir):
                result['error'] = "Results extraction failed"
                return result
            
            # Success!
            processing_time = time.time() - start_time
            
            result.update({
                'success': True,
                'processing_time': processing_time,
                'calculation_time': calc_time,
                'analysis_mode': 'pyfluent_official',
                'boundary_conditions': {
                    'reynolds': reynolds,
                    'velocity_ms': velocity,
                    'heart_rate_bpm': bc_data['metadata']['heart_rate_bpm']
                },
                'simulation_info': {
                    'time_step_s': bc_data['solver_settings']['time_step_size_s'],
                    'cycles_simulated': 3,
                    'total_time_steps': int(bc_data['metadata']['cycle_duration_s'] * 3 / bc_data['solver_settings']['time_step_size_s'])
                },
                'output_files': {
                    'case_file': f"{patient_id}_pyfluent.cas",
                    'data_file': f"{patient_id}_pyfluent.dat",
                    'ensight_file': f"{patient_id}_results",
                    'wall_data': f"{patient_id}_wall_data.csv"
                }
            })
            
            print(f"✅ SUCCESS: {patient_id} completed in {processing_time:.1f}s")
            
        finally:
            # Clean up session
            try:
                session.exit()
                print("   🧹 PyFluent session closed")
            except:
                pass
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ ERROR: {patient_id} - {str(e)}")
        result.update({
            'error': str(e),
            'processing_time': processing_time
        })
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Real PyFluent CFD Analysis')
    parser.add_argument('--vessel-dir', default='~/urp/data/uan/clean_flat_vessels')
    parser.add_argument('--bc-dir', default='~/urp/data/uan/pulsatile_boundary_conditions')
    parser.add_argument('--results-dir', default='~/urp/data/uan/pyfluent_results')
    parser.add_argument('--n-cores', type=int, default=32)
    parser.add_argument('--patient-limit', type=int, default=6)
    parser.add_argument('--precision', choices=['single', 'double'], default='double')
    
    args = parser.parse_args()
    
    print(f"🚀 REAL PYFLUENT CFD ANALYSIS")
    print(f"{'='*70}")
    print(f"PyFluent available: {PYFLUENT_AVAILABLE}")
    print(f"Cores per analysis: {args.n_cores}")
    print(f"Precision: {args.precision}")
    print(f"Patient limit: {args.patient_limit}")
    print(f"{'='*70}")
    
    if not PYFLUENT_AVAILABLE:
        print("❌ PyFluent not available!")
        print("   Install with: pip install ansys-fluent-core")
        return 1
    
    # Find analysis files
    vessel_dir = Path(os.path.expanduser(args.vessel_dir))
    bc_dir = Path(os.path.expanduser(args.bc_dir))
    
    analysis_files = []
    bc_files = list(bc_dir.glob("*_pulsatile_bc.json"))
    
    for bc_file in sorted(bc_files)[:args.patient_limit]:
        patient_id = bc_file.stem.replace('_pulsatile_bc', '')
        stl_file = vessel_dir / f"{patient_id}_clean_flat.stl"
        
        if stl_file.exists():
            analysis_files.append((str(stl_file), str(bc_file), patient_id))
            print(f"  Found: {patient_id}")
    
    print(f"\n📋 Running PyFluent analysis for {len(analysis_files)} patients...")
    
    # Create results directory
    results_dir = os.path.expanduser(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Process patients sequentially (PyFluent sessions don't parallelize well)
    start_time = time.time()
    results = []
    
    for i, (vessel_file, bc_file, patient_id) in enumerate(analysis_files):
        print(f"\n📈 Progress: {i+1}/{len(analysis_files)} ({(i+1)*100/len(analysis_files):.1f}%)")
        
        patient_data = (vessel_file, bc_file, patient_id, results_dir, args.n_cores)
        result = analyze_patient_with_pyfluent(patient_data)
        results.append(result)
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r.get('success', False)]
    
    print(f"\n{'='*70}")
    print(f"🎯 PYFLUENT CFD ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Results:")
    print(f"  • Total time: {total_time/60:.1f} minutes")
    print(f"  • Successful: {len(successful)}/{len(results)}")
    print(f"  • Average per patient: {total_time/len(results)/60:.1f} minutes")
    print(f"  • Real CFD with PyFluent: ✅")
    
    if successful:
        avg_calc_time = np.mean([r.get('calculation_time', 0) for r in successful])
        total_time_steps = sum([r.get('simulation_info', {}).get('total_time_steps', 0) for r in successful])
        
        print(f"  • Average calculation time: {avg_calc_time:.1f} seconds")
        print(f"  • Total time steps computed: {total_time_steps:,}")
        print(f"  • Real Fluent solver used: ✅")
    
    # Save results
    summary_file = os.path.join(results_dir, 'pyfluent_analysis_summary.json')
    summary_data = {
        'metadata': {
            'analysis_type': 'pyfluent_official_cfd',
            'cores_per_patient': args.n_cores,
            'precision': args.precision,
            'total_patients': len(results),
            'successful': len(successful),
            'total_time_minutes': total_time / 60,
            'pyfluent_version': pyfluent.__version__ if PYFLUENT_AVAILABLE else None
        },
        'performance': {
            'average_calculation_time': np.mean([r.get('calculation_time', 0) for r in successful]) if successful else 0,
            'total_time_steps': sum([r.get('simulation_info', {}).get('total_time_steps', 0) for r in successful])
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved: {summary_file}")
    print(f"🚀 Real PyFluent CFD analysis complete!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 