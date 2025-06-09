#!/usr/bin/env python3
"""
PyFluent Batch CFD Analyzer
Author: Jiwoo Lee

This script performs comprehensive CFD analysis on aneurysm geometries using
Ansys Fluent through the PyFluent interface. It processes STL files with 
corresponding boundary conditions and exports VTP results.

Data source: ~/urp/data/uan/clean_flat_vessels/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# PyFluent imports
try:
    import ansys.fluent.core as pyfluent
    PYFLUENT_AVAILABLE = True
    print(f"✅ PyFluent {pyfluent.__version__} loaded successfully")
except ImportError as e:
    PYFLUENT_AVAILABLE = False
    print(f"❌ PyFluent not available: {e}")

# VTK/PyVista for export
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("❌ PyVista not available for VTP export")

class PyFluentBatchAnalyzer:
    """
    Comprehensive PyFluent CFD batch analyzer for aneurysm geometries.
    """
    
    def __init__(self, data_dir: str, output_dir: str, n_cores: int = 16):
        """
        Initialize the batch analyzer.
        
        Args:
            data_dir: Directory containing STL files and boundary conditions
            output_dir: Output directory for results
            n_cores: Number of CPU cores for parallel processing
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_cores = n_cores
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # PyFluent session
        self.session = None
        
        # Results tracking
        self.results_summary = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger('PyFluentAnalyzer')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.output_dir / f"pyfluent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def find_analysis_cases(self) -> List[Tuple[Path, Path]]:
        """
        Find all STL files with corresponding boundary condition files.
        
        Returns:
            List of tuples (stl_file, bc_file)
        """
        cases = []
        
        # Find all STL files
        stl_files = list(self.data_dir.glob("*_clean_flat.stl"))
        
        for stl_file in stl_files:
            # Find corresponding boundary condition file
            base_name = stl_file.stem.replace("_clean_flat", "")
            bc_file = self.data_dir / f"{base_name}_boundary_conditions.json"
            
            if bc_file.exists():
                cases.append((stl_file, bc_file))
                self.logger.info(f"Found case: {base_name}")
            else:
                self.logger.warning(f"No boundary conditions for {stl_file.name}")
        
        self.logger.info(f"Found {len(cases)} analysis cases")
        return cases
    
    def load_boundary_conditions(self, bc_file: Path) -> Dict:
        """Load boundary conditions from JSON file."""
        try:
            with open(bc_file, 'r') as f:
                bc_data = json.load(f)
            return bc_data
        except Exception as e:
            self.logger.error(f"Failed to load boundary conditions from {bc_file}: {e}")
            return {}
    
    def launch_pyfluent_session(self) -> bool:
        """
        Launch PyFluent solver session.
        
        Returns:
            True if successful, False otherwise
        """
        if not PYFLUENT_AVAILABLE:
            self.logger.error("PyFluent not available")
            return False
        
        try:
            self.logger.info(f"🚀 Launching PyFluent solver with {self.n_cores} cores...")
            
            self.session = pyfluent.launch_fluent(
                precision="double",
                processor_count=self.n_cores,
                mode="solver",
                dimension=3,
                product_version=pyfluent.FluentVersion.v251,
                start_timeout=300,
                cleanup_on_exit=True,
                ui_mode="no_gui"
            )
            
            self.logger.info("✅ PyFluent session launched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch PyFluent session: {e}")
            return False
    
    def setup_physics_models(self, bc_data: Dict):
        """Setup physics models based on boundary conditions."""
        try:
            # Enable viscous flow
            self.session.setup.models.viscous.model = "laminar"
            
            # Set fluid properties
            fluid_props = bc_data.get('fluid_properties', {})
            density = fluid_props.get('density', 1060.0)  # kg/m³
            viscosity = fluid_props.get('dynamic_viscosity', 0.004)  # Pa·s
            
            # Set material properties
            self.session.setup.materials.fluid["water-liquid"].density.constant = density
            self.session.setup.materials.fluid["water-liquid"].viscosity.constant = viscosity
            
            # Set operating conditions
            self.session.setup.operating_conditions.operating_pressure = 101325  # Pa
            
            self.logger.info("✅ Physics models configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup physics models: {e}")
    
    def apply_boundary_conditions(self, bc_data: Dict):
        """Apply boundary conditions from the JSON data."""
        try:
            # Inlet conditions
            inlet_data = bc_data.get('inlet_conditions', {})
            if inlet_data:
                inlet_velocity = inlet_data.get('velocity_magnitude_m_s', 0.5)  # m/s
                
                # Apply velocity inlet
                self.session.setup.boundary_conditions.velocity_inlet["inlet"] = {
                    "velocity_magnitude": inlet_velocity,
                    "turbulent_intensity": 0.05,
                    "hydraulic_diameter": inlet_data.get('hydraulic_diameter_m', 0.004)
                }
            
            # Outlet conditions  
            outlet_data = bc_data.get('outlet_conditions', {})
            if outlet_data:
                outlet_pressure = outlet_data.get('pressure_pa', 10665.76)  # Pa
                
                # Apply pressure outlet
                self.session.setup.boundary_conditions.pressure_outlet["outlet"] = {
                    "pressure": outlet_pressure
                }
            
            # Wall conditions (no-slip)
            self.session.setup.boundary_conditions.wall["wall"] = {
                "wall_motion": "stationary_wall"
            }
            
            self.logger.info("✅ Boundary conditions applied")
            
        except Exception as e:
            self.logger.error(f"Failed to apply boundary conditions: {e}")
    
    def run_cfd_simulation(self, case_name: str, max_iterations: int = 500) -> bool:
        """
        Run CFD simulation.
        
        Args:
            case_name: Name of the case
            max_iterations: Maximum number of iterations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"🔄 Running CFD simulation for {case_name}...")
            
            # Initialize solution
            self.session.solution.initialization.hybrid_initialize()
            
            # Set solution parameters
            self.session.solution.run_calculation.iter_count = max_iterations
            
            # Enable residual monitoring
            self.session.solution.monitor.residual.plot = True
            
            # Run calculation
            self.session.solution.run_calculation.iterate()
            
            self.logger.info(f"✅ CFD simulation completed for {case_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to run CFD simulation for {case_name}: {e}")
            return False
    
    def export_results_to_vtp(self, case_name: str) -> Optional[str]:
        """
        Export CFD results to VTP format.
        
        Args:
            case_name: Name of the case
            
        Returns:
            Path to exported VTP file or None if failed
        """
        try:
            vtp_file = self.output_dir / f"{case_name}_results.vtp"
            
            # Export surface data
            self.session.file.export.ensight.gold.surfaces_list = ["wall", "inlet", "outlet"]
            self.session.file.export.ensight.gold.variables_list = [
                "pressure", "velocity_magnitude", "wall_shear_stress"
            ]
            
            # Export to EnSight format first
            ensight_file = self.output_dir / f"{case_name}_results"
            self.session.file.export.ensight.gold.filename = str(ensight_file)
            self.session.file.export.ensight.gold()
            
            # Convert to VTP if PyVista is available
            if PYVISTA_AVAILABLE:
                try:
                    # Load EnSight data and convert to VTP
                    mesh = pv.read(f"{ensight_file}.case")
                    mesh.save(str(vtp_file))
                    self.logger.info(f"✅ VTP file exported: {vtp_file}")
                    return str(vtp_file)
                except Exception as e:
                    self.logger.warning(f"PyVista conversion failed: {e}")
            
            self.logger.info(f"✅ EnSight files exported: {ensight_file}")
            return str(ensight_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export results for {case_name}: {e}")
            return None
    
    def analyze_single_case(self, stl_file: Path, bc_file: Path) -> Dict:
        """
        Analyze a single case.
        
        Args:
            stl_file: Path to STL geometry file
            bc_file: Path to boundary conditions file
            
        Returns:
            Dictionary with analysis results
        """
        case_name = stl_file.stem.replace("_clean_flat", "")
        start_time = time.time()
        
        result = {
            'case_name': case_name,
            'stl_file': str(stl_file),
            'bc_file': str(bc_file),
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error_message': None,
            'runtime_seconds': 0,
            'output_files': []
        }
        
        try:
            self.logger.info(f"🔍 Analyzing case: {case_name}")
            
            # Load boundary conditions
            bc_data = self.load_boundary_conditions(bc_file)
            if not bc_data:
                raise ValueError("Failed to load boundary conditions")
            
            # Import mesh
            self.logger.info(f"📁 Importing mesh: {stl_file}")
            self.session.file.read_mesh(file_name=str(stl_file))
            
            # Setup physics and boundary conditions
            self.setup_physics_models(bc_data)
            self.apply_boundary_conditions(bc_data)
            
            # Run simulation
            simulation_success = self.run_cfd_simulation(case_name)
            if not simulation_success:
                raise RuntimeError("CFD simulation failed")
            
            # Export results
            output_file = self.export_results_to_vtp(case_name)
            if output_file:
                result['output_files'].append(output_file)
            
            # Save case file
            case_file = self.output_dir / f"{case_name}.cas.h5"
            self.session.file.write_case(file_name=str(case_file))
            result['output_files'].append(str(case_file))
            
            result['success'] = True
            self.logger.info(f"✅ Case {case_name} completed successfully")
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"❌ Case {case_name} failed: {e}")
        
        result['runtime_seconds'] = time.time() - start_time
        result['end_time'] = datetime.now().isoformat()
        
        return result
    
    def run_batch_analysis(self, max_cases: Optional[int] = None) -> List[Dict]:
        """
        Run batch analysis on all available cases.
        
        Args:
            max_cases: Maximum number of cases to process (None for all)
            
        Returns:
            List of analysis results
        """
        self.logger.info("🚀 Starting PyFluent batch analysis")
        
        # Find all cases
        cases = self.find_analysis_cases()
        if max_cases:
            cases = cases[:max_cases]
        
        if not cases:
            self.logger.error("No analysis cases found")
            return []
        
        # Launch PyFluent session
        if not self.launch_pyfluent_session():
            self.logger.error("Failed to launch PyFluent session")
            return []
        
        # Process all cases
        results = []
        
        try:
            for i, (stl_file, bc_file) in enumerate(tqdm(cases, desc="Processing cases")):
                self.logger.info(f"Processing case {i+1}/{len(cases)}")
                
                result = self.analyze_single_case(stl_file, bc_file)
                results.append(result)
                
                # Save intermediate results
                self.save_results_summary(results)
        
        except KeyboardInterrupt:
            self.logger.info("Analysis interrupted by user")
        
        finally:
            # Close PyFluent session
            if self.session:
                try:
                    self.session.exit()
                    self.logger.info("PyFluent session closed")
                except:
                    pass
        
        # Save final results
        self.save_results_summary(results)
        self.generate_analysis_report(results)
        
        return results
    
    def save_results_summary(self, results: List[Dict]):
        """Save results summary to JSON file."""
        summary_file = self.output_dir / "batch_analysis_results.json"
        
        summary = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'total_cases': len(results),
                'successful_cases': sum(1 for r in results if r['success']),
                'failed_cases': sum(1 for r in results if not r['success']),
                'data_directory': str(self.data_dir),
                'output_directory': str(self.output_dir),
                'n_cores': self.n_cores
            },
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results summary saved: {summary_file}")
    
    def generate_analysis_report(self, results: List[Dict]):
        """Generate a comprehensive analysis report."""
        report_file = self.output_dir / "analysis_report.md"
        
        successful_cases = [r for r in results if r['success']]
        failed_cases = [r for r in results if not r['success']]
        
        with open(report_file, 'w') as f:
            f.write("# PyFluent Batch CFD Analysis Report\n\n")
            f.write(f"**Author:** Jiwoo Lee\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Cases:** {len(results)}\n")
            f.write(f"- **Successful:** {len(successful_cases)}\n")
            f.write(f"- **Failed:** {len(failed_cases)}\n")
            f.write(f"- **Success Rate:** {len(successful_cases)/len(results)*100:.1f}%\n\n")
            
            if successful_cases:
                f.write("## Successful Cases\n\n")
                for result in successful_cases:
                    f.write(f"### {result['case_name']}\n")
                    f.write(f"- **Runtime:** {result['runtime_seconds']:.1f} seconds\n")
                    f.write(f"- **Output Files:** {len(result['output_files'])}\n")
                    for output_file in result['output_files']:
                        f.write(f"  - `{Path(output_file).name}`\n")
                    f.write("\n")
            
            if failed_cases:
                f.write("## Failed Cases\n\n")
                for result in failed_cases:
                    f.write(f"### {result['case_name']}\n")
                    f.write(f"- **Error:** {result['error_message']}\n\n")
        
        self.logger.info(f"Analysis report generated: {report_file}")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="PyFluent Batch CFD Analyzer for Aneurysm Geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        default="/home/jiwoo/urp/data/uan/clean_flat_vessels",
        help="Directory containing STL files and boundary conditions"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--n-cores",
        type=int,
        default=16,
        help="Number of CPU cores for parallel processing"
    )
    
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🔬 PyFluent Batch CFD Analyzer")
    print("=" * 50)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"CPU Cores: {args.n_cores}")
    print(f"Max Cases: {args.max_cases or 'All'}")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PyFluentBatchAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_cores=args.n_cores
    )
    
    # Run analysis
    results = analyzer.run_batch_analysis(max_cases=args.max_cases)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ Analysis Complete!")
    print(f"Successful: {successful}/{len(results)} cases")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 