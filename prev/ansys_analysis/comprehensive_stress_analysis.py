#!/usr/bin/env python3
"""
Comprehensive ANSYS Stress Analysis for All Patient Vessel Files
- Automated boundary condition generation for each anatomical region
- PyAnsys integration for finite element analysis
- Parallel processing for all patients using 16 CPUs
- Extract biomechanical features for aneurysm prediction
"""

import numpy as np
import nibabel as nib
import trimesh
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle

# PyAnsys imports
try:
    from ansys.mapdl.core import launch_mapdl
    from ansys.dpf import core as dpf
    import pyvista as pv
    ANSYS_AVAILABLE = True
except ImportError:
    ANSYS_AVAILABLE = False
    logging.warning("PyAnsys not available. Install with: pip install ansys-mapdl-core ansys-dpf-core")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveStressAnalysis:
    """Comprehensive ANSYS stress analysis for all patient vessel regions"""
    
    def __init__(self, ansys_license_server: Optional[str] = None):
        """
        Initialize comprehensive stress analyzer
        
        Args:
            ansys_license_server: ANSYS license server (e.g., '1055@license-server')
        """
        self.test_base_dir = Path("/home/jiwoo/urp/data/uan/test")
        self.results_dir = Path("/home/jiwoo/urp/data/uan/ansys_results")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        
        self.ansys_license_server = ansys_license_server
        self.patient_data_df = None
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Anatomical region properties
        self.setup_anatomical_properties()
        
    def setup_anatomical_properties(self):
        """Define material and boundary condition properties for each anatomical region"""
        
        # Material properties for different vessel types
        self.material_properties = {
            'healthy_vessel': {
                'young_modulus': 2.0e6,      # Pa (2.0 MPa)
                'poisson_ratio': 0.45,       # Nearly incompressible
                'density': 1050,             # kg/m³
                'wall_thickness': 0.0005,    # m (0.5 mm)
                'yield_strength': 0.8e6,     # Pa (0.8 MPa)
            },
            'aneurysm_wall': {
                'young_modulus': 1.0e6,      # Pa (50% reduction)
                'poisson_ratio': 0.45,       # Same as healthy
                'density': 1050,             # kg/m³  
                'wall_thickness': 0.0003,    # m (40% thinner)
                'yield_strength': 0.5e6,     # Pa (reduced strength)
            },
            'small_vessel': {
                'young_modulus': 1.5e6,      # Pa (intermediate)
                'poisson_ratio': 0.45,       
                'density': 1050,             
                'wall_thickness': 0.0003,    # m (thinner walls)
                'yield_strength': 0.6e6,     
            }
        }
        
        # Hemodynamic properties for different anatomical regions
        self.hemodynamic_properties = {
            'ACA': {
                'systolic_pressure': 14700,   # Pa (110 mmHg)
                'diastolic_pressure': 9800,   # Pa (73 mmHg)
                'mean_pressure': 11800,       # Pa
                'flow_velocity': 0.35,        # m/s
                'reynolds_number': 1200,
                'material_type': 'small_vessel'
            },
            'Acom': {
                'systolic_pressure': 15300,   # Pa (115 mmHg)
                'diastolic_pressure': 10100,  # Pa (76 mmHg)
                'mean_pressure': 12200,       # Pa
                'flow_velocity': 0.25,        # m/s (communicating vessel)
                'reynolds_number': 900,
                'material_type': 'aneurysm_wall'  # High risk region
            },
            'ICA (total)': {
                'systolic_pressure': 16000,   # Pa (120 mmHg)
                'diastolic_pressure': 10700,  # Pa (80 mmHg)
                'mean_pressure': 13300,       # Pa
                'flow_velocity': 0.5,         # m/s (high flow)
                'reynolds_number': 2000,
                'material_type': 'healthy_vessel'
            },
            'ICA (noncavernous)': {
                'systolic_pressure': 15700,   # Pa (118 mmHg)
                'diastolic_pressure': 10400,  # Pa (78 mmHg)
                'mean_pressure': 13000,       # Pa
                'flow_velocity': 0.45,        # m/s
                'reynolds_number': 1800,
                'material_type': 'aneurysm_wall'  # Thin-walled region
            },
            'ICA (cavernous)': {
                'systolic_pressure': 15700,   # Pa (118 mmHg)
                'diastolic_pressure': 10400,  # Pa (78 mmHg)
                'mean_pressure': 13000,       # Pa
                'flow_velocity': 0.4,         # m/s
                'reynolds_number': 1600,
                'material_type': 'healthy_vessel'
            },
            'Pcom': {
                'systolic_pressure': 15000,   # Pa (112 mmHg)
                'diastolic_pressure': 9900,   # Pa (74 mmHg)
                'mean_pressure': 12000,       # Pa
                'flow_velocity': 0.3,         # m/s
                'reynolds_number': 1100,
                'material_type': 'small_vessel'
            },
            'BA': {
                'systolic_pressure': 15500,   # Pa (116 mmHg)
                'diastolic_pressure': 10200,  # Pa (77 mmHg)
                'mean_pressure': 12500,       # Pa
                'flow_velocity': 0.4,         # m/s
                'reynolds_number': 1500,
                'material_type': 'healthy_vessel'
            },
            'Other_posterior': {
                'systolic_pressure': 14500,   # Pa (109 mmHg)
                'diastolic_pressure': 9600,   # Pa (72 mmHg)
                'mean_pressure': 11600,       # Pa
                'flow_velocity': 0.3,         # m/s
                'reynolds_number': 1000,
                'material_type': 'small_vessel'
            },
            'PCA': {
                'systolic_pressure': 14800,   # Pa (111 mmHg)
                'diastolic_pressure': 9800,   # Pa (73 mmHg)
                'mean_pressure': 11800,       # Pa
                'flow_velocity': 0.35,        # m/s
                'reynolds_number': 1200,
                'material_type': 'small_vessel'
            }
        }
    
    def load_patient_data(self) -> bool:
        """Load patient data from Excel file"""
        try:
            if not self.excel_path.exists():
                logger.error(f"Excel file not found: {self.excel_path}")
                return False
                
            self.patient_data_df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded patient data: {len(self.patient_data_df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return False
    
    def get_patient_info(self, patient_id: int) -> Dict[str, Any]:
        """Get patient information for stress analysis adjustments"""
        if self.patient_data_df is None:
            return {}
            
        patient_data = self.patient_data_df[self.patient_data_df['VintID'] == patient_id]
        
        if len(patient_data) == 0:
            return {}
            
        patient_row = patient_data.iloc[0]
        
        # Extract relevant patient information
        info = {
            'patient_id': patient_id,
            'age': patient_row.get('Age', 60),  # Default age if missing
            'gender': patient_row.get('Gender', 'Unknown'),
            'hypertension': patient_row.get('Hypertension', False),
        }
        
        # Adjust pressures based on patient factors
        if info['age'] > 65:
            info['pressure_factor'] = 1.1  # 10% higher pressure for elderly
        elif info['age'] < 40:
            info['pressure_factor'] = 0.95  # 5% lower for young patients
        else:
            info['pressure_factor'] = 1.0
            
        if info['hypertension']:
            info['pressure_factor'] *= 1.15  # 15% higher for hypertensive patients
            
        return info
    
    def generate_boundary_conditions(self, stl_file: Path, region: str, patient_info: Dict) -> Dict[str, Any]:
        """Generate boundary conditions for a specific STL file and anatomical region"""
        
        # Extract region name from filename (remove parentheses and normalize)
        region_key = region.replace('(', '').replace(')', '').replace(' ', ' ')
        
        # Find matching region in hemodynamic properties
        hemodynamic = None
        for key, props in self.hemodynamic_properties.items():
            if key.replace('(', '').replace(')', '').replace(' ', ' ') == region_key:
                hemodynamic = props.copy()
                break
        
        if hemodynamic is None:
            # Default properties for unknown regions
            hemodynamic = {
                'systolic_pressure': 15000,
                'diastolic_pressure': 10000,
                'mean_pressure': 12500,
                'flow_velocity': 0.4,
                'reynolds_number': 1500,
                'material_type': 'healthy_vessel'
            }
            logger.warning(f"Using default hemodynamic properties for region: {region}")
        
        # Apply patient-specific adjustments
        pressure_factor = patient_info.get('pressure_factor', 1.0)
        hemodynamic['systolic_pressure'] *= pressure_factor
        hemodynamic['diastolic_pressure'] *= pressure_factor
        hemodynamic['mean_pressure'] *= pressure_factor
        
        # Get material properties
        material_type = hemodynamic['material_type']
        material_props = self.material_properties[material_type].copy()
        
        # Load mesh to get geometric properties
        try:
            mesh = trimesh.load(str(stl_file))
            mesh_props = {
                'volume': float(mesh.volume),
                'surface_area': float(mesh.area),
                'bounds': mesh.bounds.tolist(),
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'is_watertight': mesh.is_watertight
            }
        except Exception as e:
            logger.error(f"Error loading mesh {stl_file}: {e}")
            mesh_props = {
                'volume': 0.0, 'surface_area': 0.0, 'bounds': [[0,0,0],[1,1,1]],
                'vertices': 0, 'faces': 0, 'is_watertight': False
            }
        
        # Generate comprehensive boundary conditions
        boundary_conditions = {
            'file_info': {
                'stl_file': str(stl_file),
                'region': region,
                'patient_id': patient_info['patient_id'],
                'creation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'patient_info': patient_info,
            'mesh_properties': mesh_props,
            'material_properties': material_props,
            'hemodynamic_properties': hemodynamic,
            'boundary_conditions': {
                'internal_pressure': hemodynamic['mean_pressure'],
                'inlet_conditions': {
                    'type': 'fixed_displacement',
                    'displacement': [0.0, 0.0, 0.0]
                },
                'outlet_conditions': {
                    'type': 'pressure',
                    'pressure': hemodynamic['diastolic_pressure']
                },
                'wall_conditions': {
                    'type': 'pressure_loading',
                    'internal_pressure': hemodynamic['mean_pressure'],
                    'external_pressure': 0.0  # Atmospheric
                }
            },
            'analysis_settings': {
                'analysis_type': 'static',
                'nonlinear': True,
                'large_deformation': True,
                'solver': 'direct',
                'convergence_tolerance': 1e-6
            }
        }
        
        return boundary_conditions
    
    def run_ansys_analysis(self, boundary_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run ANSYS stress analysis for a single vessel region"""
        
        if not ANSYS_AVAILABLE:
            logger.error("PyAnsys not available. Cannot run analysis.")
            return None
        
        stl_file = boundary_conditions['file_info']['stl_file']
        patient_id = boundary_conditions['patient_info']['patient_id']
        region = boundary_conditions['file_info']['region']
        
        # Create unique working directory for this analysis
        work_dir = self.results_dir / f"patient_{patient_id:02d}" / f"{Path(stl_file).stem}_analysis"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Launch MAPDL
            mapdl = launch_mapdl(
                run_location=str(work_dir),
                mode='console',
                override=True,
                license_server_ip=self.ansys_license_server if self.ansys_license_server else None
            )
            
            mapdl.clear()
            mapdl.prep7()
            
            # Import STL mesh
            mapdl.input_file = str(stl_file)
            
            # Simplified mesh import (in practice, this would be more sophisticated)
            # Convert STL to ANSYS format
            mesh = trimesh.load(stl_file)
            
            # Create nodes and elements (simplified approach)
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Add nodes
            for i, vertex in enumerate(vertices):
                mapdl.n(i+1, vertex[0], vertex[1], vertex[2])
            
            # Define element type (shell elements for vessel walls)
            mapdl.et(1, 'SHELL181')
            mapdl.keyopt(1, 1, 0)  # Membrane + bending
            mapdl.keyopt(1, 3, 2)  # Full integration
            
            # Add elements
            for i, face in enumerate(faces):
                if len(face) == 3:  # Triangle
                    mapdl.e(face[0]+1, face[1]+1, face[2]+1)
            
            # Define material properties
            mat_props = boundary_conditions['material_properties']
            mapdl.mp('EX', 1, mat_props['young_modulus'])
            mapdl.mp('NUXY', 1, mat_props['poisson_ratio'])
            mapdl.mp('DENS', 1, mat_props['density'])
            
            # Set shell thickness
            mapdl.sectype(1, 'SHELL')
            mapdl.secdata(mat_props['wall_thickness'])
            
            # Assign material and section
            mapdl.mat(1)
            mapdl.secnum(1)
            
            # Apply boundary conditions
            bc = boundary_conditions['boundary_conditions']
            
            # Internal pressure loading
            mapdl.sf('ALL', 'PRES', bc['internal_pressure'])
            
            # Fixed supports (simplified - would identify specific regions in practice)
            # Select a small region for fixed boundary
            mapdl.nsel('S', 'LOC', 'Z', mesh.bounds[0][2])  # Bottom boundary
            mapdl.d('ALL', 'ALL', 0)  # Fix all DOF
            mapdl.nsel('ALL')  # Reselect all nodes
            
            # Solution settings
            mapdl.slashsolu()
            mapdl.antype('STATIC')
            
            if boundary_conditions['analysis_settings']['nonlinear']:
                mapdl.nlgeom('ON')  # Large deformation
                mapdl.nsubst(10, 50, 5)  # Substeps
                mapdl.autots('ON')  # Auto time stepping
            
            # Solve
            mapdl.solve()
            mapdl.finish()
            
            # Post-processing
            mapdl.post1()
            
            # Extract results
            try:
                von_mises_stress = mapdl.post_processing.nodal_eqv_stress()
                principal_stress = mapdl.post_processing.nodal_principal_stress()
                displacement = mapdl.post_processing.nodal_displacement()
                strain = mapdl.post_processing.nodal_elastic_strain()
                
                # Calculate derived quantities
                max_von_mises = float(np.max(von_mises_stress))
                mean_von_mises = float(np.mean(von_mises_stress))
                max_displacement = float(np.max(np.linalg.norm(displacement, axis=1)))
                max_principal = float(np.max(principal_stress[:, 0]))
                min_principal = float(np.min(principal_stress[:, 2]))
                
                # Safety factors
                yield_strength = mat_props['yield_strength']
                safety_factor = yield_strength / max_von_mises if max_von_mises > 0 else float('inf')
                
                results = {
                    'max_von_mises_stress': max_von_mises,
                    'mean_von_mises_stress': mean_von_mises,
                    'max_displacement': max_displacement,
                    'max_principal_stress': max_principal,
                    'min_principal_stress': min_principal,
                    'safety_factor': safety_factor,
                    'yield_strength': yield_strength,
                    'stress_ratio': max_von_mises / yield_strength,
                    'analysis_successful': True
                }
                
                # Save detailed results
                results_file = work_dir / "detailed_results.npz"
                np.savez(str(results_file), 
                        von_mises=von_mises_stress,
                        principal=principal_stress,
                        displacement=displacement,
                        strain=strain)
                
                logger.info(f"Analysis completed for {Path(stl_file).name}: "
                          f"Max stress: {max_von_mises:.0f} Pa, Safety factor: {safety_factor:.2f}")
                
            except Exception as e:
                logger.error(f"Error in post-processing for {stl_file}: {e}")
                results = {
                    'analysis_successful': False,
                    'error_message': str(e)
                }
            
            # Cleanup
            mapdl.exit()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ANSYS analysis for {stl_file}: {e}")
            return {
                'analysis_successful': False,
                'error_message': str(e)
            }
    
    def process_single_patient_stl(self, stl_file: Path) -> Tuple[str, bool, Dict]:
        """Process a single STL file for stress analysis"""
        
        try:
            # Extract patient ID and region from filename
            # Format: XX_MRAY_Region_aneurysm_N_vessels.stl
            filename_parts = stl_file.stem.split('_')
            patient_id = int(filename_parts[0])
            
            # Find region name in filename
            region = None
            for i, part in enumerate(filename_parts):
                if 'aneurysm' in part and i > 0:
                    # Region is everything between MRA and aneurysm
                    region_parts = filename_parts[2:i]
                    region = '_'.join(region_parts)
                    break
            
            if region is None:
                return f"Could not extract region from {stl_file.name}", False, {}
            
            # Get patient information
            patient_info = self.get_patient_info(patient_id)
            if not patient_info:
                patient_info = {'patient_id': patient_id, 'pressure_factor': 1.0}
            
            # Generate boundary conditions
            boundary_conditions = self.generate_boundary_conditions(stl_file, region, patient_info)
            
            # Save boundary conditions
            bc_file = stl_file.parent / f"{stl_file.stem}_boundary_conditions.json"
            with open(bc_file, 'w') as f:
                json.dump(boundary_conditions, f, indent=2)
            
            # Run ANSYS analysis
            if ANSYS_AVAILABLE:
                analysis_results = self.run_ansys_analysis(boundary_conditions)
                
                if analysis_results and analysis_results.get('analysis_successful', False):
                    # Save analysis results
                    results_file = stl_file.parent / f"{stl_file.stem}_stress_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(analysis_results, f, indent=2)
                    
                    return f"Patient {patient_id:02d} {region}: Analysis complete", True, analysis_results
                else:
                    error_msg = analysis_results.get('error_message', 'Unknown error') if analysis_results else 'No results'
                    return f"Patient {patient_id:02d} {region}: Analysis failed - {error_msg}", False, {}
            else:
                return f"Patient {patient_id:02d} {region}: BC generated (ANSYS not available)", True, {}
                
        except Exception as e:
            logger.error(f"Error processing {stl_file}: {e}")
            return f"Error processing {stl_file.name}: {str(e)}", False, {}
    
    def run_all_patients_analysis(self, max_workers: int = 16):
        """Run stress analysis for all patient STL files"""
        
        logger.info("Starting comprehensive stress analysis for all patients")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        
        # Load patient data
        if not self.load_patient_data():
            logger.warning("Could not load patient data, using defaults")
        
        # Find all STL files
        stl_files = list(self.test_base_dir.glob("**/*.stl"))
        logger.info(f"Found {len(stl_files)} STL files to analyze")
        
        if len(stl_files) == 0:
            logger.error("No STL files found in test directory")
            return
        
        start_time = time.time()
        
        # Process STL files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_stl = {
                executor.submit(self.process_single_patient_stl, stl_file): stl_file 
                for stl_file in stl_files
            }
            
            results = []
            completed = 0
            total = len(stl_files)
            successful = 0
            failed = 0
            
            for future in as_completed(future_to_stl):
                result_msg, success, analysis_results = future.result()
                results.append((result_msg, success, analysis_results))
                completed += 1
                
                if success:
                    successful += 1
                    logger.info(f"✓ ({completed}/{total}) {result_msg}")
                else:
                    failed += 1
                    logger.warning(f"✗ ({completed}/{total}) {result_msg}")
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n=== Comprehensive Stress Analysis Complete ===")
        logger.info(f"Total STL files processed: {total}")
        logger.info(f"Successful analyses: {successful}")
        logger.info(f"Failed analyses: {failed}")
        logger.info(f"Success rate: {successful/total*100:.1f}%")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Average time per file: {elapsed_time/total:.1f} seconds")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return successful, failed
    
    def generate_summary_report(self, results: List[Tuple[str, bool, Dict]]):
        """Generate a comprehensive summary report"""
        
        report_file = self.results_dir / "stress_analysis_summary.json"
        summary_data = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyses': len(results),
            'successful_analyses': sum(1 for _, success, _ in results if success),
            'failed_analyses': sum(1 for _, success, _ in results if not success),
            'success_rate': sum(1 for _, success, _ in results if success) / len(results) * 100,
            'ansys_available': ANSYS_AVAILABLE,
            'analysis_results': []
        }
        
        # Collect stress analysis statistics
        stress_values = []
        safety_factors = []
        
        for result_msg, success, analysis_results in results:
            if success and analysis_results.get('analysis_successful', False):
                stress_values.append(analysis_results.get('max_von_mises_stress', 0))
                safety_factors.append(analysis_results.get('safety_factor', float('inf')))
                
                summary_data['analysis_results'].append({
                    'description': result_msg,
                    'max_stress': analysis_results.get('max_von_mises_stress', 0),
                    'safety_factor': analysis_results.get('safety_factor', 0),
                    'stress_ratio': analysis_results.get('stress_ratio', 0)
                })
        
        if stress_values:
            summary_data['stress_statistics'] = {
                'max_stress_overall': float(np.max(stress_values)),
                'mean_stress': float(np.mean(stress_values)),
                'min_stress': float(np.min(stress_values)),
                'std_stress': float(np.std(stress_values)),
                'min_safety_factor': float(np.min(safety_factors)),
                'mean_safety_factor': float(np.mean(safety_factors))
            }
        
        with open(report_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive ANSYS stress analysis for all patient vessel files")
    parser.add_argument("--max-workers", type=int, default=16,
                       help="Number of CPU cores to use (default: 16)")
    parser.add_argument("--license-server", type=str, default=None,
                       help="ANSYS license server (e.g., '1055@license-server')")
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveStressAnalysis(ansys_license_server=args.license_server)
    analyzer.run_all_patients_analysis(max_workers=args.max_workers)

if __name__ == "__main__":
    main() 