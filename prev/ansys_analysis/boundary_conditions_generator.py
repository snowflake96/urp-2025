#!/usr/bin/env python3
"""
Boundary Conditions Generator for All Patient Vessel Files
- Generate comprehensive boundary conditions for each anatomical region
- Patient-specific hemodynamic adjustments
- Parallel processing for all patients using 16 CPUs
- Export results for ANSYS analysis
"""

import numpy as np
import trimesh
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BoundaryConditionsGenerator:
    """Generate comprehensive boundary conditions for all patient vessel regions"""
    
    def __init__(self):
        """Initialize boundary conditions generator"""
        self.test_base_dir = Path("/home/jiwoo/urp/data/uan/test")
        self.results_dir = Path("/home/jiwoo/urp/data/uan/boundary_conditions")
        self.excel_path = Path("/home/jiwoo/urp/data/segmentation/aneu/SNHU_TAnDB_DICOM.xlsx")
        
        self.patient_data_df = None
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup anatomical properties
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
                'description': 'Healthy cerebral artery wall'
            },
            'aneurysm_wall': {
                'young_modulus': 1.0e6,      # Pa (50% reduction)
                'poisson_ratio': 0.45,       # Same as healthy
                'density': 1050,             # kg/m³  
                'wall_thickness': 0.0003,    # m (40% thinner)
                'yield_strength': 0.5e6,     # Pa (reduced strength)
                'description': 'Weakened aneurysmal wall with collagen degradation'
            },
            'small_vessel': {
                'young_modulus': 1.5e6,      # Pa (intermediate)
                'poisson_ratio': 0.45,       
                'density': 1050,             
                'wall_thickness': 0.0003,    # m (thinner walls)
                'yield_strength': 0.6e6,     
                'description': 'Small cerebral vessel (communicating arteries, distal branches)'
            }
        }
        
        # Comprehensive hemodynamic properties for different anatomical regions
        self.hemodynamic_properties = {
            'ACA': {
                'systolic_pressure': 14700,   # Pa (110 mmHg)
                'diastolic_pressure': 9800,   # Pa (73 mmHg)
                'mean_pressure': 11800,       # Pa
                'pulse_pressure': 4900,       # Pa
                'flow_velocity': 0.35,        # m/s
                'flow_rate': 3.5e-6,          # m³/s (3.5 mL/s)
                'reynolds_number': 1200,
                'womersley_number': 3.8,
                'vessel_diameter': 0.003,     # m (3 mm)
                'material_type': 'small_vessel',
                'risk_factor': 'moderate',
                'description': 'Anterior Cerebral Artery - supplies frontal cortex'
            },
            'Acom': {
                'systolic_pressure': 15300,   # Pa (115 mmHg)
                'diastolic_pressure': 10100,  # Pa (76 mmHg)
                'mean_pressure': 12200,       # Pa
                'pulse_pressure': 5200,       # Pa
                'flow_velocity': 0.25,        # m/s (communicating vessel)
                'flow_rate': 2.0e-6,          # m³/s (2.0 mL/s)
                'reynolds_number': 900,
                'womersley_number': 3.2,
                'vessel_diameter': 0.002,     # m (2 mm)
                'material_type': 'aneurysm_wall',  # High risk region
                'risk_factor': 'high',
                'description': 'Anterior Communicating Artery - high aneurysm risk location'
            },
            'ICA_total': {
                'systolic_pressure': 16000,   # Pa (120 mmHg)
                'diastolic_pressure': 10700,  # Pa (80 mmHg)
                'mean_pressure': 13300,       # Pa
                'pulse_pressure': 5300,       # Pa
                'flow_velocity': 0.5,         # m/s (high flow)
                'flow_rate': 6.0e-6,          # m³/s (6.0 mL/s)
                'reynolds_number': 2000,
                'womersley_number': 4.5,
                'vessel_diameter': 0.005,     # m (5 mm)
                'material_type': 'healthy_vessel',
                'risk_factor': 'moderate',
                'description': 'Internal Carotid Artery (total) - main cerebral supply'
            },
            'ICA_noncavernous': {
                'systolic_pressure': 15700,   # Pa (118 mmHg)
                'diastolic_pressure': 10400,  # Pa (78 mmHg)
                'mean_pressure': 13000,       # Pa
                'pulse_pressure': 5300,       # Pa
                'flow_velocity': 0.45,        # m/s
                'flow_rate': 5.5e-6,          # m³/s (5.5 mL/s)
                'reynolds_number': 1800,
                'womersley_number': 4.2,
                'vessel_diameter': 0.0045,    # m (4.5 mm)
                'material_type': 'aneurysm_wall',  # Thin-walled region
                'risk_factor': 'high',
                'description': 'ICA non-cavernous (supraclinoid) - thin walls, high pressure'
            },
            'ICA_cavernous': {
                'systolic_pressure': 15700,   # Pa (118 mmHg)
                'diastolic_pressure': 10400,  # Pa (78 mmHg)
                'mean_pressure': 13000,       # Pa
                'pulse_pressure': 5300,       # Pa
                'flow_velocity': 0.4,         # m/s
                'flow_rate': 5.0e-6,          # m³/s (5.0 mL/s)
                'reynolds_number': 1600,
                'womersley_number': 4.0,
                'vessel_diameter': 0.0045,    # m (4.5 mm)
                'material_type': 'healthy_vessel',
                'risk_factor': 'low',
                'description': 'ICA cavernous segment - protected by skull base'
            },
            'Pcom': {
                'systolic_pressure': 15000,   # Pa (112 mmHg)
                'diastolic_pressure': 9900,   # Pa (74 mmHg)
                'mean_pressure': 12000,       # Pa
                'pulse_pressure': 5100,       # Pa
                'flow_velocity': 0.3,         # m/s
                'flow_rate': 2.5e-6,          # m³/s (2.5 mL/s)
                'reynolds_number': 1100,
                'womersley_number': 3.5,
                'vessel_diameter': 0.0025,    # m (2.5 mm)
                'material_type': 'small_vessel',
                'risk_factor': 'moderate',
                'description': 'Posterior Communicating Artery - connects anterior/posterior circulation'
            },
            'BA': {
                'systolic_pressure': 15500,   # Pa (116 mmHg)
                'diastolic_pressure': 10200,  # Pa (77 mmHg)
                'mean_pressure': 12500,       # Pa
                'pulse_pressure': 5300,       # Pa
                'flow_velocity': 0.4,         # m/s
                'flow_rate': 4.0e-6,          # m³/s (4.0 mL/s)
                'reynolds_number': 1500,
                'womersley_number': 4.0,
                'vessel_diameter': 0.004,     # m (4 mm)
                'material_type': 'healthy_vessel',
                'risk_factor': 'moderate',
                'description': 'Basilar Artery - main posterior circulation vessel'
            },
            'Other_posterior': {
                'systolic_pressure': 14500,   # Pa (109 mmHg)
                'diastolic_pressure': 9600,   # Pa (72 mmHg)
                'mean_pressure': 11600,       # Pa
                'pulse_pressure': 4900,       # Pa
                'flow_velocity': 0.3,         # m/s
                'flow_rate': 2.8e-6,          # m³/s (2.8 mL/s)
                'reynolds_number': 1000,
                'womersley_number': 3.3,
                'vessel_diameter': 0.003,     # m (3 mm)
                'material_type': 'small_vessel',
                'risk_factor': 'low',
                'description': 'Other posterior circulation vessels'
            },
            'PCA': {
                'systolic_pressure': 14800,   # Pa (111 mmHg)
                'diastolic_pressure': 9800,   # Pa (73 mmHg)
                'mean_pressure': 11800,       # Pa
                'pulse_pressure': 5000,       # Pa
                'flow_velocity': 0.35,        # m/s
                'flow_rate': 3.2e-6,          # m³/s (3.2 mL/s)
                'reynolds_number': 1200,
                'womersley_number': 3.7,
                'vessel_diameter': 0.0032,    # m (3.2 mm)
                'material_type': 'small_vessel',
                'risk_factor': 'moderate',
                'description': 'Posterior Cerebral Artery - supplies occipital cortex'
            }
        }
        
        # Cardiac cycle parameters
        self.cardiac_cycle = {
            'heart_rate': 70,               # beats/min
            'cycle_time': 60.0/70,          # seconds
            'systolic_fraction': 0.35,      # 35% of cycle
            'diastolic_fraction': 0.65,     # 65% of cycle
            'cardiac_output': 5.0,          # L/min
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
            return {'patient_id': patient_id, 'pressure_factor': 1.0, 'age': 60}
            
        patient_data = self.patient_data_df[self.patient_data_df['VintID'] == patient_id]
        
        if len(patient_data) == 0:
            return {'patient_id': patient_id, 'pressure_factor': 1.0, 'age': 60}
            
        patient_row = patient_data.iloc[0]
        
        # Extract relevant patient information
        info = {
            'patient_id': patient_id,
            'age': int(patient_row.get('Age', 60)),
            'gender': str(patient_row.get('Gender', 'Unknown')),
            'hypertension': bool(patient_row.get('Hypertension', False)),
        }
        
        # Calculate pressure adjustment factors
        pressure_factor = 1.0
        
        # Age-based adjustments
        if info['age'] > 65:
            pressure_factor *= 1.1      # 10% higher for elderly
        elif info['age'] < 40:
            pressure_factor *= 0.95     # 5% lower for young patients
            
        # Hypertension adjustment
        if info['hypertension']:
            pressure_factor *= 1.15     # 15% higher for hypertensive patients
            
        # Gender-based adjustments (statistical differences)
        if info['gender'].lower() == 'female':
            pressure_factor *= 0.98     # Slightly lower average BP
            
        info['pressure_factor'] = pressure_factor
        info['risk_multiplier'] = 1.0
        
        # Risk factor adjustments
        if info['age'] > 70:
            info['risk_multiplier'] *= 1.3
        if info['hypertension']:
            info['risk_multiplier'] *= 1.2
            
        return info
    
    def normalize_region_name(self, region: str) -> str:
        """Normalize region names for matching with properties"""
        region_mapping = {
            'ICA_total': 'ICA_total',
            'ICA_noncavernous': 'ICA_noncavernous', 
            'ICA_cavernous': 'ICA_cavernous',
            'ICA (total)': 'ICA_total',
            'ICA (noncavernous)': 'ICA_noncavernous',
            'ICA (cavernous)': 'ICA_cavernous',
            'Other_posterior': 'Other_posterior'
        }
        
        # Clean the region name
        clean_region = region.replace('(', '').replace(')', '').replace(' ', '_')
        
        # Check if it's in our mapping
        if clean_region in region_mapping:
            return region_mapping[clean_region]
        
        # Return as-is if not found
        return clean_region
    
    def generate_cardiac_cycle_pressures(self, base_systolic: float, base_diastolic: float, 
                                       num_points: int = 20) -> List[Dict[str, float]]:
        """Generate time-varying pressure profile for cardiac cycle"""
        
        cycle_time = self.cardiac_cycle['cycle_time']
        time_points = np.linspace(0, cycle_time, num_points)
        
        pressure_profile = []
        
        for i, t in enumerate(time_points):
            # Normalized time in cycle (0 to 1)
            phase = t / cycle_time
            
            # Generate realistic pressure waveform using Fourier series
            # Based on typical arterial pressure measurements
            if phase < 0.35:  # Systolic phase
                # Rising pressure during systole
                pressure_ratio = 0.5 * (1 + np.cos(np.pi + 2*np.pi*phase/0.35))
            else:  # Diastolic phase
                # Exponential decay during diastole
                diastolic_phase = (phase - 0.35) / 0.65
                pressure_ratio = np.exp(-3 * diastolic_phase) * 0.3
            
            # Calculate actual pressure
            pressure = base_diastolic + (base_systolic - base_diastolic) * pressure_ratio
            
            pressure_profile.append({
                'time': float(t),
                'phase': float(phase),
                'pressure': float(pressure),
                'systolic_flag': phase < 0.35
            })
        
        return pressure_profile
    
    def calculate_mesh_quality_metrics(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Calculate comprehensive mesh quality metrics"""
        
        try:
            # Basic geometric properties
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Mesh quality metrics
            quality_metrics = {
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'volume': float(mesh.volume) if mesh.volume > 0 else 0.0,
                'surface_area': float(mesh.area),
                'is_watertight': mesh.is_watertight,
                'is_winding_consistent': mesh.is_winding_consistent,
                'bounds': mesh.bounds.tolist(),
                'center_mass': mesh.center_mass.tolist(),
            }
            
            # Calculate additional quality metrics
            if len(faces) > 0:
                # Face areas
                face_areas = mesh.area_faces
                quality_metrics.update({
                    'min_face_area': float(np.min(face_areas)),
                    'max_face_area': float(np.max(face_areas)),
                    'mean_face_area': float(np.mean(face_areas)),
                    'face_area_std': float(np.std(face_areas))
                })
                
                # Aspect ratios (simplified)
                edge_lengths = []
                for face in faces:
                    for i in range(3):
                        v1 = vertices[face[i]]
                        v2 = vertices[face[(i+1)%3]]
                        edge_lengths.append(np.linalg.norm(v2 - v1))
                
                quality_metrics.update({
                    'min_edge_length': float(np.min(edge_lengths)),
                    'max_edge_length': float(np.max(edge_lengths)),
                    'mean_edge_length': float(np.mean(edge_lengths)),
                    'edge_length_std': float(np.std(edge_lengths))
                })
            
            # Geometric ratios
            bbox_dims = mesh.bounds[1] - mesh.bounds[0]
            bbox_volume = np.prod(bbox_dims)
            
            quality_metrics.update({
                'bbox_dimensions': bbox_dims.tolist(),
                'bbox_volume': float(bbox_volume),
                'volume_ratio': float(mesh.volume / bbox_volume) if bbox_volume > 0 else 0.0,
                'sphericity': float((np.pi**(1/3) * (6*mesh.volume)**(2/3)) / mesh.area) if mesh.area > 0 else 0.0
            })
            
        except Exception as e:
            logger.warning(f"Error calculating mesh quality metrics: {e}")
            quality_metrics = {
                'vertex_count': 0, 'face_count': 0, 'volume': 0.0, 'surface_area': 0.0,
                'is_watertight': False, 'error': str(e)
            }
        
        return quality_metrics
    
    def generate_comprehensive_boundary_conditions(self, stl_file: Path, region: str, 
                                                 patient_info: Dict) -> Dict[str, Any]:
        """Generate comprehensive boundary conditions for a specific STL file"""
        
        # Normalize region name
        normalized_region = self.normalize_region_name(region)
        
        # Get hemodynamic properties
        if normalized_region in self.hemodynamic_properties:
            hemodynamic = self.hemodynamic_properties[normalized_region].copy()
        else:
            # Default properties for unknown regions
            hemodynamic = {
                'systolic_pressure': 15000, 'diastolic_pressure': 10000,
                'mean_pressure': 12500, 'flow_velocity': 0.4,
                'reynolds_number': 1500, 'material_type': 'healthy_vessel',
                'description': f'Default properties for {region}'
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
        
        # Adjust material properties based on patient factors
        if patient_info.get('age', 60) > 65:
            # Stiffer arteries with age
            material_props['young_modulus'] *= 1.2
            material_props['wall_thickness'] *= 0.95
        
        # Load and analyze mesh
        try:
            mesh = trimesh.load(str(stl_file))
            mesh_quality = self.calculate_mesh_quality_metrics(mesh)
        except Exception as e:
            logger.error(f"Error loading mesh {stl_file}: {e}")
            mesh_quality = {'error': str(e), 'volume': 0.0, 'surface_area': 0.0}
        
        # Generate cardiac cycle pressure profile
        cardiac_pressures = self.generate_cardiac_cycle_pressures(
            hemodynamic['systolic_pressure'],
            hemodynamic['diastolic_pressure']
        )
        
        # Generate comprehensive boundary conditions
        boundary_conditions = {
            'metadata': {
                'stl_file': str(stl_file),
                'original_region': region,
                'normalized_region': normalized_region,
                'patient_id': patient_info['patient_id'],
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'generator_version': '1.0'
            },
            
            'patient_information': patient_info,
            
            'mesh_analysis': mesh_quality,
            
            'material_properties': material_props,
            
            'hemodynamic_properties': hemodynamic,
            
            'static_boundary_conditions': {
                'internal_pressure': hemodynamic['mean_pressure'],
                'external_pressure': 0.0,  # Atmospheric reference
                'inlet_conditions': {
                    'type': 'fixed_displacement',
                    'displacement': [0.0, 0.0, 0.0],
                    'description': 'Fixed proximal end to prevent rigid body motion'
                },
                'outlet_conditions': {
                    'type': 'pressure',
                    'pressure': hemodynamic['diastolic_pressure'],
                    'description': 'Applied diastolic pressure at distal end'
                },
                'wall_conditions': {
                    'type': 'pressure_loading',
                    'internal_pressure': hemodynamic['mean_pressure'],
                    'external_pressure': 0.0,
                    'description': 'Internal pressure loading on vessel wall'
                }
            },
            
            'transient_boundary_conditions': {
                'cardiac_cycle': self.cardiac_cycle,
                'pressure_profile': cardiac_pressures,
                'time_stepping': {
                    'total_time': self.cardiac_cycle['cycle_time'],
                    'time_steps': len(cardiac_pressures),
                    'step_size': self.cardiac_cycle['cycle_time'] / len(cardiac_pressures)
                }
            },
            
            'analysis_recommendations': {
                'analysis_type': 'static',  # or 'transient' for time-varying
                'nonlinear_geometry': True,
                'large_deformation': True,
                'solver_type': 'direct',
                'convergence_tolerance': 1e-6,
                'mesh_refinement': {
                    'target_element_size': mesh_quality.get('mean_edge_length', 0.001) * 0.5,
                    'refinement_regions': ['aneurysm_dome', 'high_curvature_areas'],
                    'inflation_layers': 3  # For boundary layer mesh
                }
            },
            
            'safety_analysis': {
                'yield_strength': material_props['yield_strength'],
                'safety_factor_target': 2.0,  # Minimum acceptable safety factor
                'critical_stress_locations': ['aneurysm_dome', 'vessel_bifurcations'],
                'rupture_risk_indicators': {
                    'wall_shear_stress_threshold': 0.5,  # Pa
                    'von_mises_stress_threshold': material_props['yield_strength'] * 0.5
                }
            },
            
            'clinical_correlations': {
                'aneurysm_risk_factors': {
                    'age_factor': patient_info.get('age', 60) / 60.0,
                    'hypertension_factor': 1.2 if patient_info.get('hypertension', False) else 1.0,
                    'gender_factor': 0.98 if patient_info.get('gender', '').lower() == 'female' else 1.0,
                    'anatomical_risk': hemodynamic.get('risk_factor', 'moderate')
                },
                'expected_outcomes': {
                    'peak_stress_location': 'aneurysm_dome',
                    'stress_concentration_factor': 2.5,  # Typical for aneurysms
                    'displacement_magnitude': '0.1-0.5 mm typical'
                }
            }
        }
        
        return boundary_conditions
    
    def process_single_stl_file(self, stl_file: Path) -> Tuple[str, bool, Dict]:
        """Process a single STL file for boundary condition generation"""
        
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
            
            # Generate boundary conditions
            boundary_conditions = self.generate_comprehensive_boundary_conditions(
                stl_file, region, patient_info
            )
            
            # Create patient-specific results directory
            patient_results_dir = self.results_dir / f"patient_{patient_id:02d}"
            patient_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save boundary conditions
            bc_filename = f"{stl_file.stem}_boundary_conditions.json"
            bc_file = patient_results_dir / bc_filename
            
            with open(bc_file, 'w') as f:
                json.dump(boundary_conditions, f, indent=2, default=str)
            
            # Generate ANSYS command file (APDL script)
            apdl_file = patient_results_dir / f"{stl_file.stem}_ansys_commands.txt"
            self.generate_ansys_commands(boundary_conditions, apdl_file)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(boundary_conditions)
            
            return (f"Patient {patient_id:02d} {region}: BC generated successfully", 
                   True, risk_metrics)
                   
        except Exception as e:
            logger.error(f"Error processing {stl_file}: {e}")
            return f"Error processing {stl_file.name}: {str(e)}", False, {}
    
    def generate_ansys_commands(self, boundary_conditions: Dict, output_file: Path):
        """Generate ANSYS APDL command file for the analysis"""
        
        bc = boundary_conditions
        material = bc['material_properties']
        hemodynamic = bc['hemodynamic_properties']
        static_bc = bc['static_boundary_conditions']
        
        commands = f"""! ANSYS APDL Commands for {bc['metadata']['original_region']} Analysis
! Generated on {bc['metadata']['generation_timestamp']}
! Patient ID: {bc['metadata']['patient_id']}

/PREP7

! Import STL mesh
! Note: Replace with actual import commands for your STL file
! CDREAD,DB,{bc['metadata']['stl_file']}

! Define element type (shell elements for vessel walls)
ET,1,SHELL181
KEYOPT,1,1,0    ! Membrane + bending capability
KEYOPT,1,3,2    ! Full integration

! Material properties
MP,EX,1,{material['young_modulus']:.0f}     ! Young's modulus (Pa)
MP,PRXY,1,{material['poisson_ratio']:.3f}   ! Poisson's ratio
MP,DENS,1,{material['density']:.0f}         ! Density (kg/m³)

! Shell section properties
SECTYPE,1,SHELL
SECDATA,{material['wall_thickness']:.6f}    ! Wall thickness (m)

! Assign material and section
MAT,1
SECNUM,1

! Boundary conditions
! Internal pressure loading
SF,ALL,PRES,{static_bc['internal_pressure']:.0f}

! Fixed supports (modify based on actual mesh geometry)
! NSEL,S,LOC,Z,ZMIN    ! Select nodes at one end
! D,ALL,ALL,0          ! Fix all DOF
! NSEL,ALL             ! Reselect all nodes

! Solution settings
/SOLU
ANTYPE,STATIC

! Nonlinear options
NLGEOM,ON           ! Large deformation
NSUBST,10,50,5      ! Load substeps
AUTOTS,ON           ! Automatic time stepping

! Solve
SOLVE

! Post-processing
/POST1
SET,LAST

! Extract results
*GET,MAX_STRESS,PLNSOL,0,S,EQV,MAX
*GET,MAX_DISP,PLNSOL,0,U,SUM,MAX

! Print results
*CFOPEN,RESULTS,TXT
*VWRITE,MAX_STRESS,MAX_DISP
(F15.2,F15.6)
*CFCLOS

! Safety factor calculation
SAFETY_FACTOR = {material['yield_strength']:.0f} / MAX_STRESS

! Export results
PLNSOL,S,EQV        ! von Mises stress
PLNSOL,U,SUM        ! Total displacement

FINISH

! Analysis Summary:
! Material: {material['description']}
! Mean pressure: {hemodynamic['mean_pressure']:.0f} Pa ({hemodynamic['mean_pressure']/133.322:.1f} mmHg)
! Expected max stress: {hemodynamic['mean_pressure']*2:.0f} Pa (stress concentration factor ~2)
! Target safety factor: > 2.0
"""
        
        with open(output_file, 'w') as f:
            f.write(commands)
    
    def calculate_risk_metrics(self, boundary_conditions: Dict) -> Dict[str, float]:
        """Calculate aneurysm rupture risk metrics from boundary conditions"""
        
        hemodynamic = boundary_conditions['hemodynamic_properties']
        material = boundary_conditions['material_properties']
        patient = boundary_conditions['patient_information']
        
        # Basic stress estimation (simplified)
        pressure = hemodynamic['mean_pressure']
        wall_thickness = material['wall_thickness']
        
        # Laplace law approximation for thin-walled vessel
        # σ = P×r/t (hoop stress)
        estimated_radius = 0.002  # 2mm typical aneurysm radius
        estimated_stress = pressure * estimated_radius / wall_thickness
        
        # Risk factors
        risk_metrics = {
            'estimated_wall_stress': estimated_stress,
            'pressure_mmHg': pressure / 133.322,
            'safety_factor_estimate': material['yield_strength'] / estimated_stress,
            'age_risk_factor': patient.get('age', 60) / 60.0,
            'hypertension_risk': 1.2 if patient.get('hypertension', False) else 1.0,
            'anatomical_risk': {'low': 0.8, 'moderate': 1.0, 'high': 1.3}.get(
                hemodynamic.get('risk_factor', 'moderate'), 1.0),
            'composite_risk_score': 0.0
        }
        
        # Composite risk score (0-10 scale)
        risk_score = 5.0  # Baseline
        risk_score *= risk_metrics['age_risk_factor']
        risk_score *= risk_metrics['hypertension_risk']
        risk_score *= risk_metrics['anatomical_risk']
        
        if risk_metrics['safety_factor_estimate'] < 2.0:
            risk_score *= 1.5
        
        risk_metrics['composite_risk_score'] = min(10.0, max(0.0, risk_score))
        
        return risk_metrics
    
    def run_all_patients_boundary_generation(self, max_workers: int = 16):
        """Generate boundary conditions for all patient STL files"""
        
        logger.info("Starting comprehensive boundary condition generation for all patients")
        logger.info(f"Using {max_workers} CPU cores for parallel processing")
        
        # Load patient data
        if not self.load_patient_data():
            logger.warning("Could not load patient data, using defaults")
        
        # Find all STL files
        stl_files = list(self.test_base_dir.glob("**/*.stl"))
        logger.info(f"Found {len(stl_files)} STL files to process")
        
        if len(stl_files) == 0:
            logger.error("No STL files found in test directory")
            return
        
        start_time = time.time()
        
        # Process STL files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_stl = {
                executor.submit(self.process_single_stl_file, stl_file): stl_file 
                for stl_file in stl_files
            }
            
            results = []
            completed = 0
            total = len(stl_files)
            successful = 0
            failed = 0
            
            all_risk_metrics = []
            
            for future in as_completed(future_to_stl):
                result_msg, success, risk_metrics = future.result()
                results.append((result_msg, success, risk_metrics))
                completed += 1
                
                if success:
                    successful += 1
                    all_risk_metrics.append(risk_metrics)
                    logger.info(f"✓ ({completed}/{total}) {result_msg}")
                else:
                    failed += 1
                    logger.warning(f"✗ ({completed}/{total}) {result_msg}")
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n=== Boundary Condition Generation Complete ===")
        logger.info(f"Total STL files processed: {total}")
        logger.info(f"Successful generations: {successful}")
        logger.info(f"Failed generations: {failed}")
        logger.info(f"Success rate: {successful/total*100:.1f}%")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Average time per file: {elapsed_time/total:.1f} seconds")
        
        # Generate summary report
        self.generate_summary_report(results, all_risk_metrics)
        
        return successful, failed
    
    def generate_summary_report(self, results: List[Tuple[str, bool, Dict]], 
                              risk_metrics: List[Dict]):
        """Generate comprehensive summary report"""
        
        report_file = self.results_dir / "boundary_conditions_summary.json"
        
        # Calculate risk statistics
        if risk_metrics:
            risk_scores = [rm.get('composite_risk_score', 0) for rm in risk_metrics]
            safety_factors = [rm.get('safety_factor_estimate', 0) for rm in risk_metrics]
            wall_stresses = [rm.get('estimated_wall_stress', 0) for rm in risk_metrics]
            
            risk_statistics = {
                'mean_risk_score': float(np.mean(risk_scores)),
                'max_risk_score': float(np.max(risk_scores)),
                'min_safety_factor': float(np.min(safety_factors)),
                'mean_safety_factor': float(np.mean(safety_factors)),
                'max_wall_stress': float(np.max(wall_stresses)),
                'mean_wall_stress': float(np.mean(wall_stresses)),
                'high_risk_count': sum(1 for score in risk_scores if score > 7.0),
                'low_safety_count': sum(1 for sf in safety_factors if sf < 2.0)
            }
        else:
            risk_statistics = {}
        
        summary_data = {
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyses': len(results),
            'successful_generations': sum(1 for _, success, _ in results if success),
            'failed_generations': sum(1 for _, success, _ in results if not success),
            'success_rate': sum(1 for _, success, _ in results if success) / len(results) * 100,
            'risk_statistics': risk_statistics,
            'output_directory': str(self.results_dir),
            'total_files_generated': sum(1 for _, success, _ in results if success) * 2,  # BC + APDL files
        }
        
        with open(report_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Print key findings
        if risk_statistics:
            logger.info(f"Risk Analysis Summary:")
            logger.info(f"  - Mean risk score: {risk_statistics['mean_risk_score']:.2f}/10")
            logger.info(f"  - High risk cases: {risk_statistics['high_risk_count']}")
            logger.info(f"  - Low safety factor cases: {risk_statistics['low_safety_count']}")
            logger.info(f"  - Mean safety factor: {risk_statistics['mean_safety_factor']:.2f}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate boundary conditions for all patient vessel files")
    parser.add_argument("--max-workers", type=int, default=16,
                       help="Number of CPU cores to use (default: 16)")
    
    args = parser.parse_args()
    
    generator = BoundaryConditionsGenerator()
    generator.run_all_patients_boundary_generation(max_workers=args.max_workers)

if __name__ == "__main__":
    main() 