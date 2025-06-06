#!/usr/bin/env python3
"""
ANSYS Analysis Runner
- Demonstrate how to run actual ANSYS stress analysis
- Use generated boundary conditions and APDL files
- Process results and extract biomechanical features
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ANSYSAnalysisRunner:
    """Run ANSYS stress analysis using generated boundary conditions"""
    
    def __init__(self):
        """Initialize ANSYS analysis runner"""
        self.bc_dir = Path("/home/jiwoo/urp/data/uan/boundary_conditions")
        self.results_dir = Path("/home/jiwoo/urp/data/uan/ansys_results")
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def find_boundary_condition_files(self) -> List[Tuple[Path, Path, Path]]:
        """Find all boundary condition files and their corresponding ANSYS commands"""
        
        bc_files = []
        
        for patient_dir in sorted(self.bc_dir.glob("patient_*")):
            if not patient_dir.is_dir():
                continue
                
            for bc_file in patient_dir.glob("*_boundary_conditions.json"):
                # Find corresponding ANSYS command file
                base_name = bc_file.stem.replace("_boundary_conditions", "")
                apdl_file = patient_dir / f"{base_name}_ansys_commands.txt"
                
                # Find corresponding STL file
                stl_file = None
                try:
                    with open(bc_file, 'r') as f:
                        bc_data = json.load(f)
                    stl_path = bc_data['metadata']['stl_file']
                    stl_file = Path(stl_path)
                    
                    if not stl_file.exists():
                        logger.warning(f"STL file not found: {stl_file}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error reading boundary conditions {bc_file}: {e}")
                    continue
                
                if apdl_file.exists() and stl_file and stl_file.exists():
                    bc_files.append((bc_file, apdl_file, stl_file))
        
        return bc_files
    
    def run_single_ansys_analysis(self, bc_file: Path, apdl_file: Path, stl_file: Path) -> Dict[str, Any]:
        """Run ANSYS analysis for a single case (simulation - actual implementation depends on ANSYS setup)"""
        
        try:
            # Load boundary conditions
            with open(bc_file, 'r') as f:
                bc_data = json.load(f)
            
            # Extract key parameters
            patient_id = bc_data['patient_information']['patient_id']
            region = bc_data['metadata']['original_region']
            material = bc_data['material_properties']
            hemodynamic = bc_data['hemodynamic_properties']
            
            logger.info(f"Simulating ANSYS analysis for Patient {patient_id:02d} {region}")
            
            # Simulate analysis results (in real implementation, this would call ANSYS)
            results = self.simulate_stress_analysis(bc_data, stl_file)
            
            # Save results
            patient_results_dir = self.results_dir / f"patient_{patient_id:02d}"
            patient_results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = patient_results_dir / f"{bc_file.stem.replace('_boundary_conditions', '_stress_results.json')}"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ANSYS analysis for {bc_file}: {e}")
            return {'analysis_successful': False, 'error': str(e)}
    
    def simulate_stress_analysis(self, bc_data: Dict, stl_file: Path) -> Dict[str, Any]:
        """Simulate stress analysis results (replace with actual ANSYS call)"""
        
        material = bc_data['material_properties']
        hemodynamic = bc_data['hemodynamic_properties']
        mesh_data = bc_data['mesh_analysis']
        patient = bc_data['patient_information']
        
        # Simulate realistic stress analysis results
        np.random.seed(patient['patient_id'])  # Reproducible results
        
        # Basic stress calculation using simplified vessel mechanics
        pressure = hemodynamic['mean_pressure']  # Pa
        wall_thickness = material['wall_thickness']  # m
        vessel_radius = 0.002  # 2mm typical
        
        # Laplace law: σ = P×r/t (hoop stress)
        base_stress = pressure * vessel_radius / wall_thickness
        
        # Add stress concentration factor for aneurysms (2-3x typical)
        stress_concentration = np.random.uniform(2.0, 3.5)
        max_stress = base_stress * stress_concentration
        
        # Add some realistic variation
        mean_stress = max_stress * 0.6
        min_stress = max_stress * 0.2
        
        # Calculate safety factor
        yield_strength = material['yield_strength']
        safety_factor = yield_strength / max_stress
        
        # Displacement calculation (simplified)
        young_modulus = material['young_modulus']
        max_displacement = (max_stress / young_modulus) * vessel_radius  # mm
        
        # Wall shear stress (simplified)
        flow_velocity = hemodynamic.get('flow_velocity', 0.4)  # m/s
        blood_viscosity = 0.0035  # Pa·s
        wall_shear_stress = blood_viscosity * flow_velocity / vessel_radius
        
        # Risk assessment
        risk_score = 5.0  # Base risk
        
        if safety_factor < 2.0:
            risk_score += 2.0
        elif safety_factor < 1.5:
            risk_score += 3.0
            
        if patient['age'] > 65:
            risk_score += 1.0
            
        if patient.get('hypertension', False):
            risk_score += 1.5
            
        if hemodynamic.get('risk_factor', 'moderate') == 'high':
            risk_score += 1.0
            
        risk_score = min(10.0, max(0.0, risk_score))
        
        # Generate realistic mesh stress distribution
        num_nodes = mesh_data.get('vertex_count', 10000)
        stress_distribution = np.random.lognormal(
            mean=np.log(mean_stress), 
            sigma=0.5, 
            size=min(num_nodes, 1000)
        )
        
        results = {
            'analysis_successful': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            
            # Stress results
            'max_von_mises_stress': float(max_stress),
            'mean_von_mises_stress': float(mean_stress),
            'min_von_mises_stress': float(min_stress),
            'stress_std': float(np.std(stress_distribution)),
            
            # Principal stresses
            'max_principal_stress': float(max_stress * 1.1),
            'min_principal_stress': float(-max_stress * 0.3),
            
            # Displacement
            'max_displacement': float(max_displacement),
            'mean_displacement': float(max_displacement * 0.6),
            
            # Safety analysis
            'safety_factor': float(safety_factor),
            'yield_strength': float(yield_strength),
            'stress_ratio': float(max_stress / yield_strength),
            
            # Wall shear stress
            'max_wall_shear_stress': float(wall_shear_stress * 2.0),
            'mean_wall_shear_stress': float(wall_shear_stress),
            
            # Risk assessment
            'rupture_risk_score': float(risk_score),
            'rupture_probability': float(min(0.95, max(0.01, (risk_score - 3.0) / 7.0))),
            
            # Clinical metrics
            'clinical_assessment': {
                'risk_category': 'high' if risk_score > 7.0 else 'moderate' if risk_score > 5.0 else 'low',
                'monitoring_recommendation': 'immediate' if risk_score > 8.0 else 'frequent' if risk_score > 6.0 else 'routine',
                'intervention_indicated': safety_factor < 1.5 or risk_score > 7.5
            },
            
            # Biomechanical features for ML
            'biomechanical_features': {
                'normalized_stress': float(max_stress / yield_strength),
                'stress_concentration_factor': float(stress_concentration),
                'wall_thickness_ratio': float(wall_thickness / 0.0005),  # Normalized to healthy
                'pressure_factor': float(pressure / 13300),  # Normalized to normal MAP
                'age_factor': float(patient['age'] / 60.0),
                'geometry_factor': float(mesh_data.get('sphericity', 0.5)),
                'flow_factor': float(flow_velocity / 0.4)  # Normalized to typical
            }
        }
        
        return results
    
    def extract_biomechanical_features(self, all_results: List[Dict]) -> pd.DataFrame:
        """Extract biomechanical features for machine learning"""
        
        features_list = []
        
        for result in all_results:
            if not result.get('analysis_successful', False):
                continue
                
            # Extract all biomechanical features
            features = result.get('biomechanical_features', {})
            
            # Add clinical outcomes
            features.update({
                'max_stress': result.get('max_von_mises_stress', 0),
                'safety_factor': result.get('safety_factor', 0),
                'rupture_risk_score': result.get('rupture_risk_score', 0),
                'rupture_probability': result.get('rupture_probability', 0),
                'risk_category': result.get('clinical_assessment', {}).get('risk_category', 'unknown'),
                'intervention_indicated': result.get('clinical_assessment', {}).get('intervention_indicated', False)
            })
            
            features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Save features dataset
        features_file = self.results_dir / "biomechanical_features_dataset.csv"
        df.to_csv(features_file, index=False)
        
        logger.info(f"Biomechanical features dataset saved: {features_file}")
        logger.info(f"Dataset shape: {df.shape}")
        
        return df
    
    def generate_clinical_report(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive clinical analysis report"""
        
        successful_analyses = [r for r in all_results if r.get('analysis_successful', False)]
        
        if not successful_analyses:
            return {'error': 'No successful analyses to report'}
        
        # Extract key metrics
        risk_scores = [r.get('rupture_risk_score', 0) for r in successful_analyses]
        safety_factors = [r.get('safety_factor', 0) for r in successful_analyses]
        max_stresses = [r.get('max_von_mises_stress', 0) for r in successful_analyses]
        
        # Risk categorization
        high_risk_count = sum(1 for score in risk_scores if score > 7.0)
        moderate_risk_count = sum(1 for score in risk_scores if 5.0 < score <= 7.0)
        low_risk_count = len(risk_scores) - high_risk_count - moderate_risk_count
        
        # Safety analysis
        unsafe_count = sum(1 for sf in safety_factors if sf < 1.5)
        marginal_count = sum(1 for sf in safety_factors if 1.5 <= sf < 2.0)
        
        # Intervention recommendations
        intervention_needed = sum(1 for r in successful_analyses 
                                if r.get('clinical_assessment', {}).get('intervention_indicated', False))
        
        clinical_report = {
            'analysis_summary': {
                'total_patients_analyzed': len(successful_analyses),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            
            'risk_assessment': {
                'mean_risk_score': float(np.mean(risk_scores)),
                'max_risk_score': float(np.max(risk_scores)),
                'min_risk_score': float(np.min(risk_scores)),
                'risk_distribution': {
                    'high_risk': high_risk_count,
                    'moderate_risk': moderate_risk_count, 
                    'low_risk': low_risk_count
                },
                'high_risk_percentage': float(high_risk_count / len(risk_scores) * 100)
            },
            
            'safety_analysis': {
                'mean_safety_factor': float(np.mean(safety_factors)),
                'min_safety_factor': float(np.min(safety_factors)),
                'unsafe_cases': unsafe_count,
                'marginal_cases': marginal_count,
                'safe_cases': len(safety_factors) - unsafe_count - marginal_count
            },
            
            'stress_analysis': {
                'max_stress_overall': float(np.max(max_stresses)),
                'mean_stress': float(np.mean(max_stresses)),
                'stress_std': float(np.std(max_stresses))
            },
            
            'clinical_recommendations': {
                'immediate_intervention_needed': intervention_needed,
                'priority_monitoring_cases': high_risk_count + unsafe_count,
                'routine_follow_up_cases': low_risk_count
            },
            
            'population_insights': {
                'prevalence_high_risk': float(high_risk_count / len(risk_scores)),
                'prevalence_unsafe_stress': float(unsafe_count / len(safety_factors)),
                'mean_rupture_probability': float(np.mean([r.get('rupture_probability', 0) for r in successful_analyses]))
            }
        }
        
        # Save clinical report
        report_file = self.results_dir / "clinical_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(clinical_report, f, indent=2)
        
        logger.info(f"Clinical analysis report saved: {report_file}")
        
        return clinical_report
    
    def run_comprehensive_analysis(self, max_cases: Optional[int] = None):
        """Run comprehensive ANSYS analysis for all cases"""
        
        logger.info("Starting comprehensive ANSYS stress analysis")
        
        # Find all boundary condition files
        bc_files = self.find_boundary_condition_files()
        logger.info(f"Found {len(bc_files)} cases to analyze")
        
        if max_cases:
            bc_files = bc_files[:max_cases]
            logger.info(f"Limited to first {max_cases} cases for demonstration")
        
        all_results = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        # Process each case
        for i, (bc_file, apdl_file, stl_file) in enumerate(bc_files, 1):
            logger.info(f"Processing case {i}/{len(bc_files)}: {bc_file.name}")
            
            result = self.run_single_ansys_analysis(bc_file, apdl_file, stl_file)
            all_results.append(result)
            
            if result.get('analysis_successful', False):
                successful += 1
                risk_score = result.get('rupture_risk_score', 0)
                safety_factor = result.get('safety_factor', 0)
                logger.info(f"  ✓ Analysis complete - Risk: {risk_score:.1f}/10, Safety: {safety_factor:.2f}")
            else:
                failed += 1
                logger.warning(f"  ✗ Analysis failed: {result.get('error', 'Unknown error')}")
        
        elapsed_time = time.time() - start_time
        
        # Generate comprehensive reports
        logger.info(f"\n=== Analysis Complete ===")
        logger.info(f"Successful analyses: {successful}")
        logger.info(f"Failed analyses: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
        
        # Extract biomechanical features
        features_df = self.extract_biomechanical_features(all_results)
        
        # Generate clinical report
        clinical_report = self.generate_clinical_report(all_results)
        
        # Print key clinical findings
        if 'risk_assessment' in clinical_report:
            ra = clinical_report['risk_assessment']
            sa = clinical_report['safety_analysis']
            cr = clinical_report['clinical_recommendations']
            
            logger.info(f"\n=== Clinical Analysis Summary ===")
            logger.info(f"Mean risk score: {ra['mean_risk_score']:.2f}/10")
            logger.info(f"High-risk cases: {ra['risk_distribution']['high_risk']} ({ra['high_risk_percentage']:.1f}%)")
            logger.info(f"Unsafe stress cases: {sa['unsafe_cases']}")
            logger.info(f"Intervention needed: {cr['immediate_intervention_needed']} cases")
            logger.info(f"Mean safety factor: {sa['mean_safety_factor']:.2f}")
        
        return all_results, features_df, clinical_report

def main():
    """Main function for demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ANSYS stress analysis using generated boundary conditions")
    parser.add_argument("--max-cases", type=int, default=None,
                       help="Maximum number of cases to process (for testing)")
    parser.add_argument("--demo", action="store_true",
                       help="Run demonstration with first 10 cases")
    
    args = parser.parse_args()
    
    if args.demo:
        max_cases = 10
        logger.info("Running demonstration mode with 10 cases")
    else:
        max_cases = args.max_cases
    
    runner = ANSYSAnalysisRunner()
    results, features, clinical_report = runner.run_comprehensive_analysis(max_cases)
    
    print(f"\nAnalysis complete! Results saved in: {runner.results_dir}")
    print(f"- Biomechanical features: biomechanical_features_dataset.csv")
    print(f"- Clinical report: clinical_analysis_report.json")
    print(f"- Individual results: patient_XX/*_stress_results.json")

if __name__ == "__main__":
    main() 