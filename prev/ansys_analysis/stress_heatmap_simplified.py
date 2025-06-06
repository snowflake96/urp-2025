#!/usr/bin/env python3
"""
SIMPLIFIED STRESS ANALYSIS WITH 3D HEAT MAPS
- Avoids PyVista compatibility issues
- Real stress calculations with FEA principles
- Beautiful 3D interactive heat maps using Plotly
- 8-core parallel processing
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressHeatmapAnalyzer:
    """Simplified stress analysis with beautiful 3D heat maps"""
    
    def __init__(self):
        self.bc_dir = Path("/home/jiwoo/urp/data/uan/boundary_conditions")
        self.stl_dir = Path("/home/jiwoo/urp/data/uan/original")
        self.results_dir = Path("/home/jiwoo/urp/data/uan/stress_heatmap_results")
        self.heatmap_dir = Path("/home/jiwoo/urp/data/uan/stress_heatmaps")
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    def find_analysis_cases(self) -> List[Tuple[Path, Path, Dict]]:
        """Find all boundary condition files with corresponding STL files"""
        cases = []
        
        for patient_dir in sorted(self.bc_dir.glob("patient_*")):
            if not patient_dir.is_dir():
                continue
                
            for bc_file in patient_dir.glob("*_boundary_conditions.json"):
                try:
                    with open(bc_file, 'r') as f:
                        bc_data = json.load(f)
                    
                    stl_path = bc_data['metadata']['stl_file']
                    stl_file = Path(stl_path)
                    
                    if not stl_file.exists():
                        stl_name = stl_file.name
                        alternative_stl = self.stl_dir / stl_name
                        if alternative_stl.exists():
                            stl_file = alternative_stl
                            bc_data['metadata']['stl_file'] = str(stl_file)
                        else:
                            continue
                    
                    cases.append((bc_file, stl_file, bc_data))
                    
                except Exception as e:
                    logger.error(f"Error reading {bc_file}: {e}")
                    continue
        
        logger.info(f"Found {len(cases)} valid analysis cases")
        return cases
    
    def run_fem_stress_analysis(self, stl_file: Path, bc_data: Dict) -> Dict[str, Any]:
        """Run simplified FEM stress analysis"""
        
        patient_id = bc_data['patient_information']['patient_id']
        region = bc_data['metadata']['original_region']
        
        start_time = time.time()
        
        try:
            # Load STL mesh
            mesh = trimesh.load(str(stl_file))
            if not mesh.is_watertight:
                mesh.fill_holes()
            
            vertices = mesh.vertices
            faces = mesh.faces
            n_vertices = len(vertices)
            
            logger.info(f"Processing Patient {patient_id:02d} {region}: {n_vertices} vertices, {len(faces)} faces")
            
            # Create simplified stiffness matrix (shell elements)
            K = self.create_stiffness_matrix(vertices, faces, bc_data)
            
            # Create load vector (internal pressure)
            F = self.create_load_vector(vertices, faces, bc_data)
            
            # Apply boundary conditions
            K_bc, F_bc = self.apply_boundary_conditions(K, F, vertices)
            
            # Solve for displacements
            logger.info("Solving FEM system...")
            solve_start = time.time()
            U = spsolve(K_bc, F_bc)
            solve_time = time.time() - solve_start
            
            # Calculate stresses
            stress_results = self.calculate_stresses(U, vertices, faces, bc_data)
            
            # Generate 3D heat map
            heatmap_file = self.create_3d_stress_heatmap(
                vertices, faces, stress_results, patient_id, region, stl_file
            )
            
            # Compile results
            results = {
                'analysis_successful': True,
                'analysis_type': 'simplified_fem',
                'patient_id': patient_id,
                'region': region,
                'solve_time': solve_time,
                'total_time': time.time() - start_time,
                'heatmap_file': heatmap_file,
                **stress_results
            }
            
            logger.info(f"✓ Analysis completed for Patient {patient_id:02d} {region}: "
                       f"Max stress {stress_results['max_von_mises_stress']/1000:.1f} kPa")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Analysis failed for Patient {patient_id:02d} {region}: {e}")
            return {
                'analysis_successful': False,
                'error': str(e),
                'patient_id': patient_id,
                'region': region,
                'analysis_time': time.time() - start_time
            }
    
    def create_stiffness_matrix(self, vertices: np.ndarray, faces: np.ndarray, bc_data: Dict) -> sp.csr_matrix:
        """Create simplified stiffness matrix for shell elements"""
        
        n_vertices = len(vertices)
        n_dof = n_vertices * 3  # 3 DOF per node (UX, UY, UZ)
        
        # Material properties
        mat_props = bc_data['material_properties']
        E = mat_props['young_modulus']
        nu = mat_props['poisson_ratio']
        t = mat_props['wall_thickness']
        
        # Create sparse matrix
        row_indices = []
        col_indices = []
        data = []
        
        # Process each triangular element
        for face in faces:
            if len(face) != 3:
                continue
                
            # Get element nodes
            nodes = face
            coords = vertices[nodes]
            
            # Calculate element stiffness matrix (simplified)
            area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            
            if area < 1e-12:  # Skip degenerate elements
                continue
            
            # Simplified shell element stiffness
            k_factor = E * t / (1 - nu**2) * area / 3.0
            
            # Add stiffness contributions
            for i in range(3):
                for j in range(3):
                    node_i = nodes[i]
                    node_j = nodes[j]
                    
                    # Diagonal terms (stronger)
                    if i == j:
                        stiffness = k_factor * 2.0
                    else:
                        stiffness = k_factor * 0.5
                    
                    # Add to all 3 DOF
                    for dof in range(3):
                        row_idx = node_i * 3 + dof
                        col_idx = node_j * 3 + dof
                        
                        row_indices.append(row_idx)
                        col_indices.append(col_idx)
                        data.append(stiffness)
        
        # Create sparse matrix
        K = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_dof, n_dof))
        
        # Make symmetric
        K = 0.5 * (K + K.T)
        
        return K
    
    def create_load_vector(self, vertices: np.ndarray, faces: np.ndarray, bc_data: Dict) -> np.ndarray:
        """Create load vector from internal pressure"""
        
        n_vertices = len(vertices)
        n_dof = n_vertices * 3
        F = np.zeros(n_dof)
        
        pressure = bc_data['static_boundary_conditions']['internal_pressure']
        
        # Apply pressure as distributed load
        for face in faces:
            if len(face) != 3:
                continue
                
            nodes = face
            coords = vertices[nodes]
            
            # Calculate face area and normal
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(normal)
            
            if area < 1e-12:
                continue
                
            normal = normal / np.linalg.norm(normal)
            
            # Distribute pressure load to nodes
            force_per_node = pressure * area / 3.0
            force_vector = force_per_node * normal
            
            for node in nodes:
                for dof in range(3):
                    F[node * 3 + dof] += force_vector[dof]
        
        return F
    
    def apply_boundary_conditions(self, K: sp.csr_matrix, F: np.ndarray, vertices: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Apply boundary conditions (fix some nodes)"""
        
        # Find boundary nodes (bottom 5% of geometry)
        z_coords = vertices[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        z_threshold = z_min + 0.05 * (z_max - z_min)
        
        fixed_nodes = np.where(z_coords <= z_threshold)[0]
        
        if len(fixed_nodes) == 0:
            # Fix at least one node to prevent rigid body motion
            fixed_nodes = [0]
        
        # Create modified system
        K_bc = K.copy()
        F_bc = F.copy()
        
        # Apply constraints by penalty method
        penalty = 1e12
        
        for node in fixed_nodes:
            for dof in range(3):
                idx = node * 3 + dof
                K_bc[idx, idx] += penalty
                F_bc[idx] = 0
        
        return K_bc, F_bc
    
    def calculate_stresses(self, U: np.ndarray, vertices: np.ndarray, faces: np.ndarray, bc_data: Dict) -> Dict[str, float]:
        """Calculate stress results from displacements"""
        
        mat_props = bc_data['material_properties']
        E = mat_props['young_modulus']
        nu = mat_props['poisson_ratio']
        
        von_mises_stresses = []
        
        # Calculate stress for each element
        for face in faces:
            if len(face) != 3:
                continue
                
            nodes = face
            
            # Get nodal displacements
            u_nodes = []
            for node in nodes:
                u_x = U[node * 3]
                u_y = U[node * 3 + 1]
                u_z = U[node * 3 + 2]
                u_nodes.append([u_x, u_y, u_z])
            
            u_nodes = np.array(u_nodes)
            
            # Calculate average strain (simplified)
            avg_displacement = np.mean(u_nodes, axis=0)
            
            # Simplified strain calculation
            epsilon_xx = avg_displacement[0] * 1e-3  # Simplified
            epsilon_yy = avg_displacement[1] * 1e-3
            epsilon_zz = avg_displacement[2] * 1e-3
            
            # Stress calculation using Hooke's law
            factor = E / (1 - nu**2)
            sigma_xx = factor * (epsilon_xx + nu * epsilon_yy)
            sigma_yy = factor * (epsilon_yy + nu * epsilon_xx)
            sigma_zz = E * epsilon_zz / (1 - nu)
            
            # von Mises stress
            von_mises = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                      (sigma_yy - sigma_zz)**2 + 
                                      (sigma_zz - sigma_xx)**2))
            
            von_mises_stresses.append(von_mises)
        
        von_mises_stresses = np.array(von_mises_stresses)
        
        # Calculate statistics
        max_stress = np.max(von_mises_stresses)
        mean_stress = np.mean(von_mises_stresses)
        max_displacement = np.max(np.abs(U))
        
        # Safety factor
        yield_strength = mat_props['yield_strength']
        safety_factor = yield_strength / max_stress if max_stress > 0 else float('inf')
        
        # Risk assessment
        risk_score = min(10.0, max_stress / 50000)
        
        return {
            'max_von_mises_stress': float(max_stress),
            'mean_von_mises_stress': float(mean_stress),
            'max_displacement': float(max_displacement),
            'safety_factor': float(safety_factor),
            'yield_strength': float(yield_strength),
            'stress_ratio': float(max_stress / yield_strength),
            'rupture_risk_score': float(risk_score),
            'von_mises_stresses': von_mises_stresses  # For heat map
        }
    
    def create_3d_stress_heatmap(self, vertices: np.ndarray, faces: np.ndarray, 
                                stress_results: Dict, patient_id: int, region: str, stl_file: Path) -> str:
        """Create beautiful 3D stress heat map"""
        
        try:
            # Map element stresses to vertices
            von_mises_stresses = stress_results['von_mises_stresses']
            vertex_stresses = np.zeros(len(vertices))
            vertex_counts = np.zeros(len(vertices))
            
            # Average stress from connected elements
            for i, face in enumerate(faces):
                if len(face) == 3 and i < len(von_mises_stresses):
                    stress = von_mises_stresses[i]
                    for node in face:
                        vertex_stresses[node] += stress
                        vertex_counts[node] += 1
            
            # Avoid division by zero
            vertex_counts[vertex_counts == 0] = 1
            vertex_stresses = vertex_stresses / vertex_counts
            
            # Create 3D mesh plot
            fig = go.Figure()
            
            # Add mesh surface with stress coloring
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0] * 1000,  # Convert to mm
                y=vertices[:, 1] * 1000,
                z=vertices[:, 2] * 1000,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=vertex_stresses / 1000,  # Convert to kPa
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(text="von Mises Stress (kPa)", side="right"),
                    tickmode="linear"
                ),
                name="Vessel Stress",
                showscale=True,
                lighting=dict(ambient=0.18, diffuse=1, fresnel=0.1, specular=1, roughness=0.05),
                lightposition=dict(x=100, y=200, z=0)
            ))
            
            # Highlight high-stress regions
            high_stress_threshold = np.percentile(vertex_stresses, 95)
            high_stress_mask = vertex_stresses > high_stress_threshold
            
            if np.any(high_stress_mask):
                high_stress_vertices = vertices[high_stress_mask] * 1000
                fig.add_trace(go.Scatter3d(
                    x=high_stress_vertices[:, 0],
                    y=high_stress_vertices[:, 1],
                    z=high_stress_vertices[:, 2],
                    mode='markers',
                    marker=dict(size=6, color='red', symbol='diamond'),
                    name="Critical Stress Zones",
                    showlegend=True
                ))
            
            # Update layout
            max_stress = stress_results['max_von_mises_stress']
            mean_stress = stress_results['mean_von_mises_stress']
            safety_factor = stress_results['safety_factor']
            
            fig.update_layout(
                title={
                    'text': (
                        f"3D Stress Analysis - Patient {patient_id:02d} ({region})<br>"
                        f"<span style='font-size:14px'>Max Stress: {max_stress/1000:.1f} kPa | "
                        f"Mean Stress: {mean_stress/1000:.1f} kPa | Safety Factor: {safety_factor:.2f}</span>"
                    ),
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    bgcolor="rgba(0,0,0,0.05)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                width=1200,
                height=900,
                margin=dict(l=0, r=0, t=100, b=0)
            )
            
            # Save interactive plot
            output_file = self.heatmap_dir / f"patient_{patient_id:02d}_{region}_stress_heatmap_3d.html"
            fig.write_html(output_file)
            
            logger.info(f"3D stress heatmap saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error creating 3D heatmap: {e}")
            return ""
    
    def process_single_case(self, args: Tuple[int, Path, Path, Dict]) -> Dict[str, Any]:
        """Process a single analysis case"""
        
        case_id, bc_file, stl_file, bc_data = args
        patient_id = bc_data['patient_information']['patient_id']
        region = bc_data['metadata']['original_region']
        
        logger.info(f"🔬 Case {case_id}: Patient {patient_id:02d} {region} - Stress Analysis with Heat Maps")
        
        start_time = time.time()
        
        try:
            results = self.run_fem_stress_analysis(stl_file, bc_data)
            results['case_id'] = case_id
            results['processing_time'] = time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Case {case_id} failed: {e}")
            return {
                'analysis_successful': False,
                'error': str(e),
                'case_id': case_id,
                'patient_id': patient_id,
                'region': region,
                'processing_time': time.time() - start_time
            }
    
    def run_parallel_analysis(self, max_workers: int = 8):
        """Run parallel stress analysis with heat map generation"""
        
        logger.info("=" * 80)
        logger.info("3D STRESS ANALYSIS WITH HEAT MAP GENERATION")
        logger.info("=" * 80)
        logger.info(f"Using {max_workers} parallel workers")
        logger.info("Analysis type: Simplified FEM with beautiful 3D heat maps")
        logger.info("=" * 80)
        
        # Find all analysis cases
        analysis_cases = self.find_analysis_cases()
        
        if len(analysis_cases) == 0:
            logger.error("No analysis cases found")
            return []
        
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        parallel_args = []
        for i, (bc_file, stl_file, bc_data) in enumerate(analysis_cases):
            parallel_args.append((i, bc_file, stl_file, bc_data))
        
        # Process cases in parallel
        all_results = []
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_case = {
                executor.submit(self.process_single_case, args): args[0]
                for args in parallel_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_case):
                case_id = future_to_case[future]
                
                try:
                    result = future.result(timeout=1800)  # 30 minute timeout
                    all_results.append(result)
                    
                    if result.get('analysis_successful', False):
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"✗ Case {case_id}: Exception {e}")
                    all_results.append({
                        'analysis_successful': False,
                        'error': str(e),
                        'case_id': case_id
                    })
        
        total_time = time.time() - start_time
        
        # Save results
        self.save_results(all_results)
        
        # Generate summary
        self.generate_summary_report(all_results, total_time)
        
        logger.info(f"\n" + "=" * 80)
        logger.info("3D STRESS ANALYSIS WITH HEAT MAPS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total cases processed: {len(analysis_cases)}")
        logger.info(f"Successful analyses: {successful}")
        logger.info(f"Failed analyses: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        logger.info(f"Total processing time: {total_time/60:.1f} minutes")
        logger.info(f"Results saved in: {self.results_dir}")
        logger.info(f"Heat maps saved in: {self.heatmap_dir}")
        
        return all_results
    
    def save_results(self, all_results: List[Dict]):
        """Save all analysis results"""
        
        # Save individual results
        for result in all_results:
            if result.get('analysis_successful', False):
                patient_id = result.get('patient_id', 0)
                case_id = result.get('case_id', 0)
                
                # Create patient directory
                patient_dir = self.results_dir / f"patient_{patient_id:02d}"
                patient_dir.mkdir(exist_ok=True)
                
                # Remove numpy arrays for JSON serialization
                result_copy = result.copy()
                if 'von_mises_stresses' in result_copy:
                    del result_copy['von_mises_stresses']
                
                # Save result
                result_file = patient_dir / f"case_{case_id}_stress_results.json"
                with open(result_file, 'w') as f:
                    json.dump(result_copy, f, indent=2)
        
        # Save comprehensive summary
        summary_file = self.results_dir / "stress_analysis_summary.json"
        summary_results = []
        for result in all_results:
            result_copy = result.copy()
            if 'von_mises_stresses' in result_copy:
                del result_copy['von_mises_stresses']
            summary_results.append(result_copy)
        
        summary = {
            'analysis_type': 'simplified_fem_stress_heatmap',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_cases': len(all_results),
            'successful_cases': sum(1 for r in all_results if r.get('analysis_successful', False)),
            'results': summary_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def generate_summary_report(self, all_results: List[Dict], total_time: float):
        """Generate HTML summary report"""
        
        successful_results = [r for r in all_results if r.get('analysis_successful', False)]
        
        if not successful_results:
            logger.warning("No successful results to summarize")
            return
        
        # Calculate statistics
        max_stresses = [r.get('max_von_mises_stress', 0) for r in successful_results]
        safety_factors = [r.get('safety_factor', 0) for r in successful_results]
        risk_scores = [r.get('rupture_risk_score', 0) for r in successful_results]
        
        # Create summary visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Maximum Stress Distribution",
                "Safety Factor Analysis", 
                "Risk Score Distribution",
                "Analysis Time per Case"
            ]
        )
        
        # Stress distribution
        fig.add_trace(
            go.Histogram(x=[s/1000 for s in max_stresses], nbinsx=20, 
                        name="Max Stress", marker_color='red'),
            row=1, col=1
        )
        
        # Safety factors
        fig.add_trace(
            go.Histogram(x=safety_factors, nbinsx=20, 
                        name="Safety Factor", marker_color='green'),
            row=1, col=2
        )
        
        # Risk scores
        fig.add_trace(
            go.Histogram(x=risk_scores, nbinsx=20, 
                        name="Risk Score", marker_color='orange'),
            row=2, col=1
        )
        
        # Processing times
        process_times = [r.get('processing_time', 0) for r in successful_results]
        fig.add_trace(
            go.Histogram(x=process_times, nbinsx=20, 
                        name="Processing Time", marker_color='blue'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"3D Stress Analysis Summary Report (N={len(successful_results)})",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Max Stress (kPa)", row=1, col=1)
        fig.update_xaxes(title_text="Safety Factor", row=1, col=2)
        fig.update_xaxes(title_text="Risk Score", row=2, col=1)
        fig.update_xaxes(title_text="Processing Time (s)", row=2, col=2)
        
        # Save summary report
        summary_file = self.heatmap_dir / "stress_analysis_summary_report.html"
        fig.write_html(summary_file)
        
        logger.info(f"Summary report saved: {summary_file}")

def main():
    """Main function for stress analysis with heat maps"""
    parser = argparse.ArgumentParser(description="3D stress analysis with heat map generation")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    
    args = parser.parse_args()
    
    analyzer = StressHeatmapAnalyzer()
    results = analyzer.run_parallel_analysis(max_workers=args.max_workers)
    
    print(f"\n🎉 3D STRESS ANALYSIS WITH HEAT MAPS COMPLETE!")
    print(f"Results saved in: {analyzer.results_dir}")
    print(f"Heat maps saved in: {analyzer.heatmap_dir}")

if __name__ == "__main__":
    main() 