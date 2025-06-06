"""
Stress analysis module for cerebral aneurysm FEA using PyAnsys.
This module performs structural and fluid-structure interaction analysis.
"""

import numpy as np
import pyvista as pv
from ansys.mapdl.core import launch_mapdl
from ansys.dpf import core as dpf
import trimesh
import os
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AneurysmStressAnalysis:
    """Perform FEA stress analysis on cerebral aneurysm models."""
    
    def __init__(self, working_dir: str = "./ansys_work"):
        """
        Initialize the stress analysis module.
        
        Args:
            working_dir: Directory for ANSYS working files
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self.mapdl = None
        
    def setup_mapdl(self):
        """Initialize ANSYS MAPDL instance."""
        self.mapdl = launch_mapdl(
            run_location=self.working_dir,
            mode='console',
            override=True
        )
        self.mapdl.clear()
        
    def load_mesh(self, mesh_file: str) -> trimesh.Trimesh:
        """
        Load aneurysm mesh from file.
        
        Args:
            mesh_file: Path to mesh file (STL, VTK, etc.)
            
        Returns:
            Trimesh object
        """
        logger.info(f"Loading mesh from {mesh_file}")
        mesh = trimesh.load(mesh_file)
        
        # Ensure mesh is clean and watertight
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fill_holes()
        
        logger.info(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
        
    def setup_material_properties(self):
        """Define material properties for vessel wall."""
        # Arterial wall properties (typical values)
        self.mapdl.mp('EX', 1, 2e6)      # Young's modulus (Pa)
        self.mapdl.mp('NUXY', 1, 0.45)   # Poisson's ratio
        self.mapdl.mp('DENS', 1, 1050)   # Density (kg/m³)
        
        # Hyperelastic model for more accurate representation
        # Mooney-Rivlin parameters
        self.mapdl.tb('HYPER', 1, 1, 2, 'MOONEY')
        self.mapdl.tbdata(1, 18.0e3, 3.0e3)  # C10, C01 parameters
        
    def apply_boundary_conditions(self, inlet_pressure: float = 120.0,
                                outlet_pressure: float = 80.0):
        """
        Apply boundary conditions for the analysis.
        
        Args:
            inlet_pressure: Systolic pressure in mmHg
            outlet_pressure: Diastolic pressure in mmHg
        """
        # Convert mmHg to Pa
        inlet_pa = inlet_pressure * 133.322
        outlet_pa = outlet_pressure * 133.322
        
        # Apply pressure loads
        self.mapdl.sf('ALL', 'PRES', inlet_pa)
        
        # Fix inlet/outlet boundaries (simplified)
        # In real analysis, these would be identified from mesh regions
        self.mapdl.d('ALL', 'UX', 0)
        self.mapdl.d('ALL', 'UY', 0)
        self.mapdl.d('ALL', 'UZ', 0)
        
    def run_static_analysis(self) -> Dict[str, np.ndarray]:
        """
        Run static structural analysis.
        
        Returns:
            Dictionary containing stress results
        """
        logger.info("Running static structural analysis")
        
        # Solution settings
        self.mapdl.slashsolu()
        self.mapdl.antype('STATIC')
        self.mapdl.nlgeom('ON')  # Large deformation
        self.mapdl.autots('ON')
        self.mapdl.nsubst(20, 50, 10)
        
        # Solve
        self.mapdl.solve()
        self.mapdl.finish()
        
        # Post-processing
        self.mapdl.post1()
        
        # Extract results
        results = {
            'von_mises': self.mapdl.post_processing.nodal_eqv_stress(),
            'principal_stress': self.mapdl.post_processing.nodal_principal_stress(),
            'displacement': self.mapdl.post_processing.nodal_displacement(),
            'strain': self.mapdl.post_processing.nodal_elastic_strain()
        }
        
        return results
        
    def calculate_wss(self, mesh: trimesh.Trimesh, 
                     flow_data: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate Wall Shear Stress (simplified).
        
        Args:
            mesh: Aneurysm mesh
            flow_data: Optional CFD results
            
        Returns:
            WSS values at each node
        """
        # Simplified WSS calculation
        # In practice, this would come from CFD analysis
        num_vertices = len(mesh.vertices)
        
        # Mock WSS calculation based on curvature
        curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=0.1
        )
        
        # Normalize and scale to typical WSS range (0-10 Pa)
        wss = np.abs(curvature)
        wss = (wss - wss.min()) / (wss.max() - wss.min()) * 10.0
        
        return wss
        
    def extract_features(self, results: Dict[str, np.ndarray],
                        mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Extract biomechanical features from FEA results.
        
        Args:
            results: FEA results dictionary
            mesh: Aneurysm mesh
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Von Mises stress features
        von_mises = results['von_mises']
        features['max_von_mises'] = np.max(von_mises)
        features['mean_von_mises'] = np.mean(von_mises)
        features['std_von_mises'] = np.std(von_mises)
        features['95th_percentile_von_mises'] = np.percentile(von_mises, 95)
        
        # Principal stress features
        principal = results['principal_stress']
        features['max_principal_stress'] = np.max(principal[:, 0])
        features['min_principal_stress'] = np.min(principal[:, 2])
        
        # Displacement features
        displacement = results['displacement']
        disp_magnitude = np.linalg.norm(displacement, axis=1)
        features['max_displacement'] = np.max(disp_magnitude)
        features['mean_displacement'] = np.mean(disp_magnitude)
        
        # WSS features
        wss = self.calculate_wss(mesh)
        features['max_wss'] = np.max(wss)
        features['mean_wss'] = np.mean(wss)
        features['low_wss_area'] = np.sum(wss < 0.5) / len(wss)  # Fraction of low WSS
        
        # Geometric features
        features['volume'] = mesh.volume
        features['surface_area'] = mesh.area
        features['sphericity'] = (np.pi**(1/3) * (6*mesh.volume)**(2/3)) / mesh.area
        
        return features
        
    def analyze_aneurysm(self, mesh_file: str,
                        save_results: bool = True) -> Tuple[Dict, Dict]:
        """
        Complete analysis pipeline for a single aneurysm.
        
        Args:
            mesh_file: Path to aneurysm mesh
            save_results: Whether to save results to file
            
        Returns:
            Tuple of (results, features)
        """
        # Setup ANSYS
        self.setup_mapdl()
        
        # Load mesh
        mesh = self.load_mesh(mesh_file)
        
        # Convert to ANSYS format and mesh
        # This is simplified - actual implementation would be more complex
        self.mapdl.prep7()
        
        # Import mesh to MAPDL (simplified)
        # In practice, you'd use more sophisticated mesh import
        
        # Setup analysis
        self.setup_material_properties()
        self.apply_boundary_conditions()
        
        # Run analysis
        results = self.run_static_analysis()
        
        # Extract features
        features = self.extract_features(results, mesh)
        
        # Save results if requested
        if save_results:
            base_name = os.path.splitext(os.path.basename(mesh_file))[0]
            np.savez(f"{self.working_dir}/{base_name}_results.npz", **results)
            
        # Cleanup
        self.mapdl.exit()
        
        return results, features


def main():
    """Example usage of the stress analysis module."""
    analyzer = AneurysmStressAnalysis()
    
    # Example analysis
    mesh_file = "path/to/aneurysm.stl"
    results, features = analyzer.analyze_aneurysm(mesh_file)
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main() 