#!/usr/bin/env python3
"""
Clean Mesh-Based Vessel Processing Pipeline

This pipeline processes vessels at the mesh level WITHOUT kissing artifact removal:
1. NIfTI → STL conversion
2. Mesh smoothing and quality improvement
3. Tetrahedralization with fTetWild

Kissing artifacts must be handled at the voxel level before mesh generation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import json
import logging
from typing import Dict, Optional
import trimesh
import pymeshlab
from skimage.measure import marching_cubes
from scipy.ndimage import binary_fill_holes


class CleanMeshPipeline:
    """Clean mesh-based pipeline focusing on quality"""
    
    def __init__(self, ftetwild_path="/opt/cvbml/repos/fTetWild/build/FloatTetwild_bin", 
                 verbose=True):
        self.ftetwild_path = Path(ftetwild_path)
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Verify fTetWild exists
        if not self.ftetwild_path.exists():
            raise FileNotFoundError(f"fTetWild not found at {self.ftetwild_path}")
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('CleanMeshPipeline')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_vessel(self, input_nifti: str, output_dir: str,
                      smooth_iterations: int = 20,
                      final_smooth_iterations: int = 5,
                      tet_epsilon: float = 1e-3,
                      aneurysm_data: Optional[Dict] = None) -> Dict:
        """
        Main pipeline entry point
        
        Parameters:
        -----------
        input_nifti : str
            Path to input NIfTI file
        output_dir : str
            Output directory for results
        smooth_iterations : int
            Initial Taubin smoothing iterations
        final_smooth_iterations : int
            Final Laplacian smoothing iterations
        tet_epsilon : float
            fTetWild epsilon parameter
        aneurysm_data : dict, optional
            Aneurysm location data (for future use)
            
        Returns:
        --------
        dict : Processing results
        """
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            'input': input_nifti,
            'output_dir': str(output_path),
            'parameters': {
                'smooth_iterations': smooth_iterations,
                'final_smooth_iterations': final_smooth_iterations,
                'tet_epsilon': tet_epsilon
            },
            'files': {},
            'statistics': {}
        }
        
        try:
            # Step 1: NIfTI to STL conversion
            self.logger.info("=== Step 1: NIfTI to STL Conversion ===")
            initial_mesh = self._nifti_to_stl(
                input_nifti,
                output_path / "01_initial.stl"
            )
            results['files']['initial_mesh'] = str(initial_mesh)
            
            # Get initial statistics
            mesh = trimesh.load(str(initial_mesh))
            results['statistics']['initial'] = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'watertight': mesh.is_watertight,
                'volume': float(mesh.volume) if mesh.is_watertight else None
            }
            
            # Step 2: Initial smoothing
            self.logger.info(f"\n=== Step 2: Initial Smoothing ({smooth_iterations} iterations) ===")
            smoothed_mesh = self._smooth_mesh(
                initial_mesh,
                output_path / "02_smoothed.stl",
                iterations=smooth_iterations,
                method='taubin'
            )
            results['files']['smoothed_mesh'] = str(smoothed_mesh)
            
            # Get smoothed statistics
            mesh = trimesh.load(str(smoothed_mesh))
            results['statistics']['smoothed'] = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'watertight': mesh.is_watertight,
                'volume': float(mesh.volume) if mesh.is_watertight else None
            }
            
            # Step 3: Final smoothing and cleanup
            self.logger.info(f"\n=== Step 3: Final Smoothing ({final_smooth_iterations} iterations) ===")
            final_mesh = self._smooth_mesh(
                smoothed_mesh,
                output_path / "03_final.stl",
                iterations=final_smooth_iterations,
                method='laplacian'
            )
            results['files']['final_mesh'] = str(final_mesh)
            
            # Get final statistics
            mesh = trimesh.load(str(final_mesh))
            results['statistics']['final'] = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'watertight': mesh.is_watertight,
                'volume': float(mesh.volume) if mesh.is_watertight else None
            }
            
            # Step 4: Tetrahedralization with fTetWild
            self.logger.info("\n=== Step 4: Tetrahedralization with fTetWild ===")
            tet_mesh = self._tetrahedralize_with_ftetwild(
                final_mesh,
                output_path / "04_tetmesh.mesh",
                epsilon=tet_epsilon
            )
            results['files']['tet_mesh'] = str(tet_mesh)
            
            # Save results summary
            with open(output_path / "pipeline_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create summary report
            self._create_summary_report(results, output_path / "summary.txt")
            
            self.logger.info("\n✅ Clean mesh pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _nifti_to_stl(self, input_file: str, output_file: Path) -> Path:
        """
        Convert NIfTI to STL with minimal processing
        """
        self.logger.info(f"Loading NIfTI: {input_file}")
        
        # Load NIfTI
        nifti = nib.load(input_file)
        volume = nifti.get_fdata()
        
        # Simple binary threshold
        mask = volume > 0.5
        
        # Fill small holes to ensure watertight mesh
        mask = binary_fill_holes(mask)
        
        # Extract surface using marching cubes
        self.logger.info("Extracting surface mesh...")
        verts, faces, normals, values = marching_cubes(mask, level=0.5)
        
        # Apply affine transformation
        affine = nifti.affine
        verts_transformed = np.dot(
            np.hstack([verts, np.ones((len(verts), 1))]),
            affine.T
        )[:, :3]
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=verts_transformed, faces=faces)
        
        # Remove small disconnected components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            self.logger.info(f"Found {len(components)} components, keeping largest")
            mesh = max(components, key=lambda m: len(m.vertices))
        
        # Fix mesh issues
        mesh.fix_normals()
        mesh.fill_holes()
        
        # Save
        mesh.export(str(output_file))
        self.logger.info(f"Saved initial mesh: {output_file}")
        self.logger.info(f"  Vertices: {len(mesh.vertices):,}")
        self.logger.info(f"  Faces: {len(mesh.faces):,}")
        self.logger.info(f"  Watertight: {mesh.is_watertight}")
        
        return output_file
    
    def _smooth_mesh(self, input_mesh: Path, output_mesh: Path,
                    iterations: int = 20, method: str = 'taubin') -> Path:
        """
        Apply mesh smoothing
        
        Parameters:
        -----------
        input_mesh : Path
            Input mesh file
        output_mesh : Path
            Output mesh file
        iterations : int
            Number of smoothing iterations
        method : str
            Smoothing method: 'taubin' or 'laplacian'
        """
        self.logger.info(f"Applying {method} smoothing ({iterations} iterations)...")
        
        # Use PyMeshLab for smoothing
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_mesh))
        
        # Get initial stats
        initial_vertices = ms.current_mesh().vertex_number()
        
        if method == 'taubin':
            # Taubin smoothing (less shrinkage)
            ms.apply_coord_taubin_smoothing(
                lambda_=0.5, 
                mu=-0.53, 
                stepsmoothnum=iterations
            )
        elif method == 'laplacian':
            # Laplacian smoothing
            ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=iterations
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Save result
        ms.save_current_mesh(str(output_mesh))
        
        final_vertices = ms.current_mesh().vertex_number()
        self.logger.info(f"  Vertices: {initial_vertices:,} → {final_vertices:,}")
        
        return output_mesh
    
    def _tetrahedralize_with_ftetwild(self, surface_file: Path, output_file: Path,
                                     epsilon: float = 1e-3) -> Path:
        """
        Run fTetWild for tetrahedralization
        """
        self.logger.info(f"Running fTetWild (epsilon={epsilon})...")
        
        cmd = [
            str(self.ftetwild_path),
            "-i", str(surface_file),
            "-o", str(output_file),
            "-e", str(epsilon),
            "--max-its", "80",
            "--stop-energy", "10"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("fTetWild completed successfully")
            
            if self.verbose and result.stdout:
                # Show last few lines of output
                for line in result.stdout.split('\n')[-10:]:
                    if line.strip():
                        self.logger.info(f"  {line}")
                        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"fTetWild failed: {e}")
            if e.stderr:
                self.logger.error(f"Error: {e.stderr}")
            raise
        
        return output_file
    
    def _create_summary_report(self, results: Dict, output_file: Path):
        """
        Create a human-readable summary report
        """
        with open(output_file, 'w') as f:
            f.write("CLEAN MESH PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Input: {results['input']}\n")
            f.write(f"Output: {results['output_dir']}\n\n")
            
            f.write("Parameters:\n")
            for key, value in results['parameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Processing Statistics:\n")
            for stage, stats in results['statistics'].items():
                f.write(f"\n{stage.upper()}:\n")
                for key, value in stats.items():
                    if value is not None:
                        if key == 'volume':
                            f.write(f"  {key}: {value:.2f} mm³\n")
                        else:
                            f.write(f"  {key}: {value:,}\n")
                    else:
                        f.write(f"  {key}: N/A\n")
            
            f.write("\nOutput Files:\n")
            for key, filepath in results['files'].items():
                f.write(f"  {key}: {Path(filepath).name}\n")
            
            f.write("\nNOTE: Kissing artifacts must be handled at the voxel level.\n")
            f.write("      This pipeline focuses on mesh quality and smoothing.\n")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean mesh-based vessel processing pipeline"
    )
    parser.add_argument('input_nifti', help='Input NIfTI file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--smooth-iterations', type=int, default=20,
                       help='Initial Taubin smoothing iterations')
    parser.add_argument('--final-smooth-iterations', type=int, default=5,
                       help='Final Laplacian smoothing iterations')
    parser.add_argument('--tet-epsilon', type=float, default=1e-3,
                       help='fTetWild epsilon parameter')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CleanMeshPipeline()
    results = pipeline.process_vessel(
        args.input_nifti,
        args.output_dir,
        smooth_iterations=args.smooth_iterations,
        final_smooth_iterations=args.final_smooth_iterations,
        tet_epsilon=args.tet_epsilon
    )
    
    print(f"\nPipeline complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 