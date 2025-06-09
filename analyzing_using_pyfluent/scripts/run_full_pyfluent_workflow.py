#!/usr/bin/env python3
"""
Full PyFluent CFD Workflow
STL → MSH → PyFluent CFD → VTP/VTU
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Ansys environment
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    logger.info(f"PyFluent version: {pyfluent.__version__}")
except ImportError as e:
    logger.error(f"Failed to import PyFluent: {e}")
    sys.exit(1)

def stl_to_msh(stl_path, msh_path):
    """Convert STL to MSH using Fluent meshing mode"""
    logger.info(f"Converting STL to MSH: {stl_path} → {msh_path}")
    meshing = launch_fluent(mode="meshing", version="3d", precision="double", processor_count=4, show_gui=False)
    meshing.tui.file.import_.stl(str(stl_path))
    meshing.tui.mesh.auto_mesh.create("watertight")
    meshing.tui.mesh.auto_mesh.tasks("Import Geometry").execute()
    meshing.tui.mesh.auto_mesh.tasks("Add Local Sizing").inputs.add_child_to_list()
    meshing.tui.mesh.auto_mesh.tasks("Add Local Sizing").inputs[0].scope.compute()
    meshing.tui.mesh.auto_mesh.tasks("Add Local Sizing").inputs[0].size.set(0.5)
    meshing.tui.mesh.auto_mesh.tasks("Add Local Sizing").execute()
    meshing.tui.mesh.auto_mesh.tasks("Generate the Surface Mesh").controls.curvature_normal_angle(40)
    meshing.tui.mesh.auto_mesh.tasks("Generate the Surface Mesh").controls.min_size(0.1)
    meshing.tui.mesh.auto_mesh.tasks("Generate the Surface Mesh").controls.max_size(1.0)
    meshing.tui.mesh.auto_mesh.tasks("Generate the Surface Mesh").execute()
    meshing.tui.mesh.auto_mesh.tasks("Update Regions").execute()
    meshing.tui.mesh.auto_mesh.tasks("Generate the Volume Mesh").controls.volume_fill_type("poly-hexcore")
    meshing.tui.mesh.auto_mesh.tasks("Generate the Volume Mesh").controls.max_cell_length(1.0)
    meshing.tui.mesh.auto_mesh.tasks("Generate the Volume Mesh").execute()
    meshing.tui.mesh.switch_to_solution_mode("yes")
    meshing.tui.file.write_case(str(msh_path))
    meshing.exit()
    logger.info("MSH file created successfully.")

def run_pyfluent_cfd(msh_path, results_dir):
    """Run CFD analysis using PyFluent and export VTP/VTU"""
    logger.info(f"Running CFD on mesh: {msh_path}")
    fluent = launch_fluent(version="3d", precision="double", processor_count=8, show_gui=False)
    fluent.tui.file.read_case(str(msh_path))
    # Set up physics
    fluent.tui.define.models.viscous.kw_sst("yes")
    fluent.tui.define.materials.change.create("blood", "yes", "constant", "1050", "no", "no", "yes", "constant", "0.0035", "no", "no", "no")
    fluent.tui.define.materials.change.fluid("blood")
    # Set up boundary conditions (auto-detect)
    boundaries = fluent.get_boundary_zone_names()
    inlet = [b for b in boundaries if "inlet" in b.lower()]
    outlet = [b for b in boundaries if "outlet" in b.lower()]
    if inlet:
        fluent.tui.define.boundary_conditions.set.velocity_inlet(inlet[0], "vmag", "no", "0.35")
    if outlet:
        fluent.tui.define.boundary_conditions.set.pressure_outlet(outlet[0], "0")
    # Initialize and solve
    fluent.tui.solve.initialize.compute_defaults.all_zones()
    fluent.tui.solve.initialize.initialize_flow()
    fluent.tui.solve.iterate(200)
    # Export VTP/VTU
    vtp_path = results_dir / "cfd_surface.vtp"
    vtu_path = results_dir / "cfd_volume.vtu"
    try:
        fluent.tui.file.export.vtk(str(results_dir / "cfd_export"), "cell-zones", "*", "yes", "yes", "pressure", "velocity", "wall-shear", "q")
    except Exception as e:
        logger.warning(f"Direct VTK export failed: {e}")
    # Save case/data
    fluent.tui.file.write_case_data(str(results_dir / "final_case.cas.h5"))
    fluent.exit()
    logger.info(f"Exported VTP: {vtp_path}\nExported VTU: {vtu_path}")

def main():
    patient_id = "78_MRA1_seg"
    base_dir = Path(__file__).parent.parent
    stl_path = base_dir / "meshes" / f"{patient_id}_aneurysm_ASCII.stl"
    results_dir = base_dir / "results" / f"{patient_id}_PYFLUENT_CFD"
    results_dir.mkdir(parents=True, exist_ok=True)
    msh_path = results_dir / f"{patient_id}.msh"
    # 1. STL → MSH
    stl_to_msh(stl_path, msh_path)
    # 2. Run CFD and export VTP/VTU
    run_pyfluent_cfd(msh_path, results_dir)
    logger.info("Workflow complete.")

if __name__ == "__main__":
    main() 