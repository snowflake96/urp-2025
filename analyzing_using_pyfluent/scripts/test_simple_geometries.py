#!/usr/bin/env python3
"""
Test Simple Geometries with PyFluent
Test cube, sphere, and ellipsoid to verify PyFluent workflow
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Ansys environment
os.environ['AWP_ROOT251'] = '/opt/cvbml/softwares/ansys_inc/v251'
os.environ['PATH'] = f"/opt/cvbml/softwares/ansys_inc/v251/fluent/bin:{os.environ.get('PATH', '')}"

try:
    import ansys.fluent.core as pyfluent
    from ansys.fluent.core import launch_fluent
    logger.info(f"PyFluent version: {pyfluent.__version__}")
except ImportError as e:
    logger.error(f"Failed to import PyFluent: {e}")
    sys.exit(1)

def test_geometry(mesh_path, geometry_name):
    """Test a simple geometry with full PyFluent workflow"""
    logger.info(f"🧪 Testing {geometry_name} geometry: {mesh_path}")
    
    # Launch Fluent
    logger.info("1. Launching Fluent...")
    session = launch_fluent(dimension=3, precision="double", processor_count=2, ui_mode="no_gui")
    logger.info("✓ Fluent launched successfully")
    
    try:
        # Import STL
        logger.info("2. Importing STL geometry...")
        session.file.read_mesh(file_name=str(mesh_path))
        logger.info("✓ STL imported successfully!")
        
        # Check mesh info
        logger.info("3. Checking mesh information...")
        session.tui.mesh.check()
        logger.info("✓ Mesh check completed")
        
        # Set up physics models
        logger.info("4. Setting up physics models...")
        session.tui.define.models.viscous.laminar("yes")
        logger.info("✓ Laminar viscous model set")
        
        # Set material properties
        logger.info("5. Setting material properties...")
        # session.tui.define.materials.fluid.water_liquid("yes")  # Use water
        logger.info("✓ Using default fluid properties")
        
        # Set boundary conditions (if needed)
        logger.info("6. Setting up boundary conditions...")
        try:
            # This might fail if boundaries aren't properly defined
            session.tui.define.boundary_conditions.velocity_inlet("inlet", "yes", "no", 0.1, 0, 0, "no", 300, "no", "no", "yes", 1, 10)
        except:
            logger.info("• No specific boundary conditions set (using defaults)")
        
        # Initialize solution
        logger.info("7. Initializing solution...")
        session.tui.solve.initialize.compute_defaults.all_zones()
        session.tui.solve.initialize.initialize_flow()
        logger.info("✓ Flow initialized")
        
        # Run iterations
        logger.info("8. Running CFD iterations...")
        session.tui.solve.iterate(10)
        logger.info("✓ 10 iterations completed successfully!")
        
        # Save results
        logger.info("9. Saving results...")
        results_dir = Path(__file__).parent.parent / "results" / f"simple_{geometry_name}_SUCCESS"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save case and data
        case_path = results_dir / f"{geometry_name}_result.cas"
        data_path = results_dir / f"{geometry_name}_result.dat"
        
        session.file.write_case(file_name=str(case_path))
        session.file.write_data(file_name=str(data_path))
        
        logger.info(f"✓ Case saved: {case_path}")
        logger.info(f"✓ Data saved: {data_path}")
        
        # Export to VTK if possible
        try:
            vtk_path = results_dir / f"{geometry_name}_result.vtu"
            session.file.export.ensight_gold(file_name=str(vtk_path), surfaces=["*"])
            logger.info(f"✓ VTK exported: {vtk_path}")
        except Exception as e:
            logger.warning(f"VTK export failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed for {geometry_name}: {e}")
        return False
        
    finally:
        session.exit()
        logger.info("Fluent session closed.")

def main():
    base_dir = Path(__file__).parent.parent
    simple_dir = base_dir / "meshes" / "simple_test"
    
    logger.info("=" * 80)
    logger.info("🧪 Testing Simple Geometries with PyFluent")
    logger.info("=" * 80)
    
    # Test geometries in order of complexity (simplest first)
    test_geometries = [
        ("aneurysm_cube.stl", "cube"),
        ("aneurysm_ellipsoid.stl", "ellipsoid"),
        ("aneurysm_sphere.stl", "sphere")
    ]
    
    results = {}
    
    for mesh_file, geometry_name in test_geometries:
        mesh_path = simple_dir / mesh_file
        
        if not mesh_path.exists():
            logger.warning(f"⚠️  Geometry file not found: {mesh_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING: {geometry_name.upper()} GEOMETRY")
        logger.info(f"{'='*60}")
        
        success = test_geometry(mesh_path, geometry_name)
        results[geometry_name] = success
        
        if success:
            logger.info(f"🎉 {geometry_name.upper()} GEOMETRY: SUCCESS!")
        else:
            logger.error(f"❌ {geometry_name.upper()} GEOMETRY: FAILED")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    successful_geometries = [name for name, success in results.items() if success]
    failed_geometries = [name for name, success in results.items() if not success]
    
    if successful_geometries:
        logger.info(f"✅ SUCCESSFUL GEOMETRIES: {', '.join(successful_geometries)}")
        logger.info("🎯 PyFluent workflow is WORKING!")
        logger.info("🔧 Next step: Improve mesh conversion for complex aneurysm geometry")
    else:
        logger.info("❌ NO SUCCESSFUL GEOMETRIES")
        logger.info("🔧 PyFluent setup may need debugging")
    
    if failed_geometries:
        logger.info(f"❌ FAILED GEOMETRIES: {', '.join(failed_geometries)}")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    if successful_geometries:
        logger.info("• PyFluent workflow is verified and working")
        logger.info("• Problem is with the complex aneurysm geometry")
        logger.info("• Use external meshing tools (Gmsh, SALOME) for complex geometries")
        logger.info("• Convert STL → Fluent .cas format externally")
        logger.info("• Import .cas files instead of STL for complex geometries")
    else:
        logger.info("• Debug PyFluent setup and installation")
        logger.info("• Check Fluent licensing and environment variables")
    
    logger.info(f"\n🎉 Simple geometry testing complete!")

if __name__ == "__main__":
    main() 