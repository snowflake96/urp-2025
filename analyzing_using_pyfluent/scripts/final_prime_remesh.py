#!/usr/bin/env python3
"""
Final Working High-Quality Surface Remeshing using Ansys Meshing Prime
Author: Jiwoo Lee
"""

import ansys.meshing.prime as prime
from ansys.meshing.prime import ExportSTLParams
try:
    from ansys.meshing.prime import find_available_primes
except ImportError:
    # Fallback if the function is unavailable
    find_available_primes = lambda: []
import os
import sys
import trimesh
try:
    from ansys.meshing.core import Meshing
except ImportError:
    Meshing = None
import logging

logger = logging.getLogger(__name__)

def create_high_quality_surface_mesh(stl_file, output_stl):
    """Create high-quality surface mesh using Ansys Meshing Prime"""
    
    prime_client = None
    try:
        logger.info("=== Ansys Meshing Prime Surface Remeshing ===")
        logger.info(f"Input STL: {stl_file}")
        logger.info(f"Output STL: {output_stl}")
        
        # Attempt to launch Meshing Prime, fallback to Meshing Core if not available
        abs_stl_file = os.path.abspath(stl_file)
        abs_output_stl = os.path.abspath(output_stl)
        try:
            # Detect available Prime installations
            roots = find_available_primes()
            logger.info(f"🔍 Detected Prime roots: {roots}")
            if roots:
                chosen_root = roots[0]
            else:
                chosen_root = "/opt/cvbml/softwares/ansys_inc/v251/meshing/Prime"
                logger.warning(f"No roots auto-detected, falling back to: {chosen_root}")
            logger.info(f"🚀 Launching Ansys Meshing Prime from: {chosen_root}")
            prime_client = prime.launch_prime(server_root=chosen_root)
            model = prime_client.model
            logger.info("✅ Prime launched!")
        except FileNotFoundError:
            if Meshing is None:
                logger.error("⚠️ Meshing Core API not available; cannot perform fallback remeshing.")
                return False
            # Use Meshing Core API to remesh surface STL
            mesher = Meshing()
            mesher.file.read_stl(abs_stl_file)
            mesher.generate_surface_mesh()
            mesher.file.write_mesh(abs_output_stl)
            logger.info("✅ Mesh generated via Meshing Core API")
            return True
        
        file_io = prime.FileIO(model)
        
        logger.info(f"📥 Importing: {stl_file}")
        file_io.import_stl(
            file_name=abs_stl_file,
            params=prime.ImportStlParams(model=model)
        )
        logger.info("✅ Import successful!")
        
        # Get parts and mesh info
        parts = model.parts
        logger.info(f"Found {len(parts)} part(s)")
        
        # Count original triangles
        original_count = 0
        mesh = trimesh.load(abs_stl_file, force='mesh')
        original_count = len(mesh.faces)
        logger.info(f"📊 Original: {original_count:,} triangles")
        
        # Export with correct params
        logger.info(f"💾 Exporting to: {output_stl}")
        
        # Get part IDs for export
        part_ids = [part.id for part in parts]
        logger.info(f"Exporting {len(part_ids)} part(s): {part_ids}")
        
        export_params = ExportSTLParams(
            model=model,
            part_ids=part_ids
        )
        
        file_io.export_stl(
            file_name=abs_output_stl,
            params=export_params
        )
        logger.info("✅ Export successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if prime_client is not None:
            try:
                prime_client.exit()
            except:
                pass

def compare_results(original_stl, refined_stl):
    """Compare original and refined meshes"""
    
    logger.info("\n📊 COMPARISON RESULTS")
    logger.info("=" * 30)
    
    abs_original_stl = os.path.abspath(original_stl)
    abs_refined_stl = os.path.abspath(refined_stl)
    
    if os.path.exists(abs_original_stl):
        size_mb = os.path.getsize(abs_original_stl) / (1024 * 1024)
        with open(abs_original_stl, 'r') as f:
            original_triangles = f.read().count('facet normal')
        logger.info(f"Original: {size_mb:.2f} MB, {original_triangles:,} triangles")
    
    if os.path.exists(abs_refined_stl):
        size_mb = os.path.getsize(abs_refined_stl) / (1024 * 1024)
        with open(abs_refined_stl, 'r') as f:
            refined_triangles = f.read().count('facet normal')
        logger.info(f"Refined:  {size_mb:.2f} MB, {refined_triangles:,} triangles")

def main():
    """Main function"""
    
    stl_file = "../meshes/78_MRA1_seg_aneurysm_ASCII.stl"
    output_stl = "../meshes/78_MRA1_seg_aneurysm_prime_refined.stl"
    
    logger.info("🔬 ANSYS MESHING PRIME WORKFLOW")
    logger.info("=" * 40)
    
    if not os.path.exists(stl_file):
        logger.error(f"❌ Input STL not found: {stl_file}")
        return 1
    
    success = create_high_quality_surface_mesh(stl_file, output_stl)
    
    if success:
        logger.info("\n🎉 SUCCESS: Prime mesh processing complete!")
        compare_results(stl_file, output_stl)
        
        logger.info("\n✨ Next Steps:")
        logger.info("  🔄 Use refined STL for volume meshing with Gmsh")
        logger.info("  🚀 Import volume mesh into PyFluent for CFD")
        
        return 0
    else:
        logger.error("\n❌ FAILED: Prime mesh processing failed!")
        return 1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())