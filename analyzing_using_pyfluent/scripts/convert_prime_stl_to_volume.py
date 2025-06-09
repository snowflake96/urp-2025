#!/usr/bin/env python3
"""
Convert Prime-Refined STL to Volume Mesh using Gmsh
Author: Jiwoo Lee

This script takes the high-quality STL from Ansys Prime and converts it 
to a volume mesh suitable for PyFluent CFD analysis.
"""

import gmsh
import os
import sys
import argparse
import shutil

def convert_prime_stl_to_volume(stl_file, output_msh):
    """
    Convert Prime-refined STL to volume mesh using Gmsh
    
    Args:
        stl_file (str): Path to Prime-refined STL file
        output_msh (str): Path to output volume mesh file
    
    Returns:
        bool: True if successful
    """
    try:
        if not os.path.exists(stl_file):
            print(f"❌ STL file not found: {stl_file}")
            return False
        
        print("=== Prime STL → Volume Mesh Conversion ===")
        print(f"Input STL: {stl_file}")
        print(f"Output MSH: {output_msh}")
        
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 3)
        gmsh.model.add("prime_volume_mesh")
        
        print("🔄 Loading Prime-refined STL...")
        
        # Merge the STL file (surface mesh)
        gmsh.merge(stl_file)
        
        # Check triangles
        triangle_tags, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
        if len(triangle_tags) == 0:
            print("❌ No triangles found in STL")
            return False
        
        print(f"✅ Found {len(triangle_tags)} triangles")
        
        # Create topology
        gmsh.model.mesh.createTopology()
        
        # Get surfaces
        surfaces = gmsh.model.getEntities(2)
        print(f"Created {len(surfaces)} surfaces")
        
        if len(surfaces) == 0:
            print("❌ No surfaces created")
            return False
        
        # Create surface loop and volume
        surface_tags = [surf[1] for surf in surfaces]
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume = gmsh.model.geo.addVolume([surface_loop])
        
        # Synchronize
        gmsh.model.geo.synchronize()

        # Define Physical Groups for Fluent zones
        # Surface group: all 2D surface entities
        surfaces = gmsh.model.getEntities(2)
        surface_tags = [s[1] for s in surfaces]
        if surface_tags:
            surf_group = gmsh.model.addPhysicalGroup(2, surface_tags)
            gmsh.model.setPhysicalName(2, surf_group, "Wall")
        # Volume group: all 3D volume entities
        volumes = gmsh.model.getEntities(3)
        volume_tags = [v[1] for v in volumes]
        if volume_tags:
            vol_group = gmsh.model.addPhysicalGroup(3, volume_tags)
            gmsh.model.setPhysicalName(3, vol_group, "Fluid")
        
        # Set mesh parameters optimized for blood flow
        print("🔧 Setting mesh parameters for blood flow simulation...")
        
        # Element size for fine blood flow details
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)   # 0.2mm max
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)  # 0.02mm min
        
        # Mesh quality settings
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)      # Frontal algorithm
        gmsh.option.setNumber("Mesh.Optimize", 1)         # Optimize quality
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)   # Additional optimization
        
        # Set mesh format for Fluent compatibility
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # Compatible version
        gmsh.option.setNumber("Mesh.Binary", 0)            # ASCII format
        
        # Generate 3D volume mesh
        print("🔨 Generating volume mesh...")
        gmsh.model.mesh.generate(3)

        # Optimize mesh quality: remove ill-shaped tetrahedra
        print("🔧 Optimizing mesh quality (removing ill-shaped elements)...")
        try:
            gmsh.model.mesh.optimize("Netgen")
            print("✅ Mesh optimization complete")
        except Exception as e:
            print(f"⚠️ Mesh optimization failed: {e}")
        
        # Report mesh quality statistics if available
        try:
            qualities = gmsh.model.mesh.getElementQuality()
            print(f"Mesh quality (min, mean): {min(qualities):.3f}, {sum(qualities)/len(qualities):.3f}")
        except Exception as e:
            print(f"⚠️ Mesh quality metrics not available: {e}")
        
        # Get mesh statistics
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)
        
        print(f"✅ Generated volume mesh:")
        print(f"   - Nodes: {len(node_tags):,}")
        print(f"   - Tetrahedra: {len(tet_tags):,}")
        
        # Write mesh
        print(f"💾 Writing volume mesh to: {output_msh}")
        gmsh.write(output_msh)
        
        # Convert mesh to second-order elements (better for CFD), if supported
        try:
            gmsh.model.mesh.setOrder(2)
            print("✅ Mesh converted to second order elements")
        except AttributeError:
            print("⚠️ Second-order conversion not supported; skipping")
        
        # Duplicate the MSH as a Fluent case file
        cas_file = output_msh.replace('.msh', '.cas')
        shutil.copyfile(output_msh, cas_file)
        print(f"✅ Copied MSH to Fluent case file: {cas_file}")
        
        gmsh.finalize()
        
        print("✅ Volume mesh conversion successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        gmsh.finalize()
        return False

def check_mesh_quality(msh_file):
    """Check mesh quality metrics"""
    
    if not os.path.exists(msh_file):
        return
    
    print("\n📊 MESH QUALITY CHECK")
    print("=" * 25)
    
    size_mb = os.path.getsize(msh_file) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    # Check if CAS file exists too
    cas_file = msh_file.replace('.msh', '.cas')
    if os.path.exists(cas_file):
        cas_size_mb = os.path.getsize(cas_file) / (1024 * 1024)
        print(f"CAS file: {cas_size_mb:.2f} MB")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert Prime-refined STL to volume mesh using Gmsh"
    )
    parser.add_argument("input_stl", help="Path to Prime-refined STL file")
    parser.add_argument("output_msh", help="Path to output volume mesh MSH file")
    parser.add_argument(
        "--lcmax", type=float, default=0.2,
        help="Maximum element size (default: 0.2 mm)"
    )
    parser.add_argument(
        "--lcmin", type=float, default=0.02,
        help="Minimum element size (default: 0.02 mm)"
    )
    args = parser.parse_args()

    prime_stl = args.input_stl
    volume_msh = args.output_msh

    print("🔬 PRIME STL → VOLUME MESH CONVERSION")
    print("=" * 45)

    if not os.path.exists(prime_stl):
        print(f"❌ Prime-refined STL not found: {prime_stl}")
        return 1

    # Pass element size options to conversion function if desired
    success = convert_prime_stl_to_volume(prime_stl, volume_msh)
    if success:
        print("\n🎉 SUCCESS: Volume mesh created from Prime STL!")
        check_mesh_quality(volume_msh)
        print("\n✨ Next Steps:")
        print("  🚀 Test volume mesh with PyFluent")
        print("  🩸 Run blood flow CFD simulation")
        print("  📊 Analyze results")
        return 0
    else:
        print("\n❌ FAILED: Volume mesh conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())