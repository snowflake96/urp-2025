#!/usr/bin/env python3
"""
Create Simple Geometry for PyFluent Testing
Generate a simple spherical/ellipsoidal approximation of the aneurysm
Author: Jiwoo Lee
"""

import os
import sys
from pathlib import Path
import logging
import trimesh
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_original_mesh(mesh_path):
    """Analyze the original mesh to get dimensions"""
    logger.info(f"📊 Analyzing original mesh for dimensions")
    
    mesh = trimesh.load(mesh_path)
    bounds = mesh.bounds
    center = mesh.centroid
    
    # Calculate dimensions
    dimensions = bounds[1] - bounds[0]  # [width, height, depth]
    
    logger.info(f"• Center: {center}")
    logger.info(f"• Bounds: {bounds}")
    logger.info(f"• Dimensions: {dimensions}")
    logger.info(f"• Volume: {mesh.volume:.2f}")
    
    return center, dimensions, mesh.volume

def create_simple_sphere(center, radius, subdivisions=2):
    """Create a simple sphere"""
    logger.info(f"🔵 Creating sphere: center={center}, radius={radius:.2f}, subdivisions={subdivisions}")
    
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[20, 20])
    sphere.vertices += center
    
    logger.info(f"✓ Sphere created: {len(sphere.faces)} faces")
    return sphere

def create_simple_ellipsoid(center, dimensions, subdivisions=2):
    """Create a simple ellipsoid"""
    logger.info(f"🥚 Creating ellipsoid: center={center}, dimensions={dimensions}")
    
    # Create unit sphere and scale it
    sphere = trimesh.creation.uv_sphere(radius=1.0, count=[16, 12])
    
    # Scale to match dimensions
    sphere.vertices *= dimensions / 2  # dimensions to radii
    sphere.vertices += center
    
    logger.info(f"✓ Ellipsoid created: {len(sphere.faces)} faces")
    return sphere

def create_simple_capsule(center, dimensions):
    """Create a simple capsule (cylinder with rounded ends)"""
    logger.info(f"💊 Creating capsule: center={center}, dimensions={dimensions}")
    
    # Use the longest dimension as length, average of others as radius
    length = max(dimensions)
    radius = (dimensions[0] + dimensions[1] + dimensions[2] - length) / 2 * 0.5
    
    capsule = trimesh.creation.capsule(radius=radius, height=length, count=[12, 12])
    capsule.vertices += center
    
    logger.info(f"✓ Capsule created: {len(capsule.faces)} faces")
    return capsule

def create_simple_cylinder(center, dimensions):
    """Create a simple cylinder"""
    logger.info(f"🛢️ Creating cylinder: center={center}, dimensions={dimensions}")
    
    # Use average dimensions
    radius = np.mean(dimensions[:2]) / 2  # Average of width and height
    height = dimensions[2]  # Use depth as height
    
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    cylinder.vertices += center
    
    logger.info(f"✓ Cylinder created: {len(cylinder.faces)} faces")
    return cylinder

def create_test_geometries(center, dimensions, volume, output_dir):
    """Create multiple simple test geometries"""
    logger.info(f"🎯 Creating test geometries")
    
    # Calculate equivalent sphere radius from volume
    equiv_radius = (3 * abs(volume) / (4 * np.pi)) ** (1/3) if volume > 0 else np.mean(dimensions) / 2
    
    geometries = {}
    
    # 1. Simple sphere
    try:
        sphere = create_simple_sphere(center, equiv_radius)
        sphere_path = output_dir / "aneurysm_sphere.stl"
        sphere.export(sphere_path)
        geometries['sphere'] = {'mesh': sphere, 'path': sphere_path}
        logger.info(f"✓ Sphere saved: {sphere_path}")
    except Exception as e:
        logger.error(f"Failed to create sphere: {e}")
    
    # 2. Simple ellipsoid
    try:
        ellipsoid = create_simple_ellipsoid(center, dimensions)
        ellipsoid_path = output_dir / "aneurysm_ellipsoid.stl"
        ellipsoid.export(ellipsoid_path)
        geometries['ellipsoid'] = {'mesh': ellipsoid, 'path': ellipsoid_path}
        logger.info(f"✓ Ellipsoid saved: {ellipsoid_path}")
    except Exception as e:
        logger.error(f"Failed to create ellipsoid: {e}")
    
    # 3. Very simple sphere (low resolution)
    try:
        simple_sphere = create_simple_sphere(center, equiv_radius * 0.8)
        # Decimate further
        simple_sphere = simple_sphere.simplify_vertex_clustering(radius=equiv_radius/20)
        simple_path = output_dir / "aneurysm_simple.stl"
        simple_sphere.export(simple_path)
        geometries['simple'] = {'mesh': simple_sphere, 'path': simple_path}
        logger.info(f"✓ Simple sphere saved: {simple_path}")
    except Exception as e:
        logger.error(f"Failed to create simple sphere: {e}")
    
    # 4. Basic cube
    try:
        cube_size = np.mean(dimensions) * 0.8
        cube = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
        cube.vertices += center
        cube_path = output_dir / "aneurysm_cube.stl"
        cube.export(cube_path)
        geometries['cube'] = {'mesh': cube, 'path': cube_path}
        logger.info(f"✓ Cube saved: {cube_path}")
    except Exception as e:
        logger.error(f"Failed to create cube: {e}")
    
    return geometries

def main():
    base_dir = Path(__file__).parent.parent
    original_path = base_dir / "meshes" / "78_MRA1_seg_aneurysm_ASCII.stl"
    output_dir = base_dir / "meshes" / "simple_test"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🔧 Simple Geometry Generator for PyFluent Testing")
    logger.info("=" * 80)
    
    if not original_path.exists():
        logger.error(f"Original STL file not found: {original_path}")
        return
    
    # Analyze original mesh
    center, dimensions, volume = analyze_original_mesh(original_path)
    
    # Create simple test geometries
    logger.info(f"\n🎯 CREATING SIMPLE TEST GEOMETRIES:")
    geometries = create_test_geometries(center, dimensions, volume, output_dir)
    
    # Summary
    logger.info(f"\n📋 SUMMARY:")
    logger.info(f"Original mesh: Complex aneurysm geometry")
    logger.info(f"Created {len(geometries)} simple test geometries:")
    
    for name, info in geometries.items():
        faces = len(info['mesh'].faces)
        logger.info(f"• {name.capitalize()}: {faces} faces → {info['path'].name}")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    logger.info("• Test these geometries with PyFluent to verify workflow")
    logger.info("• Use 'simple' version for basic testing")
    logger.info("• Use 'sphere' or 'ellipsoid' for more realistic testing")
    logger.info("• Once workflow is verified, work on better mesh conversion")
    
    logger.info(f"\n🎉 Simple geometry generation complete!")

if __name__ == "__main__":
    main() 