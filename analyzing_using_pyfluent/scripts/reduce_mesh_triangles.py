#!/usr/bin/env python3
"""
Reduce STL Mesh Triangle Count
Simplify complex STL meshes for better Fluent compatibility
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

def analyze_mesh(mesh_path):
    """Analyze mesh properties"""
    logger.info(f"📊 Analyzing mesh: {mesh_path}")
    
    mesh = trimesh.load(mesh_path)
    
    logger.info(f"• Vertices: {len(mesh.vertices):,}")
    logger.info(f"• Faces: {len(mesh.faces):,}")
    logger.info(f"• Triangles: {len(mesh.faces):,}")
    logger.info(f"• Surface Area: {mesh.area:.2f}")
    logger.info(f"• Volume: {mesh.volume:.2f}")
    logger.info(f"• Bounds: {mesh.bounds}")
    logger.info(f"• Is Watertight: {mesh.is_watertight}")
    logger.info(f"• Is Winding Consistent: {mesh.is_winding_consistent}")
    
    return mesh

def reduce_triangles(mesh, target_faces=None, reduction_ratio=0.5):
    """Reduce triangle count using mesh simplification"""
    original_faces = len(mesh.faces)
    
    if target_faces is None:
        target_faces = int(original_faces * reduction_ratio)
    
    logger.info(f"🔧 Reducing triangles: {original_faces:,} → {target_faces:,} (ratio: {target_faces/original_faces:.2f})")
    
    try:
        # Method 1: Quadric edge collapse decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"✓ Quadric decimation successful: {len(simplified.faces):,} faces")
        return simplified
        
    except Exception as e1:
        logger.warning(f"Quadric decimation failed: {e1}")
        
        try:
            # Method 2: Vertex clustering
            simplified = mesh.simplify_vertex_clustering(radius=mesh.scale / 100)
            logger.info(f"✓ Vertex clustering successful: {len(simplified.faces):,} faces")
            return simplified
            
        except Exception as e2:
            logger.warning(f"Vertex clustering failed: {e2}")
            
            try:
                # Method 3: Convex hull (most aggressive)
                simplified = mesh.convex_hull
                logger.info(f"✓ Convex hull successful: {len(simplified.faces):,} faces")
                return simplified
                
            except Exception as e3:
                logger.error(f"All simplification methods failed: {e3}")
                return mesh

def fix_mesh(mesh):
    """Fix common mesh issues"""
    logger.info("🔧 Fixing mesh issues...")
    
    # Fill holes if needed
    if not mesh.is_watertight:
        logger.info("• Filling holes...")
        mesh.fill_holes()
    
    # Fix winding if needed
    if not mesh.is_winding_consistent:
        logger.info("• Fixing winding...")
        mesh.fix_normals()
    
    # Remove duplicate vertices
    mesh.merge_vertices()
    logger.info("• Merged duplicate vertices")
    
    # Remove degenerate faces
    mesh.remove_degenerate_faces()
    logger.info("• Removed degenerate faces")
    
    return mesh

def create_multiple_versions(input_path, output_dir):
    """Create multiple simplified versions of the mesh"""
    logger.info(f"🎯 Creating multiple simplified versions")
    
    mesh = trimesh.load(input_path)
    original_faces = len(mesh.faces)
    
    # Create different reduction levels
    reduction_levels = [
        (0.8, "light"),      # 80% of original
        (0.5, "medium"),     # 50% of original  
        (0.2, "heavy"),      # 20% of original
        (0.1, "extreme"),    # 10% of original
        (1000, "fixed")      # Fixed 1000 triangles
    ]
    
    versions = {}
    
    for ratio, name in reduction_levels:
        try:
            logger.info(f"\n📦 Creating {name} version (ratio: {ratio})...")
            
            if name == "fixed":
                target_faces = 1000
            else:
                target_faces = int(original_faces * ratio)
            
            # Simplify
            simplified = reduce_triangles(mesh.copy(), target_faces=target_faces)
            
            # Fix issues
            simplified = fix_mesh(simplified)
            
            # Save
            output_path = output_dir / f"78_MRA1_seg_aneurysm_{name}.stl"
            simplified.export(output_path)
            
            versions[name] = {
                'path': output_path,
                'faces': len(simplified.faces),
                'reduction': len(simplified.faces) / original_faces,
                'mesh': simplified
            }
            
            logger.info(f"✓ Saved {name} version: {len(simplified.faces):,} faces → {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create {name} version: {e}")
    
    return versions

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "meshes" / "78_MRA1_seg_aneurysm_ASCII.stl"
    output_dir = base_dir / "meshes" / "simplified"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🔧 STL Mesh Triangle Reduction Tool")
    logger.info("=" * 80)
    
    if not input_path.exists():
        logger.error(f"Input STL file not found: {input_path}")
        return
    
    # Analyze original mesh
    logger.info(f"\n📊 ORIGINAL MESH ANALYSIS:")
    original_mesh = analyze_mesh(input_path)
    
    # Create simplified versions
    logger.info(f"\n🎯 CREATING SIMPLIFIED VERSIONS:")
    versions = create_multiple_versions(input_path, output_dir)
    
    # Summary
    logger.info(f"\n📋 SUMMARY:")
    logger.info(f"Original: {len(original_mesh.faces):,} faces")
    
    for name, info in versions.items():
        reduction_pct = (1 - info['reduction']) * 100
        logger.info(f"{name.capitalize()}: {info['faces']:,} faces ({reduction_pct:.1f}% reduction)")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    logger.info("• Try 'medium' version first (50% reduction)")
    logger.info("• Use 'light' if medium still crashes")
    logger.info("• Use 'heavy' or 'extreme' for very fast testing")
    logger.info("• All versions saved in: meshes/simplified/")
    
    logger.info(f"\n🎉 Mesh reduction complete!")

if __name__ == "__main__":
    main() 