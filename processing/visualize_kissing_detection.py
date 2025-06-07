#!/usr/bin/env python3
"""
Visualize detected kissing artifact regions
"""

import numpy as np
import trimesh
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import logging


def load_and_analyze_mesh(mesh_file):
    """Load mesh and analyze thin regions"""
    mesh = trimesh.load(mesh_file)
    
    # Parameters from the pipeline
    max_thickness = 8.0  # mm
    n_samples = 20000
    
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Sample points on the mesh
    samples, face_idx = trimesh.sample.sample_surface_even(mesh, count=n_samples)
    normals = mesh.face_normals[face_idx]
    
    # Create ray origins slightly inside the surface
    origins = samples - normals * 0.1
    
    # Cast rays to find thickness
    thickness_map = {}
    thin_samples = []
    
    print("Analyzing thickness...")
    
    # Cast rays inward
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=-normals
    )
    
    # Process intersections
    for i, loc in enumerate(locations):
        if i < len(index_ray):
            ray_idx = index_ray[i]
            thickness = np.linalg.norm(samples[ray_idx] - loc)
            
            if thickness < max_thickness:
                face = face_idx[ray_idx]
                thickness_map[face] = min(thickness_map.get(face, float('inf')), thickness)
                
                if thickness < 2.0:  # Very thin
                    thin_samples.append({
                        'point': samples[ray_idx],
                        'thickness': thickness,
                        'face': face
                    })
    
    print(f"Found {len(thickness_map)} faces with thickness < {max_thickness}mm")
    print(f"Found {len(thin_samples)} very thin samples (< 2mm)")
    
    # Analyze thickness distribution
    if thickness_map:
        thicknesses = list(thickness_map.values())
        print(f"\nThickness statistics:")
        print(f"  Min: {min(thicknesses):.2f}mm")
        print(f"  Max: {max(thicknesses):.2f}mm")
        print(f"  Mean: {np.mean(thicknesses):.2f}mm")
        print(f"  Median: {np.median(thicknesses):.2f}mm")
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(thicknesses, bins=50, alpha=0.7)
        plt.xlabel('Thickness (mm)')
        plt.ylabel('Number of faces')
        plt.title('Distribution of Detected Thin Regions')
        plt.axvline(x=2.0, color='r', linestyle='--', label='Very thin threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('thickness_distribution.png', dpi=150)
        print("\nSaved thickness distribution to thickness_distribution.png")
    
    return mesh, thickness_map, thin_samples


def create_visualization_mesh(mesh, thickness_map, output_file):
    """Create a colored mesh showing detected thin regions"""
    
    # Create color array for faces
    colors = np.ones((len(mesh.faces), 4)) * [0.8, 0.8, 0.8, 1.0]  # Default gray
    
    if thickness_map:
        # Color thin faces by thickness
        max_viz_thickness = 5.0  # mm
        
        for face, thickness in thickness_map.items():
            if face < len(colors):
                # Red for very thin, yellow for moderately thin
                normalized = min(thickness / max_viz_thickness, 1.0)
                if thickness < 1.0:
                    colors[face] = [1.0, 0.0, 0.0, 1.0]  # Red
                elif thickness < 2.0:
                    colors[face] = [1.0, 0.5, 0.0, 1.0]  # Orange
                elif thickness < 3.0:
                    colors[face] = [1.0, 1.0, 0.0, 1.0]  # Yellow
                else:
                    colors[face] = [0.0, 1.0, 0.0, 1.0]  # Green
    
    # Create colored mesh
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = colors
    
    # Export
    colored_mesh.export(output_file)
    print(f"\nSaved colored mesh to {output_file}")
    print("Color legend:")
    print("  Red: < 1mm thickness")
    print("  Orange: 1-2mm thickness")
    print("  Yellow: 2-3mm thickness")  
    print("  Green: 3-5mm thickness")
    print("  Gray: > 5mm or normal thickness")


def analyze_connectivity(mesh, thickness_map):
    """Analyze if thin regions connect different vessel components"""
    
    print("\nAnalyzing vessel connectivity...")
    
    # Find connected components in the mesh
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} connected components")
    
    if len(components) > 1:
        # Check if thin regions bridge different components
        # This would be a true kissing artifact
        
        # For each component, track which faces belong to it
        face_to_component = {}
        face_offset = 0
        
        for comp_idx, comp in enumerate(components):
            for i in range(len(comp.faces)):
                face_to_component[face_offset + i] = comp_idx
            face_offset += len(comp.faces)
        
        # Check thin faces
        bridge_candidates = []
        for face in thickness_map:
            if face in face_to_component:
                comp = face_to_component[face]
                
                # Check neighbors
                for vertex in mesh.faces[face]:
                    for neighbor_face in mesh.vertex_faces[vertex]:
                        if (neighbor_face >= 0 and 
                            neighbor_face in face_to_component and
                            face_to_component[neighbor_face] != comp):
                            bridge_candidates.append({
                                'face': face,
                                'connects': (comp, face_to_component[neighbor_face]),
                                'thickness': thickness_map[face]
                            })
                            break
        
        if bridge_candidates:
            print(f"\nFound {len(bridge_candidates)} potential bridges between components!")
            for bridge in bridge_candidates[:10]:  # Show first 10
                print(f"  Face {bridge['face']}: connects components {bridge['connects']}, "
                      f"thickness {bridge['thickness']:.2f}mm")
        else:
            print("\nNo bridges found between components - thin regions are within single components")
    else:
        print("Mesh is a single connected component")
        print("Thin regions are likely normal vessel walls, not kissing artifacts!")


def main():
    parser = argparse.ArgumentParser(description="Visualize kissing artifact detection")
    parser.add_argument('mesh_file', help='Input mesh file (STL)')
    parser.add_argument('--output', default='thin_regions_visualization.ply',
                       help='Output colored mesh file')
    
    args = parser.parse_args()
    
    # Analyze mesh
    mesh, thickness_map, thin_samples = load_and_analyze_mesh(args.mesh_file)
    
    # Create visualization
    create_visualization_mesh(mesh, thickness_map, args.output)
    
    # Analyze connectivity
    analyze_connectivity(mesh, thickness_map)
    
    print("\nVisualization complete!")
    print(f"View the colored mesh with: meshlab {args.output}")


if __name__ == "__main__":
    main() 