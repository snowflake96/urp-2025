#!/usr/bin/env python3
"""
Random Walk Vessel Exploration

This script performs random walk exploration starting from aneurysm centroids.
From each aneurysm center, it walks randomly for 1000 steps within vessel tissue,
then crops the 3D model to show only the explored vessel regions.

Usage:
    python random_walk_extraction.py <nifti_file> <coordinates_json>
    
Example:
    python random_walk_extraction.py example_data/01_MRA1_seg.nii.gz my_aneurysm_locations.json
"""

import numpy as np
import nibabel as nib
import json
import os
import random
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from skimage import measure
import argparse

class VesselRandomWalk:
    def __init__(self, image_data, vessel_threshold=0.5):
        """
        Initialize random walk exploration
        
        Args:
            image_data: 3D numpy array of vessel segmentation
            vessel_threshold: Minimum voxel value to consider as vessel
        """
        self.image_data = image_data.astype(np.float32)
        self.vessel_threshold = vessel_threshold
        self.shape = image_data.shape
        
        # 26-connected neighborhood (3D)
        self.directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.directions.append((dx, dy, dz))
        
        print(f"Initialized random walk on volume shape: {self.shape}")
        print(f"Using {len(self.directions)} possible directions")
    
    def is_valid_position(self, x, y, z):
        """Check if position is within bounds and in vessel tissue"""
        if (x < 0 or x >= self.shape[0] or 
            y < 0 or y >= self.shape[1] or 
            z < 0 or z >= self.shape[2]):
            return False
        
        return self.image_data[x, y, z] > self.vessel_threshold
    
    def get_valid_neighbors(self, x, y, z):
        """Get all valid neighboring positions"""
        neighbors = []
        for dx, dy, dz in self.directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.is_valid_position(nx, ny, nz):
                neighbors.append((nx, ny, nz))
        return neighbors
    
    def random_walk(self, start_pos, num_steps=1000, seed=None):
        """
        Perform random walk from starting position
        
        Args:
            start_pos: (x, y, z) starting coordinates
            num_steps: Number of steps to walk
            seed: Random seed for reproducibility
            
        Returns:
            visited_positions: Set of all visited positions
            walk_path: List of positions in order visited
            success: Whether walk completed successfully
        """
        if seed is not None:
            random.seed(seed)
            
        x, y, z = start_pos
        
        # Validate starting position
        if not self.is_valid_position(x, y, z):
            print(f"ERROR: Starting position {start_pos} is not valid vessel tissue!")
            return set(), [], False
        
        visited_positions = set()
        walk_path = [(x, y, z)]
        visited_positions.add((x, y, z))
        
        stuck_count = 0
        max_stuck_attempts = 50
        
        print(f"Starting random walk from {start_pos}...")
        
        for step in range(num_steps):
            if step % 200 == 0:
                print(f"  Step {step}/{num_steps}, visited {len(visited_positions)} unique positions")
            
            # Get valid neighbors
            neighbors = self.get_valid_neighbors(x, y, z)
            
            if not neighbors:
                # We're stuck! Try to backtrack
                stuck_count += 1
                if stuck_count > max_stuck_attempts:
                    print(f"  Walk terminated early at step {step} - too many stuck attempts")
                    break
                
                # Try to backtrack to a random previous position
                if len(walk_path) > 10:
                    backtrack_pos = random.choice(walk_path[-20:])  # Choose from recent positions
                    x, y, z = backtrack_pos
                    continue
                else:
                    print(f"  Walk terminated early at step {step} - no valid moves and short path")
                    break
            
            # Choose random neighbor
            x, y, z = random.choice(neighbors)
            walk_path.append((x, y, z))
            visited_positions.add((x, y, z))
            stuck_count = 0  # Reset stuck counter
        
        print(f"Random walk completed: {len(walk_path)} steps, {len(visited_positions)} unique positions")
        return visited_positions, walk_path, True
    
    def create_visited_mask(self, visited_positions):
        """Create binary mask of visited positions"""
        mask = np.zeros(self.shape, dtype=np.uint8)
        for x, y, z in visited_positions:
            mask[x, y, z] = 1
        return mask
    
    def get_bounding_box(self, visited_positions, padding=5):
        """Get bounding box of visited positions with padding"""
        if not visited_positions:
            return None
        
        positions = list(visited_positions)
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        x_min = max(0, min(x_coords) - padding)
        x_max = min(self.shape[0], max(x_coords) + padding + 1)
        y_min = max(0, min(y_coords) - padding)
        y_max = min(self.shape[1], max(y_coords) + padding + 1)
        z_min = max(0, min(z_coords) - padding)
        z_max = min(self.shape[2], max(z_coords) + padding + 1)
        
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def crop_to_visited_region(self, visited_positions, padding=5):
        """Crop original image to region explored by random walk"""
        bbox = self.get_bounding_box(visited_positions, padding)
        if bbox is None:
            return None, None
        
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        
        # Crop the original image
        cropped_image = self.image_data[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Create visited mask for cropped region
        visited_mask = np.zeros(cropped_image.shape, dtype=np.uint8)
        for x, y, z in visited_positions:
            if (x_min <= x < x_max and 
                y_min <= y < y_max and 
                z_min <= z < z_max):
                visited_mask[x-x_min, y-y_min, z-z_min] = 1
        
        return cropped_image, visited_mask
    
    def visualize_walk_3d(self, walk_path, visited_positions, aneurysm_id, output_dir):
        """Create 3D visualization of random walk path"""
        if len(walk_path) < 2:
            return
        
        # Sample points for visualization (too many points make it unreadable)
        sample_rate = max(1, len(walk_path) // 1000)
        sampled_path = walk_path[::sample_rate]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the walk path
        x_coords = [pos[0] for pos in sampled_path]
        y_coords = [pos[1] for pos in sampled_path]
        z_coords = [pos[2] for pos in sampled_path]
        
        # Color by step number
        colors = np.linspace(0, 1, len(sampled_path))
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap='viridis', 
                           alpha=0.6, s=1)
        
        # Mark starting point
        start_pos = walk_path[0]
        ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], 
                  c='red', s=50, marker='o', label=f'Start: {start_pos}')
        
        # Mark ending point
        end_pos = walk_path[-1]
        ax.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], 
                  c='blue', s=50, marker='s', label=f'End: {end_pos}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Random Walk Path - {aneurysm_id}\n'
                    f'{len(walk_path)} steps, {len(visited_positions)} unique positions')
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='Step Number (normalized)')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{aneurysm_id}_random_walk_3d.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"3D walk visualization saved: {output_path}")

def load_manual_coordinates(json_file):
    """Load manual aneurysm coordinates from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    coordinates = {}
    
    # Handle simple format (patient_id -> aneurysms array)
    for patient_id, patient_data in data.items():
        if 'aneurysms' in patient_data:
            coordinates[patient_id] = {}
            for aneurysm in patient_data['aneurysms']:
                aneurysm_id = aneurysm['aneurysm_id']
                internal_point = aneurysm['internal_point']
                coordinates[patient_id][aneurysm_id] = {
                    'centroid': internal_point,
                    'description': aneurysm.get('description', ''),
                    'confidence': 'high'  # Default confidence
                }
    
    return coordinates

def save_nifti_with_metadata(image_data, original_nifti, output_path, description=""):
    """Save NIfTI file with preserved metadata"""
    # Create new NIfTI image with same affine transformation
    new_img = nib.Nifti1Image(image_data, original_nifti.affine, original_nifti.header)
    
    # Save the file
    nib.save(new_img, output_path)
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"Saved: {output_path}")
    print(f"  Shape: {image_data.shape}")
    print(f"  Size: {file_size:.2f} MB")
    if description:
        print(f"  Description: {description}")

def create_stl_from_nifti(nifti_data, output_path, level=0.5):
    """Create STL file from NIfTI data using marching cubes"""
    try:
        # Use marching cubes to create mesh
        verts, faces, normals, values = measure.marching_cubes(
            nifti_data, level=level, spacing=(1.0, 1.0, 1.0)
        )
        
        # Create STL content
        stl_content = "solid mesh\n"
        
        for face in faces:
            # Get the three vertices of the triangle
            v1, v2, v3 = verts[face]
            
            # Calculate normal (though marching cubes provides normals, we calculate for consistency)
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            
            stl_content += f"facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
            stl_content += "outer loop\n"
            stl_content += f"vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
            stl_content += f"vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
            stl_content += f"vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n"
            stl_content += "endloop\n"
            stl_content += "endfacet\n"
        
        stl_content += "endsolid mesh\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(stl_content)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"STL saved: {output_path}")
        print(f"  Vertices: {len(verts)}")
        print(f"  Faces: {len(faces)}")
        print(f"  Size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error creating STL: {e}")
        return False

def process_all_patients(coordinates_json, nifti_base_dir, steps=2000, padding=10, seed=None):
    """Process all patients with their corresponding NIfTI files"""
    print("=== Automated Random Walk Processing ===")
    print(f"Coordinates file: {coordinates_json}")
    print(f"NIfTI base directory: {nifti_base_dir}")
    print(f"Random walk steps: {steps}")
    print(f"Crop padding: {padding}")
    if seed:
        print(f"Random seed: {seed}")
    print()
    
    # Load manual coordinates
    try:
        coordinates = load_manual_coordinates(coordinates_json)
        if not coordinates:
            print("No coordinates found in JSON file!")
            return 1
        print(f"Loaded coordinates for {len(coordinates)} patients")
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return 1
    
    # Process each patient
    total_patients = 0
    successful_patients = 0
    total_aneurysms = 0
    successful_aneurysms = 0
    
    for patient_id in coordinates.keys():
        total_patients += 1
        
        # Find corresponding NIfTI file
        nifti_file = os.path.join(nifti_base_dir, f"{patient_id}.nii.gz")
        
        if not os.path.exists(nifti_file):
            print(f"❌ NIfTI file not found for {patient_id}: {nifti_file}")
            continue
        
        print(f"\n🔄 Processing Patient: {patient_id}")
        print(f"📁 NIfTI file: {nifti_file}")
        
        # Load NIfTI file
        try:
            nifti_img = nib.load(nifti_file)
            image_data = nifti_img.get_fdata()
            print(f"📊 Loaded NIfTI: shape {image_data.shape}, range [{image_data.min():.3f}, {image_data.max():.3f}]")
        except Exception as e:
            print(f"❌ Error loading NIfTI file: {e}")
            continue
        
        # Create output directory for this patient
        output_dir = f"random_walk_{patient_id}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cropped_nifti"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cropped_stl"), exist_ok=True)
        
        # Initialize random walk for this patient
        walker = VesselRandomWalk(image_data, vessel_threshold=0.5)
        
        # Process aneurysms for this patient
        patient_aneurysms = coordinates[patient_id]
        patient_successful = 0
        
        for aneurysm_id, aneurysm_data in patient_aneurysms.items():
            total_aneurysms += 1
            centroid = aneurysm_data['centroid']
            description = aneurysm_data['description']
            
            print(f"\n  🎯 Processing {aneurysm_id}:")
            print(f"    📍 Centroid: {centroid}")
            print(f"    📝 Description: {description}")
            
            # Perform random walk
            start_time = time.time()
            visited_positions, walk_path, success = walker.random_walk(
                tuple(centroid), num_steps=steps, seed=seed
            )
            walk_time = time.time() - start_time
            
            if not success or not visited_positions:
                print(f"    ❌ Random walk failed for {aneurysm_id}")
                continue
            
            successful_aneurysms += 1
            patient_successful += 1
            print(f"    ✅ Random walk completed in {walk_time:.2f}s - {len(visited_positions)} positions explored")
            
            # Create visualization
            walker.visualize_walk_3d(
                walk_path, visited_positions, f"{patient_id}_{aneurysm_id}", 
                os.path.join(output_dir, "visualizations")
            )
            
            # Crop to visited region
            cropped_image, visited_mask = walker.crop_to_visited_region(
                visited_positions, padding=padding
            )
            
            if cropped_image is not None:
                # Save cropped NIfTI files
                cropped_nifti_path = os.path.join(
                    output_dir, "cropped_nifti", f"{aneurysm_id}_cropped_vessel.nii.gz"
                )
                save_nifti_with_metadata(
                    cropped_image, nifti_img, cropped_nifti_path,
                    f"Vessel region explored by random walk from {aneurysm_id}"
                )
                
                # Save visited mask
                mask_nifti_path = os.path.join(
                    output_dir, "cropped_nifti", f"{aneurysm_id}_walk_mask.nii.gz"
                )
                save_nifti_with_metadata(
                    visited_mask, nifti_img, mask_nifti_path,
                    f"Binary mask of positions visited by random walk"
                )
                
                # Create STL files
                stl_path = os.path.join(
                    output_dir, "cropped_stl", f"{aneurysm_id}_cropped_vessel.stl"
                )
                create_stl_from_nifti(cropped_image, stl_path, level=0.5)
                
                # Create STL of walk path
                walk_stl_path = os.path.join(
                    output_dir, "cropped_stl", f"{aneurysm_id}_walk_path.stl"
                )
                create_stl_from_nifti(visited_mask.astype(float), walk_stl_path, level=0.5)
        
        if patient_successful > 0:
            successful_patients += 1
            print(f"  ✅ Patient {patient_id}: {patient_successful}/{len(patient_aneurysms)} aneurysms successful")
        else:
            print(f"  ❌ Patient {patient_id}: No successful aneurysms")
    
    # Final summary
    print(f"\n=== FINAL PROCESSING SUMMARY ===")
    print(f"📊 Patients processed: {successful_patients}/{total_patients}")
    print(f"🎯 Aneurysms processed: {successful_aneurysms}/{total_aneurysms}")
    print(f"📁 Output directories: random_walk_[PATIENT_ID]/")
    
    if successful_aneurysms > 0:
        print(f"\n🎉 Processing completed successfully!")
        print(f"💾 Check random_walk_*/ directories for results:")
        print(f"   - visualizations/: 3D plots of walk paths")
        print(f"   - cropped_nifti/: Cropped vessel regions and walk masks")
        print(f"   - cropped_stl/: 3D STL models for visualization")
        return 0
    else:
        print(f"\n❌ No successful random walks completed")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Random Walk Vessel Exploration')
    parser.add_argument('--coordinates', default='all_patients_aneurysms.json', 
                       help='JSON file with aneurysm coordinates (default: all_patients_aneurysms.json)')
    parser.add_argument('--nifti-dir', default='~/urp/data/uan/original/', 
                       help='Directory containing NIfTI files (default: ~/urp/data/uan/original/)')
    parser.add_argument('--steps', type=int, default=2000, help='Number of random walk steps (default: 2000)')
    parser.add_argument('--padding', type=int, default=10, help='Padding around cropped region (default: 10)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--single-patient', help='Process only a specific patient ID')
    
    args = parser.parse_args()
    
    # Expand tilde in path
    nifti_dir = os.path.expanduser(args.nifti_dir)
    
    if args.single_patient:
        print("=== Single Patient Mode ===")
        # Legacy mode for single patient processing
        nifti_file = os.path.join(nifti_dir, f"{args.single_patient}.nii.gz")
        if not os.path.exists(nifti_file):
            print(f"❌ NIfTI file not found: {nifti_file}")
            return 1
        
        # Process single patient using original logic
        return process_single_patient(nifti_file, args.coordinates, args.steps, args.padding, args.seed)
    else:
        # New automated mode for all patients
        return process_all_patients(args.coordinates, nifti_dir, args.steps, args.padding, args.seed)

def process_single_patient(nifti_file, coordinates_json, steps, padding, seed):
    """Legacy function for single patient processing"""
    print("=== Single Patient Random Walk ===")
    print(f"NIfTI file: {nifti_file}")
    print(f"Coordinates: {coordinates_json}")
    print(f"Steps: {steps}")
    
    # Load NIfTI file
    try:
        nifti_img = nib.load(nifti_file)
        image_data = nifti_img.get_fdata()
        print(f"Loaded NIfTI: shape {image_data.shape}, range [{image_data.min():.3f}, {image_data.max():.3f}]")
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return 1
    
    # Load coordinates
    try:
        coordinates = load_manual_coordinates(coordinates_json)
        if not coordinates:
            print("No coordinates found in JSON file!")
            return 1
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return 1
    
    # Extract patient ID from NIfTI filename
    base_name = os.path.splitext(os.path.basename(nifti_file))[0]
    if base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0]
    
    # Find matching patient in coordinates
    target_patient = None
    for patient_id in coordinates.keys():
        if patient_id == base_name:
            target_patient = patient_id
            break
    
    if not target_patient:
        print(f"❌ No coordinates found for patient: {base_name}")
        return 1
    
    # Process only the matching patient
    output_dir = f"random_walk_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cropped_nifti"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cropped_stl"), exist_ok=True)
    
    walker = VesselRandomWalk(image_data, vessel_threshold=0.5)
    
    successful_walks = 0
    for aneurysm_id, aneurysm_data in coordinates[target_patient].items():
        centroid = aneurysm_data['centroid']
        description = aneurysm_data['description']
        
        print(f"\nProcessing {aneurysm_id}: {centroid}")
        
        visited_positions, walk_path, success = walker.random_walk(
            tuple(centroid), num_steps=steps, seed=seed
        )
        
        if success and visited_positions:
            successful_walks += 1
            print(f"✅ Success: {len(visited_positions)} positions explored")
            
            # Generate all outputs
            walker.visualize_walk_3d(walk_path, visited_positions, aneurysm_id, 
                                   os.path.join(output_dir, "visualizations"))
            
            cropped_image, visited_mask = walker.crop_to_visited_region(visited_positions, padding)
            
            if cropped_image is not None:
                # Save files
                save_nifti_with_metadata(cropped_image, nifti_img, 
                                       os.path.join(output_dir, "cropped_nifti", f"{aneurysm_id}_cropped_vessel.nii.gz"),
                                       f"Vessel region from {aneurysm_id}")
                save_nifti_with_metadata(visited_mask, nifti_img,
                                       os.path.join(output_dir, "cropped_nifti", f"{aneurysm_id}_walk_mask.nii.gz"),
                                       f"Walk mask from {aneurysm_id}")
                create_stl_from_nifti(cropped_image, 
                                    os.path.join(output_dir, "cropped_stl", f"{aneurysm_id}_cropped_vessel.stl"))
                create_stl_from_nifti(visited_mask.astype(float),
                                    os.path.join(output_dir, "cropped_stl", f"{aneurysm_id}_walk_path.stl"))
        else:
            print(f"❌ Failed: {aneurysm_id}")
    
    print(f"\n=== Summary ===")
    print(f"Successful walks: {successful_walks}")
    print(f"Output: {output_dir}/")
    
    return 0 if successful_walks > 0 else 1

if __name__ == "__main__":
    exit(main()) 