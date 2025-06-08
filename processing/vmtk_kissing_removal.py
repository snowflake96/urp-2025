#!/usr/bin/env python3
"""
VMTK-based Kissing Artifact Detection and Removal

Uses the Vascular Modeling Toolkit (VMTK) for:
1. Centerline extraction to understand vessel topology
2. Branch analysis to detect abnormal connections
3. Surface editing to separate kissing vessels

Installation:
    conda install -c vmtk vtk itk vmtk
    or
    pip install vmtk
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
import subprocess

try:
    import vmtk
    from vmtk import vmtkscripts
except ImportError:
    print("ERROR: VMTK not installed. Please install with:")
    print("  conda install -c vmtk vtk itk vmtk")
    print("  or")
    print("  pip install vmtk")
    raise

import vtk
from vtk.util import numpy_support
import trimesh
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.measure import marching_cubes


class VMTKKissingRemoval:
    """Use VMTK for kissing artifact detection and removal"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('VMTKKissing')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_nifti(self, input_nifti: str, output_nifti: str,
                     min_vessel_radius: float = 0.5,
                     max_vessel_radius: float = 5.0,
                     centerline_resampling: float = 0.5) -> Dict:
        """
        Process NIfTI to detect and remove kissing artifacts
        
        Parameters:
        -----------
        input_nifti : str
            Input segmented vessel NIfTI
        output_nifti : str
            Output cleaned NIfTI
        min_vessel_radius : float
            Minimum vessel radius in mm
        max_vessel_radius : float
            Maximum vessel radius in mm
        centerline_resampling : float
            Centerline resampling distance in mm
        """
        self.logger.info(f"Processing {input_nifti} with VMTK")
        
        # Load NIfTI
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        
        # Convert to binary mask
        mask = volume > 0.5
        
        # Step 1: Convert to VTK surface
        self.logger.info("Step 1: Converting to VTK surface...")
        surface = self._nifti_to_vtk_surface(mask, affine)
        
        # Step 2: Extract centerlines
        self.logger.info("Step 2: Extracting vessel centerlines...")
        centerlines = self._extract_centerlines(surface)
        
        if centerlines is None:
            self.logger.warning("Failed to extract centerlines, returning original")
            nib.save(nib.Nifti1Image(volume, affine), output_nifti)
            return {'status': 'failed', 'reason': 'centerline_extraction_failed'}
        
        # Step 3: Analyze topology
        self.logger.info("Step 3: Analyzing vessel topology...")
        topology = self._analyze_topology(centerlines)
        
        # Step 4: Detect kissing artifacts
        self.logger.info("Step 4: Detecting kissing artifacts...")
        kissing_regions = self._detect_kissing_artifacts(
            surface, centerlines, topology,
            min_radius=min_vessel_radius,
            max_radius=max_vessel_radius
        )
        
        self.logger.info(f"Found {len(kissing_regions)} potential kissing artifacts")
        
        # Step 5: Remove kissing artifacts
        if kissing_regions:
            self.logger.info("Step 5: Removing kissing artifacts...")
            cleaned_mask = self._remove_kissing_artifacts(
                mask, kissing_regions, affine
            )
        else:
            self.logger.info("No kissing artifacts found")
            cleaned_mask = mask
        
        # Save result
        cleaned_volume = cleaned_mask.astype(np.float32)
        nib.save(nib.Nifti1Image(cleaned_volume, affine), output_nifti)
        
        # Statistics
        stats = {
            'status': 'success',
            'input_voxels': int(mask.sum()),
            'output_voxels': int(cleaned_mask.sum()),
            'voxels_removed': int(mask.sum() - cleaned_mask.sum()),
            'kissing_artifacts_found': len(kissing_regions),
            'topology': topology
        }
        
        self.logger.info(f"Removed {stats['voxels_removed']} voxels")
        
        return stats
    
    def _nifti_to_vtk_surface(self, mask: np.ndarray, affine: np.ndarray) -> vtk.vtkPolyData:
        """Convert binary mask to VTK surface"""
        # Use marching cubes
        verts, faces, normals, values = marching_cubes(mask, level=0.5)
        
        # Apply affine transformation
        verts_transformed = np.dot(
            np.hstack([verts, np.ones((len(verts), 1))]),
            affine.T
        )[:, :3]
        
        # Create VTK polydata
        points = vtk.vtkPoints()
        for v in verts_transformed:
            points.InsertNextPoint(v)
        
        polys = vtk.vtkCellArray()
        for f in faces:
            polys.InsertNextCell(3, f)
        
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(polys)
        
        # Clean and smooth
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surface)
        cleaner.Update()
        
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(cleaner.GetOutputPort())
        smoother.SetNumberOfIterations(20)
        smoother.SetPassBand(0.1)
        smoother.Update()
        
        return smoother.GetOutput()
    
    def _extract_centerlines(self, surface: vtk.vtkPolyData) -> Optional[vtk.vtkPolyData]:
        """Extract vessel centerlines using VMTK"""
        try:
            # Use VMTK centerline extraction
            centerlineFilter = vmtkscripts.vmtkCenterlines()
            centerlineFilter.Surface = surface
            centerlineFilter.SeedSelectorName = 'openprofiles'  # Automatic seed selection
            centerlineFilter.AppendEndPoints = 1
            centerlineFilter.Resampling = 1
            centerlineFilter.ResamplingStepLength = 0.5
            centerlineFilter.Execute()
            
            return centerlineFilter.Centerlines
            
        except Exception as e:
            self.logger.error(f"Centerline extraction failed: {e}")
            return None
    
    def _analyze_topology(self, centerlines: vtk.vtkPolyData) -> Dict:
        """Analyze vessel topology from centerlines"""
        topology = {
            'num_branches': 0,
            'num_bifurcations': 0,
            'num_endpoints': 0,
            'suspicious_loops': []
        }
        
        if centerlines is None:
            return topology
        
        # Get branch information
        branchExtractor = vmtkscripts.vmtkBranchExtractor()
        branchExtractor.Centerlines = centerlines
        branchExtractor.Execute()
        
        topology['num_branches'] = branchExtractor.Centerlines.GetNumberOfCells()
        
        # Analyze each branch
        for i in range(branchExtractor.Centerlines.GetNumberOfCells()):
            cell = branchExtractor.Centerlines.GetCell(i)
            num_points = cell.GetNumberOfPoints()
            
            if num_points < 2:
                continue
            
            # Check for loops (branch that connects back)
            start_point = np.array(cell.GetPoints().GetPoint(0))
            end_point = np.array(cell.GetPoints().GetPoint(num_points - 1))
            
            distance = np.linalg.norm(end_point - start_point)
            branch_length = self._compute_branch_length(cell)
            
            # If endpoints are close but branch is long, might be a loop
            if distance < 5.0 and branch_length > 20.0:
                topology['suspicious_loops'].append({
                    'branch_id': i,
                    'endpoint_distance': float(distance),
                    'branch_length': float(branch_length)
                })
        
        # Count bifurcations
        bifurcationReferenceSystems = vmtkscripts.vmtkBifurcationReferenceSystems()
        bifurcationReferenceSystems.Centerlines = centerlines
        bifurcationReferenceSystems.Execute()
        
        if bifurcationReferenceSystems.ReferenceSystems:
            topology['num_bifurcations'] = bifurcationReferenceSystems.ReferenceSystems.GetNumberOfPoints()
        
        return topology
    
    def _detect_kissing_artifacts(self, surface: vtk.vtkPolyData, 
                                 centerlines: vtk.vtkPolyData,
                                 topology: Dict,
                                 min_radius: float = 0.5,
                                 max_radius: float = 5.0) -> List[Dict]:
        """Detect kissing artifacts using centerline and surface analysis"""
        kissing_regions = []
        
        # Method 1: Check for abnormal vessel thickness
        self.logger.info("  Checking vessel thickness variations...")
        thickness_artifacts = self._detect_thickness_anomalies(
            surface, centerlines, min_radius, max_radius
        )
        kissing_regions.extend(thickness_artifacts)
        
        # Method 2: Check for loops in topology
        if topology['suspicious_loops']:
            self.logger.info(f"  Found {len(topology['suspicious_loops'])} suspicious loops")
            for loop in topology['suspicious_loops']:
                kissing_regions.append({
                    'type': 'topological_loop',
                    'location': self._get_loop_center(centerlines, loop['branch_id']),
                    'score': 0.8,
                    'details': loop
                })
        
        # Method 3: Check for abnormal surface proximity
        self.logger.info("  Checking surface proximity...")
        proximity_artifacts = self._detect_proximity_artifacts(surface)
        kissing_regions.extend(proximity_artifacts)
        
        return kissing_regions
    
    def _detect_thickness_anomalies(self, surface: vtk.vtkPolyData, 
                                   centerlines: vtk.vtkPolyData,
                                   min_radius: float, max_radius: float) -> List[Dict]:
        """Detect regions with abnormal vessel thickness"""
        artifacts = []
        
        # Compute distance from surface to centerline
        distanceFilter = vtk.vtkDistancePolyDataFilter()
        distanceFilter.SetInputData(0, surface)
        distanceFilter.SetInputData(1, centerlines)
        distanceFilter.Update()
        
        distances = distanceFilter.GetOutput()
        distance_array = distances.GetPointData().GetArray("Distance")
        
        if distance_array:
            # Find points with abnormal thickness
            for i in range(distances.GetNumberOfPoints()):
                dist = distance_array.GetValue(i)
                
                # Too thick (potential kissing artifact)
                if dist > max_radius * 1.5:
                    point = distances.GetPoint(i)
                    artifacts.append({
                        'type': 'thickness_anomaly',
                        'location': point,
                        'thickness': dist,
                        'score': min(1.0, dist / (max_radius * 2))
                    })
        
        # Cluster nearby anomalies
        if artifacts:
            return self._cluster_artifacts(artifacts, cluster_radius=5.0)
        
        return []
    
    def _detect_proximity_artifacts(self, surface: vtk.vtkPolyData) -> List[Dict]:
        """Detect regions where surface is too close to itself"""
        artifacts = []
        
        # Use VTK collision detection
        collisionFilter = vtk.vtkCollisionDetectionFilter()
        collisionFilter.SetInputData(0, surface)
        collisionFilter.SetInputData(1, surface)
        collisionFilter.SetBoxTolerance(0.001)
        collisionFilter.SetCellTolerance(0.0)
        collisionFilter.SetNumberOfCellsPerNode(2)
        collisionFilter.SetCollisionModeToFirstContact()
        collisionFilter.GenerateScalarsOn()
        collisionFilter.Update()
        
        contacts = collisionFilter.GetContactsOutput()
        
        if contacts and contacts.GetNumberOfCells() > 0:
            # Extract contact points
            for i in range(contacts.GetNumberOfCells()):
                cell = contacts.GetCell(i)
                if cell.GetNumberOfPoints() >= 2:
                    p1 = np.array(cell.GetPoints().GetPoint(0))
                    p2 = np.array(cell.GetPoints().GetPoint(1))
                    center = (p1 + p2) / 2
                    
                    artifacts.append({
                        'type': 'surface_proximity',
                        'location': center.tolist(),
                        'distance': float(np.linalg.norm(p2 - p1)),
                        'score': 0.9
                    })
        
        return self._cluster_artifacts(artifacts, cluster_radius=3.0)
    
    def _remove_kissing_artifacts(self, mask: np.ndarray, 
                                 kissing_regions: List[Dict],
                                 affine: np.ndarray) -> np.ndarray:
        """Remove kissing artifacts from the mask"""
        cleaned_mask = mask.copy()
        
        # Convert world coordinates to voxel coordinates
        inv_affine = np.linalg.inv(affine)
        
        for region in kissing_regions:
            if region['score'] < 0.5:  # Skip low confidence
                continue
            
            # Get region center in voxel coordinates
            world_coord = np.array(region['location'] + [1])
            voxel_coord = np.dot(inv_affine, world_coord)[:3]
            voxel_coord = np.round(voxel_coord).astype(int)
            
            # Remove a sphere of voxels around the artifact
            radius_voxels = int(region.get('radius', 3.0) / np.min(np.abs(affine.diagonal()[:3])))
            
            # Create sphere mask
            x, y, z = np.ogrid[-radius_voxels:radius_voxels+1,
                              -radius_voxels:radius_voxels+1,
                              -radius_voxels:radius_voxels+1]
            sphere = x**2 + y**2 + z**2 <= radius_voxels**2
            
            # Apply removal
            x_start = max(0, voxel_coord[0] - radius_voxels)
            x_end = min(mask.shape[0], voxel_coord[0] + radius_voxels + 1)
            y_start = max(0, voxel_coord[1] - radius_voxels)
            y_end = min(mask.shape[1], voxel_coord[1] + radius_voxels + 1)
            z_start = max(0, voxel_coord[2] - radius_voxels)
            z_end = min(mask.shape[2], voxel_coord[2] + radius_voxels + 1)
            
            # Adjust sphere mask if near boundaries
            sphere_x_start = radius_voxels - (voxel_coord[0] - x_start)
            sphere_x_end = sphere_x_start + (x_end - x_start)
            sphere_y_start = radius_voxels - (voxel_coord[1] - y_start)
            sphere_y_end = sphere_y_start + (y_end - y_start)
            sphere_z_start = radius_voxels - (voxel_coord[2] - z_start)
            sphere_z_end = sphere_z_start + (z_end - z_start)
            
            cleaned_mask[x_start:x_end, y_start:y_end, z_start:z_end] &= \
                ~sphere[sphere_x_start:sphere_x_end,
                       sphere_y_start:sphere_y_end,
                       sphere_z_start:sphere_z_end]
        
        # Clean up small disconnected components
        labeled, num_labels = label(cleaned_mask)
        if num_labels > 1:
            # Keep only the largest component
            sizes = [np.sum(labeled == i) for i in range(1, num_labels + 1)]
            largest = np.argmax(sizes) + 1
            cleaned_mask = (labeled == largest)
        
        return cleaned_mask
    
    def _compute_branch_length(self, cell: vtk.vtkCell) -> float:
        """Compute the length of a branch"""
        length = 0.0
        for i in range(1, cell.GetNumberOfPoints()):
            p1 = np.array(cell.GetPoints().GetPoint(i-1))
            p2 = np.array(cell.GetPoints().GetPoint(i))
            length += np.linalg.norm(p2 - p1)
        return length
    
    def _get_loop_center(self, centerlines: vtk.vtkPolyData, branch_id: int) -> List[float]:
        """Get the center of a loop"""
        cell = centerlines.GetCell(branch_id)
        center = np.zeros(3)
        for i in range(cell.GetNumberOfPoints()):
            center += np.array(cell.GetPoints().GetPoint(i))
        center /= cell.GetNumberOfPoints()
        return center.tolist()
    
    def _cluster_artifacts(self, artifacts: List[Dict], cluster_radius: float) -> List[Dict]:
        """Cluster nearby artifacts"""
        if not artifacts:
            return []
        
        # Simple clustering based on distance
        clusters = []
        used = set()
        
        for i, artifact in enumerate(artifacts):
            if i in used:
                continue
            
            cluster = {
                'type': artifact['type'] + '_cluster',
                'location': artifact['location'],
                'score': artifact['score'],
                'radius': cluster_radius,
                'artifacts': [artifact]
            }
            
            # Find nearby artifacts
            for j, other in enumerate(artifacts[i+1:], i+1):
                if j not in used:
                    dist = np.linalg.norm(
                        np.array(artifact['location']) - np.array(other['location'])
                    )
                    if dist < cluster_radius:
                        used.add(j)
                        cluster['artifacts'].append(other)
                        # Update cluster center
                        all_locs = np.array([a['location'] for a in cluster['artifacts']])
                        cluster['location'] = all_locs.mean(axis=0).tolist()
                        cluster['score'] = max(a['score'] for a in cluster['artifacts'])
            
            clusters.append(cluster)
        
        return clusters


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VMTK-based kissing artifact removal"
    )
    parser.add_argument('input_nifti', help='Input segmented vessel NIfTI')
    parser.add_argument('output_nifti', help='Output cleaned NIfTI')
    parser.add_argument('--min-radius', type=float, default=0.5,
                       help='Minimum vessel radius in mm')
    parser.add_argument('--max-radius', type=float, default=5.0,
                       help='Maximum vessel radius in mm')
    parser.add_argument('--centerline-resampling', type=float, default=0.5,
                       help='Centerline resampling distance in mm')
    
    args = parser.parse_args()
    
    # Process
    processor = VMTKKissingRemoval()
    stats = processor.process_nifti(
        args.input_nifti,
        args.output_nifti,
        min_vessel_radius=args.min_radius,
        max_vessel_radius=args.max_radius,
        centerline_resampling=args.centerline_resampling
    )
    
    print(f"\nProcessing complete!")
    print(f"Kissing artifacts found: {stats['kissing_artifacts_found']}")
    print(f"Voxels removed: {stats['voxels_removed']}")
    print(f"Output saved to: {args.output_nifti}")


if __name__ == "__main__":
    main() 