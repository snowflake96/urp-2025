#!/usr/bin/env python3
"""
Create Extraction Settings

This script creates a configuration JSON file that specifies extraction parameters
for each patient. This file serves as input settings for the extraction process.
"""

import json
import os
import numpy as np
import argparse
from typing import Dict, List


def create_patient_extraction_settings(patient_id: str, aneurysm_data: Dict) -> Dict:
    """
    Create extraction settings for a single patient based on aneurysm characteristics.
    """
    aneurysm_info = aneurysm_data['aneurysms'][0]  # First aneurysm
    aneurysm_location = aneurysm_info['mesh_vertex_coords']
    
    # Default extraction parameters
    base_settings = {
        'extraction_method': 'geodesic_small_orthogonal',
        'aneurysm_location': aneurysm_location,
        'target_vertices': 4500,
        'max_distance_mm': 16.0,
        'min_distance_mm': 0.0,
        'orthogonal_constraints': True,
        'vessel_direction_analysis': True,
        'constraint_radius_mm': 12.0,
        'max_angle_deviation_deg': 35.0
    }
    
    # Adjust settings based on patient ID or characteristics
    # You can customize these rules based on your needs
    
    # For some patients, use larger extraction radius
    if patient_id.startswith(('06_', '07_', '08_')):
        base_settings['max_distance_mm'] = 16.0
        base_settings['target_vertices'] = 4500
    elif patient_id.startswith(('09_', '11_')):
        base_settings['max_distance_mm'] = 15.0
        base_settings['target_vertices'] = 4200
    else:
        # Default for other patients
        base_settings['max_distance_mm'] = 18.0
        base_settings['target_vertices'] = 5000
    
    # Adjust based on MRA1 vs MRA2
    if 'MRA1' in patient_id:
        # MRA1 might need slightly larger extraction
        base_settings['max_distance_mm'] *= 1.1
        base_settings['target_vertices'] = int(base_settings['target_vertices'] * 1.1)
    elif 'MRA2' in patient_id:
        # MRA2 might need more conservative extraction
        base_settings['max_distance_mm'] *= 0.95
        base_settings['target_vertices'] = int(base_settings['target_vertices'] * 0.95)
    
    # Add patient-specific overrides if needed
    patient_overrides = {
        # Example overrides for specific patients
        '06_MRA2_seg': {
            'max_distance_mm': 14.0,
            'target_vertices': 4000,
            'comment': 'Smaller extraction for testing'
        },
        '07_MRA1_seg': {
            'max_distance_mm': 18.0,
            'target_vertices': 4800,
            'comment': 'Larger extraction needed'
        }
        # Add more patient-specific settings as needed
    }
    
    if patient_id in patient_overrides:
        overrides = patient_overrides[patient_id]
        for key, value in overrides.items():
            if key != 'comment':
                base_settings[key] = value
        base_settings['override_reason'] = overrides.get('comment', 'Custom settings')
    
    # Ensure reasonable bounds
    base_settings['max_distance_mm'] = max(8.0, min(25.0, base_settings['max_distance_mm']))
    base_settings['target_vertices'] = max(2000, min(8000, base_settings['target_vertices']))
    
    return base_settings


def main():
    """Main function to create extraction settings file"""
    parser = argparse.ArgumentParser(description='Create Extraction Settings File')
    
    parser.add_argument('--aneurysm-json',
                       default='../all_patients_aneurysms_for_stl.json',
                       help='JSON file with aneurysm coordinates')
    
    parser.add_argument('--output-json',
                       default='../vessel_extraction_settings.json',
                       help='Output JSON file with extraction settings')
    
    args = parser.parse_args()
    
    # Load aneurysm data
    print(f"Loading aneurysm data from: {args.aneurysm_json}")
    with open(args.aneurysm_json, 'r') as f:
        aneurysm_data = json.load(f)
    
    print(f"Creating extraction settings for {len(aneurysm_data)} patients...")
    
    # Create settings for all patients
    extraction_settings = {
        'metadata': {
            'created_date': '2025-06-08',
            'description': 'Extraction settings for geodesic small orthogonal vessel extraction',
            'extraction_method': 'geodesic_small_orthogonal',
            'total_patients': len(aneurysm_data),
            'parameter_explanations': {
                'max_distance_mm': 'Maximum distance from aneurysm location to extract (mm)',
                'target_vertices': 'Target number of vertices in extracted vessel',
                'orthogonal_constraints': 'Apply vessel direction constraints for circular openings',
                'constraint_radius_mm': 'Distance from aneurysm where constraints start (mm)',
                'max_angle_deviation_deg': 'Maximum angle deviation from vessel direction (degrees)'
            }
        },
        'default_settings': {
            'extraction_method': 'geodesic_small_orthogonal',
            'target_vertices': 4500,
            'max_distance_mm': 16.0,
            'min_distance_mm': 0.0,
            'orthogonal_constraints': True,
            'vessel_direction_analysis': True,
            'constraint_radius_mm': 12.0,
            'max_angle_deviation_deg': 35.0
        },
        'size_categories': {
            'small_extraction': {
                'max_distance_mm': 12.0,
                'target_vertices': 3500,
                'description': 'Conservative extraction for complex anatomy'
            },
            'medium_extraction': {
                'max_distance_mm': 16.0,
                'target_vertices': 4500,
                'description': 'Standard extraction for most cases'
            },
            'large_extraction': {
                'max_distance_mm': 20.0,
                'target_vertices': 6000,
                'description': 'Extended extraction for large vessels'
            }
        },
        'patient_settings': {}
    }
    
    # Generate settings for each patient
    for patient_id, patient_data in aneurysm_data.items():
        patient_settings = create_patient_extraction_settings(patient_id, patient_data)
        extraction_settings['patient_settings'][patient_id] = patient_settings
        
        # Determine size category
        max_dist = patient_settings['max_distance_mm']
        if max_dist <= 13.0:
            category = 'small_extraction'
        elif max_dist <= 17.0:
            category = 'medium_extraction'
        else:
            category = 'large_extraction'
        
        patient_settings['size_category'] = category
        
        print(f"  {patient_id}: {category} ({max_dist:.1f}mm, {patient_settings['target_vertices']} vertices)")
    
    # Generate summary statistics
    all_max_distances = [settings['max_distance_mm'] for settings in extraction_settings['patient_settings'].values()]
    all_target_vertices = [settings['target_vertices'] for settings in extraction_settings['patient_settings'].values()]
    
    extraction_settings['summary'] = {
        'distance_statistics': {
            'mean_max_distance_mm': float(np.mean(all_max_distances)),
            'std_max_distance_mm': float(np.std(all_max_distances)),
            'min_max_distance_mm': float(np.min(all_max_distances)),
            'max_max_distance_mm': float(np.max(all_max_distances))
        },
        'vertex_statistics': {
            'mean_target_vertices': float(np.mean(all_target_vertices)),
            'std_target_vertices': float(np.std(all_target_vertices)),
            'min_target_vertices': int(np.min(all_target_vertices)),
            'max_target_vertices': int(np.max(all_target_vertices))
        },
        'category_counts': {
            'small_extractions': sum(1 for s in extraction_settings['patient_settings'].values() if s['size_category'] == 'small_extraction'),
            'medium_extractions': sum(1 for s in extraction_settings['patient_settings'].values() if s['size_category'] == 'medium_extraction'),
            'large_extractions': sum(1 for s in extraction_settings['patient_settings'].values() if s['size_category'] == 'large_extraction')
        }
    }
    
    # Save settings file
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(extraction_settings, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Extraction Settings File Created")
    print(f"Total patients: {len(aneurysm_data)}")
    
    print(f"\nDistance Settings Summary:")
    print(f"  Average max distance: {extraction_settings['summary']['distance_statistics']['mean_max_distance_mm']:.1f} ± {extraction_settings['summary']['distance_statistics']['std_max_distance_mm']:.1f} mm")
    print(f"  Range: {extraction_settings['summary']['distance_statistics']['min_max_distance_mm']:.1f} - {extraction_settings['summary']['distance_statistics']['max_max_distance_mm']:.1f} mm")
    
    print(f"\nVertex Settings Summary:")
    print(f"  Average target vertices: {extraction_settings['summary']['vertex_statistics']['mean_target_vertices']:.0f} ± {extraction_settings['summary']['vertex_statistics']['std_target_vertices']:.0f}")
    print(f"  Range: {extraction_settings['summary']['vertex_statistics']['min_target_vertices']} - {extraction_settings['summary']['vertex_statistics']['max_target_vertices']}")
    
    print(f"\nExtraction Categories:")
    print(f"  Small extractions: {extraction_settings['summary']['category_counts']['small_extractions']}")
    print(f"  Medium extractions: {extraction_settings['summary']['category_counts']['medium_extractions']}")
    print(f"  Large extractions: {extraction_settings['summary']['category_counts']['large_extractions']}")
    
    print(f"\nSettings file saved to: {args.output_json}")
    print(f"🎯 Extraction settings configured for all patients!")
    
    # Show example usage
    print(f"\nExample usage:")
    print(f"  python extract_vessels_geodesic_small_orthogonal.py --settings-file {args.output_json} --patient-id 06_MRA2_seg")
    print(f"  # This would use: max_distance={extraction_settings['patient_settings'].get('06_MRA2_seg', {}).get('max_distance_mm', 'N/A')}mm, target_vertices={extraction_settings['patient_settings'].get('06_MRA2_seg', {}).get('target_vertices', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 