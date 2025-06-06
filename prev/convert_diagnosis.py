"""
Convert diagnosis Excel file to JSON format for aneurysm detection
"""

import pandas as pd
import json
from pathlib import Path

def convert_diagnosis_excel_to_json(excel_path, output_path=None):
    """
    Convert diagnosis Excel file to JSON format.
    
    Args:
        excel_path (str): Path to the Excel file
        output_path (str): Output JSON file path
    """
    if output_path is None:
        output_path = "diagnosis/uan_diagnosis.json"
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        print(f"Excel file loaded: {excel_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Create diagnosis dictionary
        diagnosis_data = {}
        
        # Process each row
        for idx, row in df.iterrows():
            # Try to find patient ID or case ID
            patient_id = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['patient', 'case', 'id', 'subject']):
                    patient_id = str(row[col]) if not pd.isna(row[col]) else f"patient_{idx+1}"
                    break
            
            if patient_id is None:
                patient_id = f"patient_{idx+1}"
            
            # Extract relevant information
            patient_data = {}
            
            # Look for aneurysm count
            aneurysm_count = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['aneurysm', 'aneursym', 'count', 'number']):
                    if not pd.isna(row[col]):
                        try:
                            aneurysm_count = int(float(row[col]))
                        except:
                            # Try to extract number from text
                            import re
                            match = re.search(r'\d+', str(row[col]))
                            if match:
                                aneurysm_count = int(match.group())
                    break
            
            # Look for location information
            locations = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['location', 'site', 'position', 'artery']):
                    if not pd.isna(row[col]):
                        locations.append(str(row[col]))
            
            # Look for size information
            sizes = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['size', 'diameter', 'volume', 'mm']):
                    if not pd.isna(row[col]):
                        try:
                            size_val = float(row[col])
                            sizes.append(size_val)
                        except:
                            sizes.append(str(row[col]))
            
            # Store patient data
            patient_data = {
                'patient_id': patient_id,
                'expected_aneurysm_count': aneurysm_count,
                'locations': locations,
                'sizes': sizes,
                'raw_data': row.to_dict()  # Keep all original data
            }
            
            diagnosis_data[patient_id] = patient_data
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(diagnosis_data, f, indent=2, default=str)
        
        print(f"\nDiagnosis data saved to: {output_path}")
        print(f"Total patients: {len(diagnosis_data)}")
        
        # Show summary
        for patient_id, data in list(diagnosis_data.items())[:3]:  # Show first 3
            print(f"\n{patient_id}:")
            print(f"  Expected aneurysms: {data['expected_aneurysm_count']}")
            print(f"  Locations: {data['locations']}")
            print(f"  Sizes: {data['sizes']}")
        
        return diagnosis_data
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return None

def create_specific_diagnosis(patient_file="01_MRA1_seg.nii.gz"):
    """
    Create specific diagnosis for our test file based on the converted data.
    """
    # Check if we have converted diagnosis data
    diagnosis_path = "diagnosis/uan_diagnosis.json"
    if Path(diagnosis_path).exists():
        with open(diagnosis_path, 'r') as f:
            diagnosis_data = json.load(f)
        
        # Try to find matching patient data
        patient_name = patient_file.replace('.nii.gz', '').replace('.nii', '')
        
        for patient_id, data in diagnosis_data.items():
            if patient_name.lower() in patient_id.lower() or patient_id.lower() in patient_name.lower():
                # Create specific diagnosis file
                specific_diagnosis = {
                    'patient_file': patient_file,
                    'expected_aneurysm_count': data.get('expected_aneurysm_count', 2),
                    'aneurysm_labels': None,  # Let the algorithm find them
                    'expected_locations': [],  # We'll let the algorithm find them
                    'location_tolerance': 50,
                    'notes': f"Diagnosis for {patient_file} based on clinical data"
                }
                
                output_path = f"diagnosis_{patient_name}.json"
                with open(output_path, 'w') as f:
                    json.dump(specific_diagnosis, f, indent=2)
                
                print(f"Created specific diagnosis: {output_path}")
                return output_path
    
    # Default diagnosis if no specific data found
    default_diagnosis = {
        'patient_file': patient_file,
        'expected_aneurysm_count': 2,  # Based on user feedback
        'aneurysm_labels': None,
        'expected_locations': [],
        'location_tolerance': 50,
        'notes': "Default diagnosis - expecting 2 aneurysms based on clinical assessment"
    }
    
    output_path = f"diagnosis_{patient_file.replace('.nii.gz', '').replace('.nii', '')}.json"
    with open(output_path, 'w') as f:
        json.dump(default_diagnosis, f, indent=2)
    
    print(f"Created default diagnosis: {output_path}")
    return output_path

if __name__ == "__main__":
    # Convert the Excel file
    excel_path = "diagnosis/SNHU_TAnDB_DICOM.xlsx"
    diagnosis_data = convert_diagnosis_excel_to_json(excel_path)
    
    # Create specific diagnosis for our test file
    specific_file = create_specific_diagnosis("01_MRA1_seg.nii.gz")
    print(f"\nUse this diagnosis file with: python processing/NIfTI_find_aneurysm.py ' example_data/01_MRA1_seg.nii.gz' -d {specific_file}") 