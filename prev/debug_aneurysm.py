import sys
sys.path.append('processing')

from NIfTI_find_aneurysm import AneurysmDetector
import json

# Load diagnosis
with open('diagnosis_01_MRA1_seg.json', 'r') as f:
    diagnosis = json.load(f)

print("Diagnosis loaded:", diagnosis)

try:
    detector = AneurysmDetector(" example_data/01_MRA1_seg.nii.gz", diagnosis)
    print("Detector created successfully")
    
    # Test individual steps
    labels = detector.get_aneurysm_labels()
    print("Aneurysm labels found:", labels)
    print("Type of labels:", type(labels))
    
    if labels:
        for label in labels:
            print(f"Processing label: {label} (type: {type(label)})")
            
    results = detector.find_aneurysms()
    print("Results:", results)
    
except Exception as e:
    import traceback
    print("Error occurred:")
    print(traceback.format_exc()) 