# Aneurysm Cropping Project

## Overview
This project processes MRA (Magnetic Resonance Angiography) scans with COSTA segmentation results to extract local vascular regions around aneurysms for machine learning analysis.

## Features
- **Automated aneurysm detection** from segmentation masks
- **Local region cropping** (64×64×32 voxels around aneurysm centers)
- **Clinical data integration** from Excel patient database
- **Comprehensive metadata** for each cropped aneurysm
- **Visualization tools** for quality assessment

## Data Structure

### Input Data
- **Patient Information**: `SNHU_TAnDB_DICOM.xlsx` (111 patients with clinical data)
- **Segmentation Results**: `~/Downloads/segmentation/aneu/UAN_processed/Input/` (168 MRA scans)
  - Raw MRA images: `{PatientID}_MRA{1/2}/Raw/{PatientID}_MRA{1/2}.nii.gz`
  - Aneurysm masks: `{PatientID}_MRA{1/2}/Output/{PatientID}_MRA{1/2}.nii.gz`

### Output Structure
```
output/
├── aneurysm_cropping_summary.csv          # Complete summary of all aneurysms
├── patient_10/
│   ├── 10_MRA1/
│   │   ├── aneurysm_1_raw.nii.gz          # Cropped raw image (64×64×32)
│   │   ├── aneurysm_1_mask.nii.gz         # Cropped segmentation mask
│   │   ├── aneurysm_1_metadata.json       # Clinical + technical metadata
│   │   ├── aneurysm_2_raw.nii.gz
│   │   └── ...
│   └── 10_MRA2/
└── patient_11/
    └── ...
```

## Key Scripts

### 1. `aneurysm_cropper.py`
Main processing script that:
- Loads patient clinical data from Excel
- Finds aneurysm centers from segmentation masks
- Crops 64×64×32 voxel regions around each aneurysm
- Saves raw images, masks, and metadata

### 2. `visualize_aneurysms.py` 
Visualization and analysis tools:
- Volume distribution analysis
- Individual aneurysm visualization
- Patient-wise statistics
- Largest/smallest aneurysm exploration

### 3. `explore_patient_data.py`
Data exploration utility for understanding the dataset structure.

## Usage

### Basic Processing
```python
# Process first 5 patients (for testing)
from aneurysm_cropper import AneurysmCropper
cropper = AneurysmCropper()
cropper.process_all_patients(max_patients=5, crop_size=(64, 64, 32))

# Process all patients
cropper.process_all_patients(crop_size=(64, 64, 32))
```

### Visualization
```python
from visualize_aneurysms import AneurysmVisualizer
viz = AneurysmVisualizer()

# Generate overview plots
viz.plot_volume_distribution()
viz.analyze_by_patient()

# View specific aneurysm
viz.visualize_aneurysm(patient_id=10, folder_name="10_MRA1", aneurysm_index=1)

# Show largest aneurysms
viz.show_largest_aneurysms(n=5)
```

## Current Results (First 5 Patients)

### Summary Statistics
- **Patients processed**: 3 unique patients (5 MRA scans)
- **Total aneurysms found**: 268
- **Volume range**: 10 - 767,746 voxels
- **Average volume**: 8,034 voxels

### Patient Breakdown
- **Patient 10**: 89 aneurysms (46 from MRA1, 43 from MRA2)
- **Patient 11**: 129 aneurysms (51 from MRA1, 78 from MRA2)  
- **Patient 12**: 50 aneurysms (50 from MRA1)

## Metadata Schema

Each aneurysm includes comprehensive metadata:

```json
{
  "patient_id": 10,
  "folder_name": "10_MRA1",
  "aneurysm_index": 1,
  "aneurysm_info": {
    "center": [70, 325, 150],
    "volume": 328,
    "bbox": [65, 75, 320, 330, 145, 155]
  },
  "crop_info": {
    "original_center": [70, 325, 150],
    "crop_bbox": [38, 102, 293, 357, 134, 166],
    "crop_size": [64, 64, 32]
  },
  "anatomical_locations": ["MCA", "ICA (total)"],
  "clinical_info": {
    "연령": 45,
    "성별": 1,
    "Smoking_record_combined (non:0, current:1, ex:2)": 0,
    "HT": 1,
    "DM": 0,
    "Max or ruptureA size": 3.2
  }
}
```

## Next Steps

### 1. Process All Patients
```bash
python aneurysm_cropper.py
# Then modify to remove max_patients limit
```

### 2. Quality Assessment
- Review volume distribution for outliers
- Inspect largest/smallest aneurysms
- Check anatomical location distribution

### 3. Machine Learning Preparation
- **Classification**: Predict aneurysm rupture risk from images + clinical data
- **Segmentation**: Train models on cropped aneurysm regions
- **Feature extraction**: Extract morphological features

### 4. Data Augmentation
Consider implementing:
- Rotation/translation augmentation
- Intensity normalization
- Multi-scale cropping (32×32×16, 128×128×64)

## Clinical Variables Available

- **Demographics**: Age, gender
- **Aneurysm characteristics**: Location, size, multiplicity
- **Risk factors**: Smoking, hypertension, diabetes, cholesterol
- **Anatomical locations**: MCA, ACA, Acom, ICA, Pcom, BA, PCA

## File Formats
- **Images**: NIfTI (.nii.gz) format
- **Metadata**: JSON format
- **Summary**: CSV format for easy analysis

## Dependencies
```
pandas
numpy
nibabel
matplotlib
seaborn
scikit-image
scipy
openpyxl
```

---

**Contact**: For questions about the aneurysm analysis pipeline or to extend the cropping functionality. 