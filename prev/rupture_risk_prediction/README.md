# Aneurysm Rupture Risk Prediction

This project uses PyAnsys for finite element analysis (FEA) to predict the rupture risk of cerebral aneurysms based on structural stress analysis.

## Overview

The project workflow consists of:
1. Preprocessing 3D aneurysm models
2. Performing FEA using PyAnsys to calculate stress distributions
3. Extracting biomechanical features (Wall Shear Stress, Von Mises stress, etc.)
4. Training machine learning models to predict rupture risk

## Data Requirements

### Input Data Structure
```
data/
├── ruptured/
│   ├── patient_001.stl
│   ├── patient_001_metadata.json
│   └── ...
└── unruptured/
    ├── patient_101.stl
    ├── patient_101_metadata.json
    └── ...
```

### Metadata Format
```json
{
    "patient_id": "001",
    "age": 55,
    "gender": "F",
    "aneurysm_location": "MCA",
    "aneurysm_size": 7.2,
    "aspect_ratio": 1.8,
    "rupture_status": "ruptured",
    "scan_date": "2023-01-15"
}
```

## Usage

### 1. Data Preprocessing
```python
python src/preprocessing/prepare_meshes.py --input_dir data/ruptured --output_dir data/processed
```

### 2. Run FEA Analysis
```python
python src/fea_analysis/run_stress_analysis.py --mesh_dir data/processed --output_dir results/fea
```

### 3. Extract Features
```python
python src/feature_extraction/extract_features.py --fea_results results/fea --output results/features.csv
```

### 4. Train Prediction Model
```python
python src/models/train_classifier.py --features results/features.csv --output models/rupture_risk_model.pkl
```

## Key Features Analyzed

- **Wall Shear Stress (WSS)**: Frictional force of blood flow on vessel wall
- **Von Mises Stress**: Overall stress intensity in the aneurysm wall
- **Oscillatory Shear Index (OSI)**: Directional change of WSS during cardiac cycle
- **Pressure Distribution**: Blood pressure patterns within the aneurysm
- **Wall Thickness Variation**: Structural integrity indicators

## Model Performance Metrics

- Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- AUC-ROC
- F1 Score 