# Aneurysm Growth Prediction using PyAnsys and GAN

This project combines finite element analysis (FEA) using PyAnsys with Generative Adversarial Networks (GANs) to predict cerebral aneurysm growth patterns over time.

## Overview

The project workflow consists of:
1. Processing time-series 3D aneurysm data
2. Performing FEA on each time point to analyze stress evolution
3. Training a GAN to learn growth patterns
4. Predicting future aneurysm morphology and stress distributions

## Data Requirements

### Input Data Structure
```
data/time_series/
├── patient_001/
│   ├── baseline/
│   │   ├── aneurysm.stl
│   │   └── metadata.json
│   ├── 6_months/
│   │   ├── aneurysm.stl
│   │   └── metadata.json
│   ├── 12_months/
│   │   ├── aneurysm.stl
│   │   └── metadata.json
│   └── patient_info.json
└── patient_002/
    └── ...
```

### Metadata Format
```json
{
    "patient_id": "001",
    "scan_date": "2023-01-15",
    "time_point": "baseline",
    "aneurysm_volume": 125.3,
    "max_diameter": 7.2,
    "clinical_notes": "No symptoms reported"
}
```

## Architecture

### FEA Pipeline
- Mesh generation and refinement
- Boundary condition application
- Stress analysis computation
- Feature extraction over time

### GAN Architecture
- **Generator**: Predicts future aneurysm shape and stress distribution
- **Discriminator**: Distinguishes real vs. generated aneurysm evolution
- **Temporal Component**: Captures time-dependent growth patterns

## Usage

### 1. Preprocess Time Series Data
```python
python src/preprocessing/prepare_time_series.py --input_dir data/time_series --output_dir data/processed
```

### 2. Run FEA Analysis on All Time Points
```python
python src/fea_analysis/batch_stress_analysis.py --input_dir data/processed --output_dir results/fea_time_series
```

### 3. Train GAN Model
```python
python src/gan_models/train_growth_gan.py --data_dir results/fea_time_series --epochs 1000 --batch_size 16
```

### 4. Predict Future Growth
```python
python src/prediction/predict_growth.py --model_path models/growth_gan.pth --patient_data data/new_patient --output results/predictions
```

## Key Features

### Biomechanical Features
- Temporal WSS patterns
- Stress concentration evolution
- Volume growth rate
- Shape index changes
- Aneurysm neck evolution

### GAN Features
- 3D shape generation
- Stress field prediction
- Uncertainty quantification
- Multi-timepoint forecasting

## Evaluation Metrics

- **Shape Accuracy**: Dice coefficient, Hausdorff distance
- **Stress Prediction**: Mean absolute error, correlation coefficient
- **Growth Rate**: Predicted vs. actual volume change
- **Clinical Relevance**: Risk stratification accuracy 