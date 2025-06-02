# Cerebrovascular Aneurysm Analysis using PyAnsys

This repository contains two projects for cerebrovascular aneurysm analysis using static stress analysis with PyAnsys:

1. **Aneurysm Rupture Risk Prediction** - Predicts rupture risk using 3D aneurysm models and FEA
2. **Aneurysm Growth Prediction** - Predicts aneurysm growth using PyAnsys and GAN with time-interval tracked data

## Project Structure

```
├── rupture_risk_prediction/       # Rupture risk prediction project
│   ├── data/                      # Input data directory
│   │   ├── ruptured/             # 3D files of ruptured aneurysms
│   │   └── unruptured/           # 3D files of unruptured aneurysms
│   ├── src/                       # Source code
│   │   ├── preprocessing/        # Data preprocessing scripts
│   │   ├── fea_analysis/         # Finite Element Analysis scripts
│   │   ├── feature_extraction/   # Feature extraction from FEA results
│   │   └── models/               # ML models for risk prediction
│   ├── results/                   # Analysis results
│   └── notebooks/                 # Jupyter notebooks for analysis
│
├── growth_prediction/             # Aneurysm growth prediction project
│   ├── data/                      # Input data directory
│   │   ├── time_series/          # Time-interval tracked aneurysm data
│   │   └── processed/            # Processed data for training
│   ├── src/                       # Source code
│   │   ├── preprocessing/        # Data preprocessing scripts
│   │   ├── fea_analysis/         # Finite Element Analysis scripts
│   │   ├── gan_models/           # GAN architecture and training
│   │   └── prediction/           # Growth prediction models
│   ├── results/                   # Analysis results
│   └── notebooks/                 # Jupyter notebooks for analysis
│
├── shared/                        # Shared utilities and resources
│   ├── utils/                     # Utility functions
│   ├── mesh_processing/          # Mesh processing utilities
│   └── visualization/            # Visualization tools
│
└── docs/                          # Documentation

```

## Prerequisites

- Python 3.8+
- ANSYS 2023 R1 or later
- PyAnsys packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd urp-2025
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

### Rupture Risk Prediction
- 3D aneurysm models in STL/VTK format
- Metadata including patient demographics, aneurysm location, morphological features

### Growth Prediction
- Time-series 3D models of the same aneurysm at different time points
- Temporal metadata including scan dates and clinical observations

## Usage

See individual project README files for detailed usage instructions:
- [Rupture Risk Prediction](./rupture_risk_prediction/README.md)
- [Growth Prediction](./growth_prediction/README.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Your Name - Initial work

## Acknowledgments

- KAIST URP Program
- Medical imaging data providers
- PyAnsys development team 