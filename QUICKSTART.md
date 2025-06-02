# Quick Start Guide

## Setting Up Your Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the project in development mode:**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

3. **Verify ANSYS installation:**
   ```python
   from ansys.mapdl.core import launch_mapdl
   mapdl = launch_mapdl()
   print(mapdl)
   ```

## Project 1: Rupture Risk Prediction

### Step 1: Prepare Your Data
Place your aneurysm STL files in the appropriate directories:
```
rupture_risk_prediction/data/
├── ruptured/
│   ├── patient_001.stl
│   └── patient_001.json  # metadata
└── unruptured/
    ├── patient_101.stl
    └── patient_101.json  # metadata
```

### Step 2: Preprocess Meshes
```bash
cd rupture_risk_prediction
python src/preprocessing/prepare_meshes.py \
    --input_dir data/ruptured \
    --output_dir data/processed \
    --edge_length 0.5
```

### Step 3: Run FEA Analysis
```python
from rupture_risk_prediction.src.fea_analysis.stress_analysis import AneurysmStressAnalysis

analyzer = AneurysmStressAnalysis(working_dir="./ansys_work")
results, features = analyzer.analyze_aneurysm("data/processed/patient_001.stl")
```

### Step 4: Visualize Results
```python
from shared.visualization.visualize_results import AneurysmVisualizer
import trimesh

visualizer = AneurysmVisualizer()
mesh = trimesh.load("data/processed/patient_001.stl")
visualizer.visualize_mesh_with_stress(mesh, results['von_mises'])
```

## Project 2: Growth Prediction

### Step 1: Prepare Time-Series Data
```
growth_prediction/data/time_series/
├── patient_001/
│   ├── baseline/
│   │   └── aneurysm.stl
│   ├── 6_months/
│   │   └── aneurysm.stl
│   └── 12_months/
        └── aneurysm.stl
```

### Step 2: Train GAN Model
```python
from growth_prediction.src.gan_models.growth_gan import train_growth_gan
from torch.utils.data import DataLoader

# Create your dataset and dataloader
# dataset = AneurysmGrowthDataset("data/processed")
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train the model
# train_growth_gan(dataloader, num_epochs=1000)
```

## Example Notebooks

Create Jupyter notebooks in the respective `notebooks/` directories:

```python
# Example notebook cell for rupture risk analysis
import pandas as pd
import numpy as np
from rupture_risk_prediction.src.fea_analysis.stress_analysis import AneurysmStressAnalysis
from shared.visualization.visualize_results import AneurysmVisualizer

# Load and analyze
analyzer = AneurysmStressAnalysis()
results, features = analyzer.analyze_aneurysm("path/to/aneurysm.stl")

# Visualize
viz = AneurysmVisualizer()
viz.create_feature_importance_plot(
    feature_names=list(features.keys()),
    importances=np.random.rand(len(features))  # Replace with actual importances
)
```

## Common Issues and Solutions

### Issue: ANSYS License Error
**Solution:** Ensure ANSYS is properly installed and licensed. Check environment variables:
```bash
echo $ANSYS_ROOT  # Should point to ANSYS installation
```

### Issue: Memory Error with Large Meshes
**Solution:** Reduce mesh density or use batch processing:
```python
preprocessor = AneurysmMeshPreprocessor(target_edge_length=1.0)  # Larger edge length
```

### Issue: CUDA Out of Memory (GAN Training)
**Solution:** Reduce batch size or model complexity:
```python
gan = AneurysmGrowthGAN(device='cpu')  # Use CPU if GPU memory is limited
```

## Next Steps

1. **Collect More Data:** The models will perform better with more training data
2. **Fine-tune Hyperparameters:** Adjust learning rates, model architectures
3. **Validate Results:** Compare FEA results with clinical outcomes
4. **Deploy Models:** Create a web interface or clinical tool

## Resources

- [PyAnsys Documentation](https://docs.pyansys.com/)
- [PyVista Examples](https://docs.pyvista.org/examples/index.html)
- [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Contact

For questions or issues, please create an issue on the GitHub repository. 