# PyFluent Aneurysm CFD Analysis - 78_MRA1_seg

## Overview

This directory contains specialized **PyFluent CFD analysis** tools for analyzing the **78_MRA1_seg patient's aneurysm** using **Ansys Fluent 2025 R1** through the PyFluent Python interface.

## 📁 Directory Structure

```
analyzing_using_pyfluent/
├── boundary_conditions/
│   └── 78_MRA1_seg_pyfluent_bc.json     # Boundary conditions
├── meshes/
│   └── 78_MRA1_seg_aneurysm.stl         # STL mesh file (to be added)
├── results/
│   └── 78_MRA1_seg_pyfluent/            # Output directory
├── scripts/
│   ├── analyze_78_MRA1_seg.py           # Main analysis script
│   ├── pyfluent_real_cfd.py             # General PyFluent CFD script
│   └── PYFLUENT_IMPLEMENTATION_GUIDE.md # Implementation guide
└── README.md                            # This file
```

## 🔧 Prerequisites

### **Software Requirements**
- **Ansys Fluent 2025 R1** installed at `/opt/cvbml/softwares/ansys_inc/v251`
- **Python 3.10+** with conda/virtual environment
- **PyFluent 0.32.1+** (already installed)

### **Hardware Requirements**
- **32+ CPU cores** (recommended for parallel processing)
- **16+ GB RAM** (32+ GB recommended for complex meshes)
- **SSD storage** for I/O performance

### **License Requirements**
- Valid **Ansys license** with Fluent solver access
- License server properly configured

## 🚀 Quick Start

### **1. Prepare the Mesh File**
Place the STL mesh file for the 78_MRA1_seg aneurysm at:
```bash
meshes/78_MRA1_seg_aneurysm.stl
```

### **2. Set Environment Variables**
```bash
export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251
```

### **3. Run Analysis**
```bash
cd scripts/
python analyze_78_MRA1_seg.py
```

## 📋 Boundary Conditions Summary

The analysis uses **comprehensive boundary conditions** specifically designed for aneurysm CFD:

### **Flow Conditions**
- **Inlet velocity**: 0.127 m/s (mean), 0.254 m/s (peak)
- **Flow direction**: [-0.986, -0.065, 0.152]
- **Reynolds number**: 238 (laminar flow)
- **Womersley number**: 3.8 (pulsatile characteristics)

### **Blood Properties**
- **Density**: 1060 kg/m³
- **Dynamic viscosity**: 0.0035 Pa·s
- **Temperature**: 310.15 K (37°C)

### **Simulation Settings**
- **Time step**: 0.001 s
- **Cardiac cycles**: 3
- **Total time steps**: 3000
- **Solver**: Coupled, unsteady

### **Aneurysm Parameters**
- **Size ratio**: 1.6
- **Aspect ratio**: 1.6  
- **Critical WSS threshold**: 0.4 Pa
- **High WSS threshold**: 1.5 Pa

## 🔬 Analysis Features

### **Physics Models**
- ✅ **Laminar flow** (Re = 238)
- ✅ **Energy equation** enabled
- ✅ **Pulsatile flow** with cardiac cycle
- ✅ **Blood material properties**

### **Boundary Conditions**
- ✅ **Velocity inlet** with physiological waveform
- ✅ **Pressure outlet** (0 Pa gauge)
- ✅ **No-slip walls** at body temperature

### **Solver Configuration**
- ✅ **Coupled pressure-velocity**
- ✅ **Second-order time integration**
- ✅ **Bounded central differencing**
- ✅ **Presto pressure discretization**

### **Output Variables**
- 🩸 **Pressure distribution**
- 🌊 **Wall shear stress (WSS)**
- 🏃 **Velocity magnitude**
- 🌀 **Strain rate magnitude**
- 🔄 **Q-criterion** (vortex identification)

## 📊 Expected Results

### **Output Files**
```
results/78_MRA1_seg_pyfluent/
├── 78_MRA1_seg_analysis.cas          # Fluent case file
├── 78_MRA1_seg_analysis.dat          # Fluent data file
├── 78_MRA1_seg_results.case          # EnSight case file
├── 78_MRA1_seg_results.geo           # EnSight geometry
├── 78_MRA1_seg_wall_hemodynamics.csv # Wall data (CSV)
└── 78_MRA1_seg_analysis_summary.json # Analysis summary
```

### **Hemodynamic Metrics**
- **WSS range**: 0.1 - 2.0 Pa (physiological)
- **Pressure drop**: 50 - 200 Pa
- **Peak velocities**: ~0.25 m/s
- **Aneurysm flow patterns**: Recirculation zones

## 🎯 Key Analysis Objectives

### **Rupture Risk Assessment**
1. **Wall Shear Stress Analysis**
   - Low WSS areas (< 0.4 Pa) → Growth risk
   - High WSS areas (> 1.5 Pa) → Rupture risk
   
2. **Flow Pattern Analysis**
   - Stagnation zones in aneurysm dome
   - Vortex formation and breakdown
   - Oscillatory shear index

3. **Pressure Analysis**
   - Pressure loading on aneurysm wall
   - Pressure gradient effects

### **Clinical Significance**
- **Aspect ratio**: 1.6 (moderate rupture risk)
- **Size ratio**: 1.6 (geometric risk factor)
- **Flow impingement**: Critical for wall stress

## 💻 Running the Analysis

### **Basic Execution**
```bash
# Navigate to scripts directory
cd analyzing_using_pyfluent/scripts/

# Run the analysis (requires mesh file)
python analyze_78_MRA1_seg.py
```

### **Custom Parameters**
You can modify the analysis by editing:
- `boundary_conditions/78_MRA1_seg_pyfluent_bc.json`
- Core count in `analyze_78_MRA1_seg.py` (line ~555)

### **Monitoring Progress**
The script provides detailed progress updates:
- ✅ Session launch status
- 📐 Mesh import statistics  
- ⚙️ Physics model configuration
- 🔄 Simulation progress
- 📊 Results extraction

## 🔍 Post-Processing

### **Visualization in ParaView**
1. Open `78_MRA1_seg_results.case` in ParaView
2. Load variables: pressure, wall-shear-stress, velocity-magnitude
3. Create contour plots and streamlines

### **Data Analysis**
```python
import pandas as pd

# Load wall data
wall_data = pd.read_csv('results/78_MRA1_seg_pyfluent/78_MRA1_seg_wall_hemodynamics.csv')

# Analyze WSS distribution
wss_mean = wall_data['wall-shear-stress'].mean()
wss_max = wall_data['wall-shear-stress'].max()
low_wss_area = (wall_data['wall-shear-stress'] < 0.4).sum()

print(f"Mean WSS: {wss_mean:.3f} Pa")
print(f"Max WSS: {wss_max:.3f} Pa") 
print(f"Low WSS points: {low_wss_area}")
```

## ⚠️ Important Notes

### **Mesh Requirements**
- **STL format** with watertight geometry
- **High quality** mesh for aneurysm region
- **Boundary layer** refinement at walls

### **Computational Resources**
- **32 cores**: ~30-60 minutes simulation time
- **64 GB RAM**: Recommended for large meshes
- **Fast storage**: SSD recommended for I/O

### **Convergence Monitoring**
- Monitor residuals < 1e-4
- Check mass conservation
- Verify solution stability

## 🆘 Troubleshooting

### **Common Issues**

1. **Mesh Import Failure**
   ```
   ❌ Mesh import failed
   → Check STL file format and location
   → Ensure mesh is watertight
   ```

2. **License Error**
   ```
   ❌ License error
   → Check ANSYSLMD_LICENSE_FILE
   → Verify license server connectivity
   ```

3. **Convergence Problems**
   ```
   ❌ Convergence failure
   → Reduce time step size
   → Increase under-relaxation factors
   ```

### **Performance Optimization**
- Use **double precision** for accuracy
- **32 cores** optimal for this mesh size
- **SSD storage** for faster I/O

## 📞 Support

For issues with:
- **PyFluent**: Check [PyFluent Documentation](https://fluent.docs.pyansys.com/)
- **Ansys Fluent**: Consult Ansys documentation
- **Analysis setup**: Review boundary conditions JSON file

## 🎯 Next Steps

1. **Obtain STL mesh** for 78_MRA1_seg aneurysm
2. **Run CFD analysis** with provided scripts
3. **Post-process results** in ParaView
4. **Extract hemodynamic metrics** for clinical assessment
5. **Compare with clinical data** for validation

---

**Ready for professional aneurysm CFD analysis with PyFluent!** 🚀 