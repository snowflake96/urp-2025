# PyFluent CFD Analysis Implementation Guide

## Overview

This guide demonstrates how to implement **real Ansys Fluent CFD analysis** using **PyFluent** for aneurysm hemodynamic analysis. PyFluent provides a Python interface to the industry-standard Ansys Fluent CFD solver.

## 🎯 What PyFluent Provides

### ✅ **Industry-Standard CFD**
- **Ansys Fluent solver**: Validated, clinical-grade CFD algorithms
- **Robust turbulence models**: k-ω SST, Reynolds Stress Models
- **Parallel processing**: True 32-core utilization
- **Professional output**: EnSight, Tecplot, VTK formats

### ✅ **Python Integration**
- **Pythonic API**: Clean, object-oriented interface
- **Automation**: Scriptable workflows for batch processing
- **Integration**: Works with NumPy, Pandas, PyVista
- **Visualization**: Built-in postprocessing capabilities

## 🔧 System Requirements

### **Required Software**
```bash
# Licensed Ansys Fluent installation
# Version 2022 R2 or later recommended
export AWP_ROOT232=/usr/ansys_inc/v232  # Linux
# Windows: Set automatically by installer

# PyFluent library
pip install ansys-fluent-core
```

### **Hardware Requirements**
- **CPU**: 32+ cores recommended for parallel processing
- **RAM**: 16+ GB (32+ GB for large meshes)
- **Storage**: SSD recommended for I/O performance
- **License**: Valid Ansys license server access

## 📋 Complete PyFluent Workflow

### **1. Session Management**
```python
import ansys.fluent.core as pyfluent

# Launch Fluent with 32 cores
session = pyfluent.launch_fluent(
    version="3d",
    precision="double",
    processor_count=32,
    mode="solver",
    show_gui=False,
    cleanup_on_exit=True
)
```

### **2. Mesh Import and Validation**
```python
# Import STL mesh
session.file.import_.cff_files(file_names=["vessel.stl"])

# Check mesh quality
session.mesh.check()

# Get mesh statistics
mesh_info = session.mesh.get_info()
print(f"Nodes: {mesh_info['nodes']}")
print(f"Cells: {mesh_info['cells']}")
```

### **3. Physics Models Setup**
```python
# Enable energy equation
session.setup.models.energy.enabled = True

# Viscous model selection
reynolds = 238  # From boundary conditions
if reynolds < 2000:
    session.setup.models.viscous.model = "laminar"
else:
    session.setup.models.viscous.model = "k-omega-sst"

# Unsteady solver for pulsatile flow
session.solution.methods.unsteady_formulation.enabled = True
session.solution.methods.unsteady_formulation.time_formulation = "second-order-implicit"
```

### **4. Material Properties**
```python
# Blood properties
session.setup.materials.fluid["blood"] = {
    "density": {"option": "constant", "value": 1060},  # kg/m³
    "viscosity": {"option": "constant", "value": 0.0035},  # Pa·s
    "specific_heat": {"option": "constant", "value": 3617},  # J/kg·K
    "thermal_conductivity": {"option": "constant", "value": 0.52}  # W/m·K
}

# Assign blood to fluid zone
fluid_zones = session.setup.boundary_conditions.get_zones(zone_type="fluid")
session.setup.boundary_conditions.fluid[fluid_zones[0]].material = "blood"
```

### **5. Boundary Conditions**
```python
# Inlet: Velocity inlet
inlet_zones = session.setup.boundary_conditions.get_zones(zone_type="velocity-inlet")
inlet = session.setup.boundary_conditions.velocity_inlet[inlet_zones[0]]

inlet.momentum.velocity.value = 0.127  # m/s
inlet.momentum.velocity.direction = [-0.986, -0.065, 0.152]
inlet.turbulence.turbulent_intensity = 0.05
inlet.thermal.temperature.value = 310.15  # K

# Outlet: Pressure outlet
outlet_zones = session.setup.boundary_conditions.get_zones(zone_type="pressure-outlet")
outlet = session.setup.boundary_conditions.pressure_outlet[outlet_zones[0]]
outlet.momentum.gauge_pressure.value = 0  # Pa

# Walls: No-slip
wall_zones = session.setup.boundary_conditions.get_zones(zone_type="wall")
for wall_zone in wall_zones:
    wall = session.setup.boundary_conditions.wall[wall_zone]
    wall.momentum.shear_condition = "no-slip"
    wall.thermal.thermal_condition = "temperature"
    wall.thermal.temperature.value = 310.15  # K
```

### **6. Solver Configuration**
```python
# Pressure-velocity coupling
session.solution.methods.p_v_coupling.setup.coupled_algorithm = "coupled"

# Spatial discretization
session.solution.methods.spatial_discretization.pressure = "presto"
session.solution.methods.spatial_discretization.momentum = "bounded-central-differencing"

# Time stepping
session.solution.run_calculation.time_step_size = 0.001  # s
session.solution.run_calculation.max_iterations_per_time_step = 20

# Convergence criteria
session.solution.monitor.residual.equations.continuity.absolute_criteria = 1e-4
session.solution.monitor.residual.equations.x_momentum.absolute_criteria = 1e-4
```

### **7. Solution Execution**
```python
# Initialize solution
session.solution.initialization.hybrid_initialize()

# Run transient calculation (3 cardiac cycles)
time_steps = 3000  # 3 seconds at 0.001s time step
session.solution.run_calculation.dual_time_iterate(
    number_of_time_steps=time_steps,
    max_iterations_per_time_step=20
)
```

### **8. Results Export**
```python
# Export case and data files
session.file.write_case_data(file_name="patient_results")

# Export EnSight format for visualization
session.file.export.ensight_gold(
    file_name="patient_ensight",
    surfaces_list=["wall"],
    variables_list=["pressure", "wall-shear-stress", "velocity-magnitude"]
)

# Export wall data as CSV
session.surface.export_to_csv(
    file_name="wall_data.csv",
    surfaces_list=["wall"],
    variables_list=["x-coordinate", "y-coordinate", "z-coordinate", 
                   "pressure", "wall-shear-stress", "velocity-magnitude"]
)
```

### **9. Session Cleanup**
```python
# Close Fluent session
session.exit()
```

## 📊 Expected Results

### **Hemodynamic Parameters**
- **Wall Shear Stress**: 0.1 - 1.2 Pa (physiological range)
- **Pressure**: 0 - 120 Pa (relative to outlet)
- **Velocity**: 0.05 - 0.20 m/s (typical cerebral arteries)
- **Reynolds Number**: 150 - 400 (laminar flow regime)

### **Output Files**
```
pyfluent_results/
├── patient_id/
│   ├── patient_id.cas          # Fluent case file
│   ├── patient_id.dat          # Fluent data file
│   ├── patient_id.case         # EnSight case file
│   ├── patient_id.geo          # EnSight geometry
│   ├── patient_id.vel          # EnSight velocity
│   ├── patient_id.pres         # EnSight pressure
│   ├── patient_id.wss          # EnSight WSS
│   └── wall_data.csv           # CSV data export
```

## 🚀 Batch Processing Script

```python
#!/usr/bin/env python3
"""
PyFluent Batch Analysis for Multiple Patients
"""

import ansys.fluent.core as pyfluent
import json
import os
from pathlib import Path

def analyze_patient_with_pyfluent(stl_file, bc_file, output_dir, n_cores=32):
    """Complete PyFluent analysis for one patient"""
    
    # Load boundary conditions
    with open(bc_file, 'r') as f:
        bc_data = json.load(f)
    
    # Launch Fluent
    session = pyfluent.launch_fluent(
        version="3d",
        precision="double",
        processor_count=n_cores,
        mode="solver",
        show_gui=False
    )
    
    try:
        # Import mesh
        session.file.import_.cff_files(file_names=[stl_file])
        session.mesh.check()
        
        # Setup physics models
        reynolds = bc_data['inlet_conditions']['reynolds_number']
        if reynolds < 2000:
            session.setup.models.viscous.model = "laminar"
        else:
            session.setup.models.viscous.model = "k-omega-sst"
        
        # Setup materials and boundary conditions
        # ... (as shown above)
        
        # Run simulation
        session.solution.initialization.hybrid_initialize()
        session.solution.run_calculation.dual_time_iterate(
            number_of_time_steps=3000,
            max_iterations_per_time_step=20
        )
        
        # Export results
        patient_id = Path(stl_file).stem.replace('_clean_flat', '')
        session.file.write_case_data(file_name=f"{output_dir}/{patient_id}")
        session.file.export.ensight_gold(
            file_name=f"{output_dir}/{patient_id}_results",
            surfaces_list=["wall"],
            variables_list=["pressure", "wall-shear-stress", "velocity-magnitude"]
        )
        
        return True
        
    finally:
        session.exit()

# Process multiple patients
patients = ["08_MRA1_seg", "09_MRA1_seg", "23_MRA2_seg", 
           "38_MRA1_seg", "44_MRA1_seg", "78_MRA1_seg"]

for patient_id in patients:
    stl_file = f"vessels/{patient_id}_clean_flat.stl"
    bc_file = f"boundary_conditions/{patient_id}_pulsatile_bc.json"
    
    print(f"Processing {patient_id}...")
    success = analyze_patient_with_pyfluent(stl_file, bc_file, "results")
    print(f"{'✅' if success else '❌'} {patient_id} completed")
```

## 🔍 Advantages Over Custom Implementation

### **PyFluent Benefits**
1. **Validated Solver**: Industry-standard Ansys Fluent algorithms
2. **Robust Numerics**: Proven convergence and stability
3. **Professional Output**: Compatible with all major visualization tools
4. **Support**: Official Ansys support and documentation
5. **Scalability**: Handles complex geometries and large meshes
6. **Accuracy**: Clinical-grade CFD results

### **Comparison with Custom CFD**
| Feature | PyFluent | Custom Implementation |
|---------|----------|----------------------|
| **Solver Validation** | ✅ Industry standard | ❌ Requires validation |
| **Turbulence Models** | ✅ Full library | ⚠️ Limited options |
| **Parallel Processing** | ✅ Optimized | ⚠️ Manual implementation |
| **Mesh Handling** | ✅ Robust | ⚠️ Error-prone |
| **Convergence** | ✅ Proven algorithms | ❌ Stability issues |
| **Support** | ✅ Professional | ❌ Community only |

## 📚 Additional Resources

### **Documentation**
- [PyFluent Documentation](https://fluent.docs.pyansys.com/)
- [Ansys Fluent User Guide](https://ansyshelp.ansys.com/Views/Secured/prod_page.html?pn=Fluent)
- [PyFluent Examples](https://fluent.docs.pyansys.com/version/stable/examples/index.html)

### **Installation**
```bash
# Install PyFluent
pip install ansys-fluent-core

# Additional visualization tools
pip install ansys-fluent-visualization
pip install pyvista
```

### **License Requirements**
- Valid Ansys license with Fluent solver access
- License server configuration
- Sufficient license tokens for parallel processing

## 🎯 Conclusion

**PyFluent provides the most robust and clinically-validated approach** for aneurysm CFD analysis. While it requires a licensed Ansys installation, the benefits include:

- **Industry-standard accuracy**
- **Professional visualization output**
- **Proven parallel performance**
- **Clinical validation**
- **Long-term support**

For production aneurysm analysis, **PyFluent is the recommended approach** over custom CFD implementations. 