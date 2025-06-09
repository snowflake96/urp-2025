# PyFluent 3D CFD Analysis Classes

Professional Python classes for STL → MSH → CFD → VTK workflow.

## Classes Created

1. **stl_to_msh_converter.py** - Convert STL to MSH using Ansys meshing API
2. **boundary_conditions_setup.py** - Setup hemodynamic boundary conditions  
3. **pyfluent_3d_analyzer.py** - Run 3D CFD and export VTP/VTU files
4. **complete_workflow_demo.py** - Complete workflow demonstration

## Quick Usage

```python
# Complete workflow
from complete_workflow_demo import run_complete_workflow
success, stats = run_complete_workflow("patient_id", processor_count=16)

# Individual classes
from stl_to_msh_converter import STLtoMSHConverter
from boundary_conditions_setup import BoundaryConditionsSetup  
from pyfluent_3d_analyzer import PyFluent3DAnalyzer
```

## Features

- Importable and reusable classes
- Ansys 2025 R1 PyFluent integration
- Parallel processing (16-32 cores)
- Automatic boundary conditions for hemodynamics
- Direct VTP/VTU export for ParaView
- Context managers for cleanup
- Comprehensive error handling 