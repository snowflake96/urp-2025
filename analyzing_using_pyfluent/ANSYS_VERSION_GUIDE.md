# Ansys and PyFluent Version Compatibility Guide

## Available Ansys Versions on System

- **v211** - Ansys 2021 R1 (`/opt/cvbml/softwares/ansys_inc/v211`)
- **v251** - Ansys 2025 R1 (`/opt/cvbml/softwares/ansys_inc/v251`) ✅ Currently in use

## PyFluent Version Compatibility

### Current Setup
- **PyFluent**: 0.32.1 (latest)
- **Issues**: Many TUI commands return "menu not found" errors with Ansys 2025 R1

### Recommended Version Combinations

#### Option 1: Use Older PyFluent with Current Ansys
```bash
# Downgrade PyFluent for better compatibility
pip install ansys-fluent-core==0.20.0

# Set environment
export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251
```

#### Option 2: Use Older Ansys with Older PyFluent
```bash
# Install compatible PyFluent
pip install ansys-fluent-core==0.16.0

# Set environment for Ansys 2021 R1
export AWP_ROOT211=/opt/cvbml/softwares/ansys_inc/v211
export FLUENT_PATH=/opt/cvbml/softwares/ansys_inc/v211/fluent
```

#### Option 3: Use Journal Files (Most Reliable)
```python
# Create journal file for complex operations
journal_content = """
/file/read-case "mesh.msh"
/define/models/viscous/laminar yes
/solve/initialize/initialize-flow yes
/solve/iterate 100
/file/export/ensight-gold "results" yes
exit
yes
"""

# Run with subprocess
subprocess.run(['fluent', '3d', '-g', '-i', 'commands.jou'])
```

## Fixing the Spider-Like Effect in VTU Files

### Problem
Regular grid (hexahedral) mesh creates spider-like artifacts when visualized

### Solution
Use conforming tetrahedral mesh that respects vessel geometry:

1. **Tetrahedral Mesh** (Cell Type 10) instead of Hexahedral (Cell Type 12)
2. **Conforming Points** placed only inside vessel boundaries
3. **Adaptive Density** based on distance from vessel wall

### Results
- ✅ No spider artifacts
- ✅ Realistic flow patterns
- ✅ Better visualization in ParaView

## Recommended Workflow

### For Mesh Generation
```bash
# Use Ansys Meshing GUI or ICEM CFD
/opt/cvbml/softwares/ansys_inc/v251/ansys/bin/workbench251

# Or use open-source alternatives
# - GMSH
# - Salome
# - TetGen
```

### For CFD Analysis
```python
# Use PyFluent with journal files for reliability
import subprocess

# Create MSH file externally, then:
journal = """
/file/read-mesh "aneurysm.msh"
/mesh/scale 0.001 0.001 0.001
/define/materials/copy blood
/define/boundary-conditions/velocity-inlet inlet yes
/solve/iterate 500
/file/export/vtk "results" yes wall yes pressure velocity-magnitude wall-shear done
"""

with open('cfd_analysis.jou', 'w') as f:
    f.write(journal)

subprocess.run(['fluent', '3d', '-g', '-i', 'cfd_analysis.jou'])
```

## Version Compatibility Matrix

| PyFluent Version | Ansys 2021 R1 | Ansys 2025 R1 | Notes |
|-----------------|---------------|---------------|-------|
| 0.32.1 | ⚠️ Partial | ⚠️ API Issues | Latest, many deprecated methods |
| 0.20.0 | ✅ Good | ✅ Good | Recommended for stability |
| 0.16.0 | ✅ Excellent | ❌ Not tested | Best for 2021 R1 |

## Troubleshooting

### "menu not found" Errors
- Use older PyFluent version
- Use journal files instead of Python API
- Check TUI command syntax for version

### API Method Changes
```python
# Old API (< 0.20)
solver.tui.file.read_mesh()

# New API (> 0.20)
solver.file.read(file_type="mesh")
```

### License Issues
```bash
# Check license status
/opt/cvbml/softwares/ansys_inc/v251/fluent/bin/fluentlm_status

# Set license server
export ANSYSLMD_LICENSE_FILE=1055@license-server
```

## Conclusion

For best results:
1. **Use PyFluent 0.20.0** for balanced compatibility
2. **Create conforming tetrahedral meshes** to avoid visualization artifacts
3. **Use journal files** for complex operations
4. **Test with both Ansys versions** to find what works best 
 