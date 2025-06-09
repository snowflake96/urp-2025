# PyFluent STL → MSH → CFD → VTK Workflow Scripts

This directory contains three different approaches to convert STL files to VTK format through PyFluent CFD analysis:

## Available Scripts

### 1. `stl_to_msh_to_vtk_workflow.py`
- **Approach**: Traditional TUI (Text User Interface) commands
- **Pros**: Well-documented commands, direct control
- **Cons**: May have compatibility issues with newer PyFluent versions
- **Use when**: You're familiar with Fluent TUI commands

### 2. `modern_pyfluent_workflow.py`
- **Approach**: Modern PyFluent API with watertight meshing workflow
- **Pros**: More Pythonic, better error handling, uses latest API
- **Cons**: Requires PyFluent 0.20+ with proper API support
- **Use when**: You have the latest PyFluent version

### 3. `simple_stl_to_vtk_pipeline.py`
- **Approach**: Simplified workflow with fallback options
- **Pros**: Most compatible, includes journal file generation
- **Cons**: Basic CFD setup, may need manual intervention
- **Use when**: Other methods fail or for quick testing

## Workflow Overview

```
STL File → Fluent Meshing → Volume Mesh (MSH) → PyFluent Solver → VTK Export
```

## Requirements

- Ansys Fluent 2025 R1 (or compatible version)
- PyFluent 0.32.1
- Python 3.8+
- 32 cores recommended for parallel processing

## Usage

1. **Activate virtual environment**:
   ```bash
   source /path/to/venv/bin/activate
   ```

2. **Set Ansys environment**:
   ```bash
   export AWP_ROOT251=/opt/cvbml/softwares/ansys_inc/v251
   ```

3. **Run a script**:
   ```bash
   python stl_to_msh_to_vtk_workflow.py
   # or
   python modern_pyfluent_workflow.py
   # or
   python simple_stl_to_vtk_pipeline.py
   ```

## Input Files

The scripts expect STL files in: `../meshes/`
- `78_MRA1_seg_aneurysm.stl` (original)
- `78_MRA1_seg_aneurysm_CLEANED.stl` (cleaned version, preferred)
- `78_MRA1_seg_aneurysm_ASCII.stl` (ASCII format)

## Output Files

Results are saved to: `../results/<patient>_<workflow_name>/`
- `.msh` - Volume mesh file
- `.vtp` - Surface data (wall shear stress, pressure on walls)
- `.vtu` - Volume data (velocity field, pressure field)
- `workflow_summary.json` - Detailed workflow parameters

## Troubleshooting

### PyFluent API Issues
- Many methods have changed between versions
- Check PyFluent documentation for your version
- Use `dir(solver)` to explore available methods

### Mesh Generation Fails
- Ensure STL is watertight and properly oriented
- Try different mesh sizing parameters
- Check if Fluent meshing license is available

### Export Issues
- VTK export may not be directly available
- EnSight format can be converted to VTK using ParaView
- ASCII export can be post-processed

### Memory Issues
- Reduce processor count if running out of memory
- Increase mesh size to reduce element count
- Use coarser mesh for initial testing

## CFD Parameters

Default blood flow parameters:
- **Density**: 1060 kg/m³
- **Viscosity**: 0.0035 Pa·s (3.5 cP)
- **Inlet velocity**: 0.3 m/s
- **Outlet pressure**: 0 Pa (gauge)
- **Wall**: No-slip condition
- **Model**: Laminar flow (can be changed to turbulent)

## Next Steps

After successful VTK generation:
1. Open `.vtp`/`.vtu` files in ParaView
2. Visualize wall shear stress patterns
3. Analyze flow streamlines
4. Export images or animations

## Notes

- The existing `78_MRA1_seg_converted.msh` (128B) appears to be incomplete
- A proper volume mesh should be several MB in size
- CFD convergence typically requires 200-500 iterations
- Wall shear stress is the key hemodynamic parameter for aneurysm analysis
