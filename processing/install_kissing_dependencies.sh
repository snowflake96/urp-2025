#!/bin/bash
# Install dependencies for advanced kissing artifact removal

echo "Installing dependencies for kissing artifact removal pipelines..."
echo "This will install VMTK and GUDHI in the current conda environment"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Get current conda environment
CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "Current conda environment: $CONDA_ENV"
echo ""

# Ask for confirmation
read -p "Install VMTK and GUDHI in '$CONDA_ENV' environment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

echo ""
echo "=== Installing VMTK ==="
echo "This provides centerline extraction and vessel analysis tools"
conda install -c vmtk -y vtk itk vmtk

if [ $? -eq 0 ]; then
    echo "✓ VMTK installed successfully"
else
    echo "✗ VMTK installation failed"
    echo "Try manual installation:"
    echo "  conda install -c vmtk vtk itk vmtk"
    echo "  or"
    echo "  pip install vmtk"
fi

echo ""
echo "=== Installing GUDHI ==="
echo "This provides topological data analysis tools"
conda install -c conda-forge -y gudhi

if [ $? -eq 0 ]; then
    echo "✓ GUDHI installed successfully"
else
    echo "✗ GUDHI installation failed"
    echo "Try manual installation:"
    echo "  conda install -c conda-forge gudhi"
    echo "  or"
    echo "  pip install gudhi"
fi

echo ""
echo "=== Verifying installations ==="

# Test VMTK
python -c "import vmtk; print('✓ VMTK is working')" 2>/dev/null || echo "✗ VMTK import failed"

# Test GUDHI
python -c "import gudhi; print('✓ GUDHI is working')" 2>/dev/null || echo "✗ GUDHI import failed"

echo ""
echo "=== Optional: Install GPU support ==="
echo "For GPU acceleration (if you have CUDA):"
echo "  pip install cupy-cuda11x  # Replace 11x with your CUDA version"

echo ""
echo "Installation complete!"
echo ""
echo "You can now use:"
echo "  - processing/vmtk_kissing_removal.py"
echo "  - processing/topological_kissing_detection.py"
echo "  - processing/comprehensive_kissing_pipeline.py" 