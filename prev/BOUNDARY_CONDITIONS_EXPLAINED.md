# Boundary Conditions Creation for Cerebrovascular Aneurysm Analysis

## Overview

This document explains how boundary conditions are systematically created for PyAnsys finite element analysis (FEA) of cerebrovascular aneurysms. The process transforms medical imaging data into physics-based computational models for stress analysis and rupture risk prediction.

## Table of Contents

1. [Conceptual Framework](#conceptual-framework)
2. [Data Processing Pipeline](#data-processing-pipeline)  
3. [Anatomical Region Classification](#anatomical-region-classification)
4. [Boundary Condition Types](#boundary-condition-types)
5. [Mesh Generation Process](#mesh-generation-process)
6. [Physics-Based Parameters](#physics-based-parameters)
7. [Implementation Workflow](#implementation-workflow)
8. [Validation and Quality Control](#validation-and-quality-control)

---

## 1. Conceptual Framework

### 1.1 Medical-to-Engineering Translation

**Medical Input** → **Engineering Model** → **Physics Simulation**

```
MRA Scan (DICOM) → Segmented Vessel → 3D Mesh → Boundary Conditions → FEA Analysis
```

The boundary condition creation process bridges medical imaging data with computational fluid dynamics (CFD) and finite element analysis (FEA) requirements.

### 1.2 Key Principles

1. **Anatomical Accuracy**: Boundary conditions must reflect physiological reality
2. **Regional Specificity**: Each anatomical region has unique hemodynamic characteristics  
3. **Clinical Relevance**: Parameters align with medical knowledge and patient data
4. **Computational Feasibility**: Models must be solvable within reasonable time/resources

---

## 2. Data Processing Pipeline

### 2.1 Input Data Structure

```
Patient Data:
├── Demographics (Age, Gender, Medical History)
├── MRA Scans (Raw NIfTI files, ~50-130 MB each)
├── Segmentation Masks (Binary vessel masks, ~0.1-0.4 MB)
└── Anatomical Annotations (Excel spreadsheet with region classifications)
```

### 2.2 Processing Steps

#### Step 1: Region-Based Cropping
```python
# Anatomically-guided cropping based on medical annotations
raw_cropped, mask_cropped, crop_info = crop_region(
    raw_data_smoothed, mask_data, region="MCA", 
    crop_size=(80, 80, 60)  # Tailored to anatomy
)
```

#### Step 2: Mesh Generation
```python
# Convert binary mask to 3D surface mesh using marching cubes
vertices, faces, normals = measure.marching_cubes(
    mask_data, level=0.5, spacing=voxel_spacing
)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
```

#### Step 3: Boundary Region Identification
```python
# Identify inlet/outlet regions and wall surfaces
boundary_regions = identify_boundary_regions(mask_cropped, crop_info)
```

---

## 3. Anatomical Region Classification

### 3.1 Cerebrovascular Regions

Each anatomical region has specific boundary condition requirements:

| Region | Description | Hemodynamic Profile | Risk Factors |
|--------|-------------|-------------------|--------------|
| **MCA** | Middle Cerebral Artery | High flow, lateral branches | Hypertension |
| **ACA** | Anterior Cerebral Artery | Moderate flow, frontal supply | Age-related |
| **Acom** | Anterior Communicating | Variable flow, communicating | High risk location |
| **ICA (total)** | Internal Carotid Artery | High pressure, main supply | Atherosclerosis |
| **ICA (noncavernous)** | ICA supraclinoid | Thin walls, high pressure | Wall thinning |
| **ICA (cavernous)** | ICA cavernous segm | Protected by skull base | Trauma |
| **Pcom** | Posterior Communicating | Communicating flow | Congenital variants |
| **BA** | Basilar Artery | Posterior circulation | Vertebral dominance |
| **PCA** | Posterior Cerebral | Cortical branches | Posterior infarcts |

### 3.2 Region-Specific Characteristics

#### Example: ICA (noncavernous)
```json
{
  "description": "ICA non-cavernous - supraclinoid region",
  "boundary_conditions": {
    "inlet_location": "cavernous_exit",
    "outlet_location": "terminal_bifurcation", 
    "pressure_profile": "high_flow_pulsatile",
    "wall_properties": "thin_intracranial"
  },
  "hemodynamics": {
    "typical_flow_rate": "4-6 mL/s",
    "pressure_range": "80-120 mmHg", 
    "reynolds_number": "1500-2500",
    "wall_shear_stress": "1-3 Pa"
  }
}
```

---

## 4. Boundary Condition Types

### 4.1 Static Analysis Boundary Conditions

**Purpose**: Steady-state stress analysis under mean arterial pressure

```python
# Static pressure loading
mean_arterial_pressure = (systolic + 2*diastolic) / 3  # ~100 mmHg
internal_pressure = 13300 Pa  # Convert to Pascals

# Applied to:
# - Inner vessel surface (pressure loading)
# - Vessel ends (fixed supports or prescribed displacement)
# - Aneurysm dome (peak stress location)
```

**Boundary Conditions:**
- **Inlet**: Fixed displacement (proximal vessel end)
- **Outlet**: Free boundary or prescribed pressure  
- **Wall Surface**: Internal pressure loading
- **External Surface**: Atmospheric pressure (0 Pa reference)

### 4.2 Transient Analysis Boundary Conditions

**Purpose**: Time-varying analysis capturing cardiac cycle effects

```python
# Cardiac cycle pressure profile (20 time steps)
def cardiac_pressure_profile(t, heart_rate=70):
    cycle_time = 60.0 / heart_rate  # 0.857 seconds
    phase = 2 * np.pi * t / cycle_time
    
    # Fourier series approximation of arterial pressure
    systolic = 16000  # 120 mmHg
    diastolic = 10700  # 80 mmHg
    
    pressure = diastolic + (systolic - diastolic) * (
        0.5 * (1 + np.cos(phase + np.pi)) if phase < np.pi else 0.3
    )
    return pressure
```

**Time-Dependent Loading:**
- **0.0s - 0.3s**: Systolic phase (peak pressure)
- **0.3s - 0.8s**: Diastolic phase (lower pressure)
- **Frequency**: 70 beats/min (typical resting heart rate)

### 4.3 Fluid-Structure Interaction (FSI) Boundary Conditions

**Purpose**: Coupled fluid-solid analysis for comprehensive hemodynamics

```python
# Fluid domain boundary conditions
fluid_bc = {
    "inlet": {
        "type": "velocity_inlet",
        "profile": "parabolic",  # Fully developed flow
        "peak_velocity": 0.5,   # m/s
        "time_varying": True
    },
    "outlet": {
        "type": "pressure_outlet", 
        "pressure": 0,          # Gauge pressure
        "backflow_prevention": True
    },
    "walls": {
        "type": "no_slip",     # Zero velocity at walls
        "moving_mesh": True    # FSI coupling
    }
}

# Solid domain boundary conditions  
solid_bc = {
    "vessel_ends": "fixed_support",
    "fsi_interface": "pressure_from_fluid",
    "external_surface": "atmospheric_pressure"
}
```

---

## 5. Mesh Generation Process

### 5.1 Surface Mesh Extraction

```python
def create_stl_from_mask(mask_data, voxel_spacing, output_path):
    """
    Convert binary segmentation mask to STL surface mesh
    
    Process:
    1. Marching cubes algorithm extracts isosurface
    2. Mesh smoothing reduces staircase artifacts
    3. Quality checks ensure watertight geometry
    4. Export to STL for PyAnsys import
    """
    
    # Extract surface using marching cubes
    vertices, faces, normals = measure.marching_cubes(
        mask_data, level=0.5, spacing=voxel_spacing
    )
    
    # Create and smooth mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh = mesh.smoothed(iterations=2)
    
    # Quality metrics
    quality_info = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces), 
        'volume': mesh.volume,
        'surface_area': mesh.area,
        'is_watertight': mesh.is_watertight,
        'bounds': mesh.bounds
    }
    
    return mesh, quality_info
```

### 5.2 Mesh Quality Requirements

| Parameter | Requirement | Typical Range |
|-----------|-------------|---------------|
| **Vertices** | 15,000 - 35,000 | 20,000 ± 8,000 |
| **Faces** | 15,000 - 35,000 | 22,000 ± 8,000 |
| **Aspect Ratio** | < 3.0 | 1.2 - 2.5 |
| **Watertight** | Required | 100% |
| **Surface Smoothness** | C1 continuity | Gaussian curvature |

---

## 6. Physics-Based Parameters

### 6.1 Material Properties

#### Healthy Vessel Wall
```python
vessel_material = {
    "young_modulus": 2.0e6,      # Pa (2.0 MPa)
    "poisson_ratio": 0.45,       # Nearly incompressible
    "density": 1050,             # kg/m³
    "wall_thickness": 0.0005,    # m (0.5 mm)
    "yield_strength": 0.8e6,     # Pa (0.8 MPa)
    "fatigue_limit": 0.4e6       # Pa (0.4 MPa)
}
```

#### Aneurysm Wall (Weakened)
```python
aneurysm_material = {
    "young_modulus": 1.0e6,      # Pa (50% reduction)
    "poisson_ratio": 0.45,       # Same as healthy
    "density": 1050,             # kg/m³  
    "wall_thickness": 0.0003,    # m (40% thinner)
    "yield_strength": 0.5e6,     # Pa (reduced strength)
    "collagen_degradation": 0.3   # 30% degradation factor
}
```

#### Blood Properties
```python
blood_properties = {
    "density": 1060,             # kg/m³
    "dynamic_viscosity": 0.0035, # Pa·s  
    "kinematic_viscosity": 3.3e-6, # m²/s
    "bulk_modulus": 2.2e9,       # Pa (nearly incompressible)
    "temperature": 37            # °C (body temperature)
}
```

### 6.2 Hemodynamic Parameters

#### Pressure Loading
```python
hemodynamic_conditions = {
    "systolic_pressure": 16000,   # Pa (120 mmHg)
    "diastolic_pressure": 10700,  # Pa (80 mmHg) 
    "mean_arterial_pressure": 13300, # Pa (100 mmHg)
    "pulse_pressure": 5300,       # Pa (40 mmHg)
    "heart_rate": 70,            # beats/min
    "cardiac_output": 5.0        # L/min
}
```

#### Flow Characteristics
```python
flow_parameters = {
    "peak_velocity": 0.5,        # m/s (systolic peak)
    "mean_velocity": 0.3,        # m/s (time-averaged)
    "reynolds_number": 1800,     # Typically 1500-2500
    "womersley_number": 4.5,     # Unsteady flow parameter
    "flow_rate": 4.5e-6         # m³/s (4.5 mL/s)
}
```

---

## 7. Implementation Workflow

### 7.1 Automated Boundary Condition Generation

```python
class BoundaryConditionGenerator:
    """
    Automated boundary condition generation for cerebrovascular FEA
    """
    
    def __init__(self, patient_data, region_type):
        self.patient_data = patient_data
        self.region_type = region_type
        self.bc_template = self.load_region_template(region_type)
    
    def generate_static_bc(self):
        """Generate static analysis boundary conditions"""
        bc = {
            "pressure_loading": self.calculate_pressure_loading(),
            "support_conditions": self.define_support_conditions(),
            "material_properties": self.assign_material_properties(),
            "contact_definitions": self.define_contact_regions()
        }
        return bc
    
    def generate_transient_bc(self):
        """Generate time-dependent boundary conditions"""
        time_points = np.linspace(0, self.cardiac_cycle_time, 20)
        
        bc_sequence = []
        for t in time_points:
            bc = {
                "time": t,
                "pressure": self.cardiac_pressure_profile(t),
                "flow_rate": self.cardiac_flow_profile(t),
                "boundary_conditions": self.time_dependent_bc(t)
            }
            bc_sequence.append(bc)
        
        return bc_sequence
```

### 7.2 PyAnsys Integration

```python
def apply_boundary_conditions_to_ansys(bc_data, ansys_model):
    """
    Apply generated boundary conditions to PyAnsys model
    """
    
    # Material assignment
    ansys_model.materials.add("vessel_wall")
    ansys_model.materials["vessel_wall"].assign_properties(
        young_modulus=bc_data["material"]["young_modulus"],
        poisson_ratio=bc_data["material"]["poisson_ratio"],
        density=bc_data["material"]["density"]
    )
    
    # Pressure loading
    vessel_interior = ansys_model.geometry.get_named_selection("vessel_interior")
    ansys_model.loads.pressure.add(
        location=vessel_interior,
        magnitude=bc_data["pressure"]["internal_pressure"]
    )
    
    # Support conditions
    vessel_inlet = ansys_model.geometry.get_named_selection("vessel_inlet")
    ansys_model.constraints.fixed_support.add(location=vessel_inlet)
    
    # Mesh controls
    ansys_model.mesh.set_element_size(
        size=bc_data["mesh"]["target_element_size"]
    )
    
    return ansys_model
```

---

## 8. Validation and Quality Control

### 8.1 Geometric Validation

```python
def validate_geometry(mesh_info, region_type):
    """
    Validate mesh geometry against clinical expectations
    """
    
    validation_results = {
        "mesh_quality": "PASS" if mesh_info["is_watertight"] else "FAIL",
        "size_check": "PASS" if 15000 < mesh_info["vertices"] < 50000 else "WARNING",
        "aspect_ratio": calculate_aspect_ratio(mesh_info),
        "volume_realistic": validate_volume(mesh_info["volume"], region_type)
    }
    
    return validation_results
```

### 8.2 Physics Validation

```python
def validate_boundary_conditions(bc_data):
    """
    Validate boundary conditions against physiological ranges
    """
    
    checks = {
        "pressure_range": 80 <= bc_data["pressure_mmHg"] <= 180,
        "reynolds_number": 1000 <= bc_data["reynolds"] <= 3000,
        "wall_stress": bc_data["wall_stress"] < bc_data["yield_strength"],
        "flow_rate": 2e-6 <= bc_data["flow_rate"] <= 8e-6  # m³/s
    }
    
    return all(checks.values()), checks
```

### 8.3 Clinical Correlation

```python
def correlate_with_clinical_data(bc_results, patient_data):
    """
    Correlate computational results with clinical observations
    """
    
    correlations = {
        "age_factor": apply_age_correction(patient_data["age"]),
        "gender_factor": apply_gender_correction(patient_data["gender"]),
        "hypertension": adjust_for_hypertension(patient_data["bp_history"]),
        "aneurysm_size": correlate_size_risk(bc_results["geometry"])
    }
    
    return correlations
```

---

## 9. Example: Complete Boundary Condition Creation

### 9.1 Input Data
```
Patient ID: 33406918
Region: MCA (Middle Cerebral Artery)
Age: 65, Gender: Female
Hypertension: Yes
Aneurysm Size: 7.2 mm
```

### 9.2 Generated Boundary Conditions

```json
{
  "region_name": "MCA",
  "patient_id": 33406918,
  "boundary_conditions": {
    "inlet_location": "proximal_end",
    "outlet_location": "distal_end",
    "pressure_profile": "pulsatile",
    "wall_properties": "medium_stiffness"
  },
  "mesh_info": {
    "vertices": 32985,
    "faces": 31278,
    "volume": 1.47e-6,
    "surface_area": 0.00284,
    "is_watertight": true
  },
  "physics_parameters": {
    "internal_pressure": 13300,
    "young_modulus": 1800000,
    "wall_thickness": 0.0004,
    "reynolds_number": 1850,
    "peak_velocity": 0.48
  }
}
```

### 9.3 PyAnsys Implementation

```python
# Load mesh and apply boundary conditions
mesh_file = "MCA_mesh.stl"
bc_file = "MCA_boundary_conditions.json" 

# Initialize PyAnsys model
mapdl = pyansys.Mapdl()
mapdl.prep7()

# Import geometry
mapdl.import_stl(mesh_file)

# Apply material properties
mapdl.mp("EX", 1, 1.8e6)  # Young's modulus
mapdl.mp("PRXY", 1, 0.45)  # Poisson's ratio
mapdl.mp("DENS", 1, 1050)  # Density

# Apply pressure loading
mapdl.sf("ALL", "PRES", 13300)  # Internal pressure

# Apply boundary constraints
mapdl.nsel("S", "LOC", "X", x_min)  # Select inlet nodes
mapdl.d("ALL", "ALL", 0)  # Fix all DOF at inlet

# Solve
mapdl.solve()
results = mapdl.result
```

---

## 10. Summary

The boundary condition creation process systematically transforms medical imaging data into physics-based computational models through:

1. **Anatomically-guided processing** of MRA scans and segmentation masks
2. **Region-specific parameterization** based on cerebrovascular anatomy
3. **Physics-based material and loading conditions** derived from clinical data  
4. **Automated mesh generation** with quality validation
5. **Seamless PyAnsys integration** for finite element analysis

This comprehensive approach ensures that computational models accurately represent both the geometric complexity and physiological reality of cerebrovascular aneurysms, enabling reliable stress analysis and rupture risk assessment.

---

## References

1. Sforza et al. "Hemodynamics of Cerebral Aneurysms" (2016)
2. Valencia et al. "Blood Flow Dynamics in Patient-Specific Aneurysm Models" (2018)
3. PyAnsys Documentation - MAPDL User Guide (2024)
4. Cebral et al. "Computational Fluid Dynamics Modeling of Intracranial Aneurysms" (2011)
5. Humphrey & Taylor. "Intracranial and Abdominal Aortic Aneurysms" (2008) 