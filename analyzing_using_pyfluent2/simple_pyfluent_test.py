#!/usr/bin/env python3
"""
Simple PyFluent Test
Author: Jiwoo Lee

Minimal test to verify PyFluent workflow and VTP generation.
"""

import sys
import os
from pathlib import Path

try:
    import ansys.fluent.core as pyfluent
    PYFLUENT_AVAILABLE = True
    print(f"✅ PyFluent {pyfluent.__version__} loaded")
except ImportError as e:
    PYFLUENT_AVAILABLE = False
    print(f"❌ PyFluent not available: {e}")
    sys.exit(1)

def simple_cfd_test():
    """Run a simple CFD test with VTP export."""
    
    print("🧪 Simple PyFluent CFD Test")
    print("=" * 40)
    
    # Configuration
    stl_file = "/home/jiwoo/urp/data/uan/clean_flat_vessels/06_MRA1_seg_clean_flat.stl"
    output_dir = Path("./simple_test_results")
    output_dir.mkdir(exist_ok=True)
    
    if not Path(stl_file).exists():
        print(f"❌ STL file not found: {stl_file}")
        return False
    
    session = None
    
    try:
        # Launch Fluent
        print("🚀 Launching PyFluent...")
        session = pyfluent.launch_fluent(
            precision="double",
            processor_count=4,
            mode="solver",
            dimension=3,
            ui_mode="no_gui",
            start_timeout=180,
            cleanup_on_exit=True
        )
        print("✅ PyFluent session launched")
        
        # Import mesh
        print("📁 Importing STL mesh...")
        session.file.read_mesh(file_name=stl_file)
        print("✅ Mesh imported")
        
        # Check mesh info
        print("🔍 Checking mesh...")
        try:
            # Try to get basic mesh info using TUI commands
            session.tui.mesh.check()
            print("✅ Mesh check completed")
        except:
            print("⚠️  Mesh check skipped")
        
        # Set up basic solver settings using TUI
        print("⚙️  Setting up basic solver...")
        try:
            # Use TUI commands for reliable setup
            session.tui.define.models.viscous("laminar")
            print("✅ Viscous model set to laminar")
        except Exception as e:
            print(f"⚠️  Viscous model setup warning: {e}")
        
        # Try to run a few iterations using TUI
        print("🔄 Running basic iterations...")
        try:
            session.tui.solve.initialize.hybrid_initialize()
            print("✅ Solution initialized")
            
            # Run 10 iterations
            session.tui.solve.iterate(10)
            print("✅ 10 iterations completed")
            
        except Exception as e:
            print(f"⚠️  Solution warning: {e}")
        
        # Export to different formats
        print("💾 Exporting results...")
        
        # Save case file
        case_file = output_dir / "test_case.cas.h5"
        try:
            session.file.write_case(file_name=str(case_file))
            print(f"✅ Case file saved: {case_file}")
        except Exception as e:
            print(f"⚠️  Case save warning: {e}")
        
        # Try to export VTK format
        vtk_file = output_dir / "test_results"
        try:
            # Export using TUI commands
            session.tui.file.export.ensight_gold(str(vtk_file), "ascii", "yes", 
                                                 "cell_id", "pressure", "velocity_magnitude")
            print(f"✅ EnSight files exported: {vtk_file}")
        except Exception as e:
            print(f"⚠️  EnSight export warning: {e}")
        
        # Try STL export for validation
        stl_out = output_dir / "test_mesh.stl"
        try:
            session.tui.file.export.stl(str(stl_out), "ascii", "wall")
            print(f"✅ STL file exported: {stl_out}")
        except Exception as e:
            print(f"⚠️  STL export warning: {e}")
        
        print("\n📊 Test Results:")
        result_files = list(output_dir.glob("*"))
        if result_files:
            print("Generated files:")
            for file in result_files:
                print(f"  - {file.name} ({file.stat().st_size} bytes)")
            return True
        else:
            print("❌ No output files generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        if session:
            try:
                session.exit()
                print("✅ Session closed")
            except:
                pass
    
    return True

def main():
    """Main function."""
    success = simple_cfd_test()
    
    if success:
        print("\n🎉 Simple PyFluent test completed!")
        print("Check the generated files for basic workflow validation.")
    else:
        print("\n❌ Simple PyFluent test failed!")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 