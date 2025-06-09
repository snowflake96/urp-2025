#!/usr/bin/env python3
"""
Test Single Case PyFluent Analysis
Author: Jiwoo Lee

Simple test script to analyze a single aneurysm case with PyFluent
and generate VTP output for visualization.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from pyfluent_batch_analyzer import PyFluentBatchAnalyzer

def test_single_case():
    """Test analysis on a single case."""
    
    # Configuration
    data_dir = "/home/jiwoo/urp/data/uan/clean_flat_vessels"
    output_dir = "./single_case_test_results"
    n_cores = 8  # Use fewer cores for testing
    
    print("🧪 Testing Single Case PyFluent Analysis")
    print("=" * 50)
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"CPU Cores: {n_cores}")
    print("=" * 50)
    
    try:
        # Create analyzer
        analyzer = PyFluentBatchAnalyzer(
            data_dir=data_dir,
            output_dir=output_dir,
            n_cores=n_cores
        )
        
        # Find available cases
        cases = analyzer.find_analysis_cases()
        
        if not cases:
            print("❌ No analysis cases found!")
            return False
        
        # Test with the first case
        stl_file, bc_file = cases[0]
        case_name = stl_file.stem.replace("_clean_flat", "")
        
        print(f"\n🔍 Testing case: {case_name}")
        print(f"STL file: {stl_file.name}")
        print(f"BC file: {bc_file.name}")
        
        # Run single case analysis
        results = analyzer.run_batch_analysis(max_cases=1)
        
        if results and results[0]['success']:
            print(f"\n✅ Test successful!")
            print(f"Output files:")
            for output_file in results[0]['output_files']:
                print(f"  - {Path(output_file).name}")
            
            # Check if VTP file was created
            vtp_files = list(Path(output_dir).glob("*.vtp"))
            if vtp_files:
                print(f"\n🎯 VTP files generated:")
                for vtp_file in vtp_files:
                    print(f"  - {vtp_file}")
                    print(f"    Size: {vtp_file.stat().st_size / 1024:.1f} KB")
            
            return True
        else:
            print(f"❌ Test failed!")
            if results:
                print(f"Error: {results[0]['error_message']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main function."""
    success = test_single_case()
    
    if success:
        print(f"\n🎉 Single case test completed successfully!")
        print(f"Next steps:")
        print(f"  1. Check the VTP file in ParaView")
        print(f"  2. Run full batch analysis: python pyfluent_batch_analyzer.py --max-cases 5")
        print(f"  3. Process all cases: python pyfluent_batch_analyzer.py")
    else:
        print(f"\n❌ Single case test failed!")
        print(f"Check the logs and fix any issues before running batch analysis.")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 