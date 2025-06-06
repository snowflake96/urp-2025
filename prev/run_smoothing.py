#!/usr/bin/env python3
"""
Run aneurysm smoothing process to address blocky artifacts
"""

import sys
sys.path.append('.')
sys.path.append('smoothing')

from smoothing.aneurysm_smoother import AneurysmSmoother

def main():
    print("=== Starting Aneurysm Smoothing Process ===\n")
    
    # Initialize smoother
    print("Initializing AneurysmSmoother...")
    smoother = AneurysmSmoother()
    print("✓ Setup successful!")
    
    print("\nAvailable smoothing methods:")
    print("1. adaptive - Adaptive Gaussian smoothing targeting specific sharpness")
    print("2. volume_preserving - Volume-preserving Gaussian smoothing")  
    print("3. morphological - Morphological operations with smoothing")
    
    # Start with adaptive smoothing on first 3 patients
    print("\n=== Testing Adaptive Smoothing (first 3 patients) ===")
    adaptive_results = smoother.process_all_patients(
        method='adaptive', 
        max_patients=3,
        target_sharpness=0.5
    )
    
    print("\n=== Testing Volume-Preserving Smoothing (first 3 patients) ===")
    volume_results = smoother.process_all_patients(
        method='volume_preserving',
        max_patients=3, 
        smoothing_strength=1.0
    )
    
    print("\n=== Testing Morphological Smoothing (first 3 patients) ===")
    morph_results = smoother.process_all_patients(
        method='morphological',
        max_patients=3,
        kernel_size=3
    )
    
    print("\n=== Smoothing Test Complete ===")
    print(f"Results saved to: ~/urp/data/smoothing/")
    print("\nTo process all patients with best method:")
    print("smoother.process_all_patients(method='adaptive', target_sharpness=0.5)")

if __name__ == "__main__":
    main() 