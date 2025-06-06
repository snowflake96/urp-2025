#!/usr/bin/env python3
"""
Run aneurysm cropping process
"""

from aneurysm_cropper import AneurysmCropper

def main():
    print("=== Starting Aneurysm Cropping Process ===\n")
    
    # Initialize cropper
    print("Initializing AneurysmCropper...")
    cropper = AneurysmCropper()
    print("✓ Setup successful!")
    
    # Start with a test of first 3 patients
    print("\nStarting cropping process with first 3 patients for testing...")
    cropper.process_all_patients(max_patients=3, crop_size=(64, 64, 32))
    
    print("\n=== Test Complete ===")
    print("To process all patients, run:")
    print("cropper.process_all_patients(crop_size=(64, 64, 32))")

if __name__ == "__main__":
    main() 