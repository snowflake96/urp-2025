import sys
sys.path.append('processing')

from NIfTI_find_aneurysm import AneurysmDetector
import json
import numpy as np
from scipy import ndimage

# Load diagnosis
with open('diagnosis_01_MRA1_seg.json', 'r') as f:
    diagnosis = json.load(f)

detector = AneurysmDetector(" example_data/01_MRA1_seg.nii.gz", diagnosis)
labels = detector.get_aneurysm_labels()

print(f"Found labels: {labels}")

for label in labels:
    print(f"\nProcessing label: {label}")
    
    # Handle floating point precision issues
    if isinstance(label, float):
        label_mask = np.abs(detector.image_data - label) < 1e-6
    else:
        label_mask = (detector.image_data == label)
    
    print(f"Label mask volume: {np.sum(label_mask)}")
    
    # First, try with original mask
    labeled_array, num_components = ndimage.label(label_mask)
    print(f"Original connected components: {num_components}")
    
    # If only one component but we expect more, try morphological separation
    expected_count = diagnosis.get('expected_aneurysm_count', 1)
    print(f"Expected count: {expected_count}")
    
    if num_components == 1 and expected_count > 1:
        print("Trying morphological separation...")
        
        from scipy.ndimage import binary_erosion, binary_dilation
        
        # Apply erosion to break connections
        eroded = binary_erosion(label_mask, iterations=2)
        labeled_eroded, num_eroded = ndimage.label(eroded)
        print(f"After erosion: {num_eroded} components")
        
        if num_eroded > 1:
            # Get component sizes and keep only the largest ones
            component_sizes = []
            for comp_id in range(1, num_eroded + 1):
                comp_mask = (labeled_eroded == comp_id)
                size = np.sum(comp_mask)
                component_sizes.append((comp_id, size))
                print(f"  Component {comp_id}: {size} voxels")
            
            # Sort by size and keep only the top components
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            max_components = min(expected_count, len(component_sizes))
            print(f"Max components to keep: {max_components}")
            
            # Only keep components that are reasonably large
            min_size = np.sum(label_mask) * 0.05  # At least 5% of total volume
            print(f"Minimum size threshold: {min_size}")
            
            kept_components = []
            for comp_id, size in component_sizes[:max_components]:
                if size >= min_size:
                    kept_components.append(comp_id)
                    print(f"  Keeping component {comp_id} with {size} voxels")
            
            if len(kept_components) > 1:
                print(f"Creating separated mask with {len(kept_components)} components")
                
                # Dilate each kept component back to approximate original size
                separated_mask = np.zeros_like(label_mask)
                new_comp_id = 1
                for comp_id in kept_components:
                    comp_mask = (labeled_eroded == comp_id)
                    # Dilate back
                    dilated = binary_dilation(comp_mask, iterations=3)
                    # Only keep parts that were in original mask
                    dilated = dilated & label_mask
                    separated_mask[dilated] = new_comp_id
                    print(f"  Separated component {new_comp_id}: {np.sum(dilated)} voxels")
                    new_comp_id += 1
                
                labeled_array = separated_mask
                num_components = len(kept_components)
                print(f"Final num_components: {num_components}")
                
                # Verify the final result
                final_labeled, final_count = ndimage.label(labeled_array > 0)
                print(f"Verification - actual components in final mask: {final_count}")
                
                # Check each component individually
                for i in range(1, num_components + 1):
                    comp_volume = np.sum(labeled_array == i)
                    print(f"  Final component {i}: {comp_volume} voxels")
    
    print(f"Final decision: num_components = {num_components}")
    if num_components > 1:
        print("Will process as multiple components")
    else:
        print("Will process as single component") 