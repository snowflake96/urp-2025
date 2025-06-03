#!/usr/bin/env python3
"""
Visualization script for cropped aneurysms
"""

import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns

class AneurysmVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.summary_file = self.output_dir / "aneurysm_cropping_summary.csv"
        self.summary_df = None
        
        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file)
            print(f"Loaded summary with {len(self.summary_df)} aneurysms")
        else:
            print("Summary file not found!")
    
    def plot_volume_distribution(self):
        """Plot distribution of aneurysm volumes"""
        if self.summary_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0,0].hist(self.summary_df['volume'], bins=50, alpha=0.7)
        axes[0,0].set_xlabel('Volume (voxels)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Aneurysm Volume Distribution')
        
        # Log scale histogram
        axes[0,1].hist(np.log10(self.summary_df['volume'] + 1), bins=50, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Log10(Volume + 1)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Aneurysm Volume Distribution (Log Scale)')
        
        # Box plot by patient
        if 'patient_id' in self.summary_df.columns:
            self.summary_df.boxplot(column='volume', by='patient_id', ax=axes[1,0])
            axes[1,0].set_title('Volume Distribution by Patient')
            axes[1,0].set_xlabel('Patient ID')
            
        # Volume vs anatomical location
        if 'anatomical_locations' in self.summary_df.columns:
            # Count aneurysms by location
            locations = self.summary_df['anatomical_locations'].value_counts().head(10)
            axes[1,1].bar(range(len(locations)), locations.values)
            axes[1,1].set_xticks(range(len(locations)))
            axes[1,1].set_xticklabels(locations.index, rotation=45, ha='right')
            axes[1,1].set_title('Aneurysms by Anatomical Location')
            axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_aneurysm(self, patient_id, folder_name, aneurysm_index, slice_range=None):
        """Visualize a specific aneurysm"""
        
        # Find the aneurysm files
        patient_dir = self.output_dir / f"patient_{patient_id}" / folder_name
        raw_file = patient_dir / f"aneurysm_{aneurysm_index}_raw.nii.gz"
        mask_file = patient_dir / f"aneurysm_{aneurysm_index}_mask.nii.gz"
        metadata_file = patient_dir / f"aneurysm_{aneurysm_index}_metadata.json"
        
        if not all([f.exists() for f in [raw_file, mask_file, metadata_file]]):
            print(f"Files not found for patient {patient_id}, {folder_name}, aneurysm {aneurysm_index}")
            return
        
        # Load data
        raw_img = nib.load(raw_file)
        mask_img = nib.load(mask_file)
        raw_data = raw_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Determine slice range
        if slice_range is None:
            # Find slices with aneurysm
            aneurysm_slices = np.where(np.sum(mask_data, axis=(1,2)) > 0)[0]
            if len(aneurysm_slices) > 0:
                slice_range = (max(0, aneurysm_slices[0] - 2), 
                              min(raw_data.shape[0], aneurysm_slices[-1] + 3))
            else:
                slice_range = (raw_data.shape[0]//2 - 5, raw_data.shape[0]//2 + 5)
        
        # Create visualization
        n_slices = slice_range[1] - slice_range[0]
        cols = min(5, n_slices)
        rows = (n_slices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, slice_idx in enumerate(range(slice_range[0], slice_range[1])):
            row = i // cols
            col = i % cols
            
            if slice_idx < raw_data.shape[0]:
                # Raw image
                axes[row, col].imshow(raw_data[slice_idx], cmap='gray')
                
                # Overlay mask
                mask_slice = mask_data[slice_idx]
                if np.sum(mask_slice) > 0:
                    axes[row, col].contour(mask_slice, colors='red', linewidths=2)
                
                axes[row, col].set_title(f'Slice {slice_idx}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_slices, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Patient {patient_id} - {folder_name} - Aneurysm {aneurysm_index}\n'
                    f'Volume: {metadata["aneurysm_info"]["volume"]} voxels\n'
                    f'Center: {metadata["aneurysm_info"]["center"]}\n'
                    f'Locations: {", ".join(metadata.get("anatomical_locations", []))}', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'aneurysm_viz_p{patient_id}_{folder_name}_a{aneurysm_index}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print clinical info
        if 'clinical_info' in metadata:
            clinical = metadata['clinical_info']
            print(f"\nClinical Information:")
            print(f"  Age: {clinical.get('연령', 'N/A')}")
            print(f"  Gender: {clinical.get('성별', 'N/A')}")
            print(f"  Smoking: {clinical.get('Smoking_record_combined (non:0, current:1, ex:2)', 'N/A')}")
            print(f"  Hypertension: {clinical.get('HT', 'N/A')}")
            print(f"  Diabetes: {clinical.get('DM', 'N/A')}")
    
    def show_largest_aneurysms(self, n=5):
        """Show the largest aneurysms"""
        if self.summary_df is None:
            return
        
        largest = self.summary_df.nlargest(n, 'volume')
        print(f"\n=== {n} Largest Aneurysms ===")
        
        for idx, row in largest.iterrows():
            print(f"\nRank {idx+1}:")
            print(f"  Patient: {row['patient_id']}, Scan: {row['folder_name']}, Aneurysm: {row['aneurysm_index']}")
            print(f"  Volume: {row['volume']:,} voxels")
            print(f"  Center: ({row['center_x']}, {row['center_y']}, {row['center_z']})")
            print(f"  Locations: {row.get('anatomical_locations', 'N/A')}")
            
            # Visualize this aneurysm
            self.visualize_aneurysm(row['patient_id'], row['folder_name'], row['aneurysm_index'])
    
    def show_smallest_aneurysms(self, n=5):
        """Show the smallest aneurysms"""
        if self.summary_df is None:
            return
        
        smallest = self.summary_df.nsmallest(n, 'volume')
        print(f"\n=== {n} Smallest Aneurysms ===")
        
        for idx, row in smallest.iterrows():
            print(f"\nRank {idx+1}:")
            print(f"  Patient: {row['patient_id']}, Scan: {row['folder_name']}, Aneurysm: {row['aneurysm_index']}")
            print(f"  Volume: {row['volume']} voxels")
            print(f"  Center: ({row['center_x']}, {row['center_y']}, {row['center_z']})")
            print(f"  Locations: {row.get('anatomical_locations', 'N/A')}")
    
    def analyze_by_patient(self):
        """Analyze aneurysms by patient"""
        if self.summary_df is None:
            return
        
        patient_stats = self.summary_df.groupby('patient_id').agg({
            'volume': ['count', 'mean', 'std', 'min', 'max'],
            'aneurysm_index': 'max'  # Number of aneurysms per patient
        }).round(2)
        
        patient_stats.columns = ['Count', 'Mean_Volume', 'Std_Volume', 'Min_Volume', 'Max_Volume', 'Max_Index']
        
        print("\n=== Patient Analysis ===")
        print(patient_stats)
        
        return patient_stats

def main():
    print("=== Aneurysm Visualization Tool ===\n")
    
    viz = AneurysmVisualizer()
    
    # Plot volume distributions
    print("1. Generating volume distribution plots...")
    viz.plot_volume_distribution()
    
    # Show patient analysis
    print("\n2. Patient analysis...")
    viz.analyze_by_patient()
    
    # Show some example aneurysms
    print("\n3. Showing largest aneurysms...")
    viz.show_largest_aneurysms(n=3)
    
    print("\n4. Showing smallest aneurysms (info only)...")
    viz.show_smallest_aneurysms(n=3)
    
    print("\nVisualization complete!")
    print("To visualize a specific aneurysm, use:")
    print("viz.visualize_aneurysm(patient_id, folder_name, aneurysm_index)")

if __name__ == "__main__":
    main() 