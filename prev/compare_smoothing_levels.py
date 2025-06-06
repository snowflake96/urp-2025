#!/usr/bin/env python3
"""
Compare Different Smoothing Levels Visually
- Compare original, basic smoothed, and 3 more smoothing levels
- Generate side-by-side visualizations
- Show volume statistics and smoothness metrics
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmoothingComparator:
    """Compare different smoothing levels visually"""
    
    def __init__(self):
        self.original_dir = Path("/home/jiwoo/urp/data/uan/original")
        self.smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed")
        self.more_smoothed_dir = Path("/home/jiwoo/urp/data/uan/original_smoothed_more")
        self.output_dir = Path("/home/jiwoo/urp/data/uan/smoothing_comparisons")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_smoothing_series(self, patient_id: int, mra_num: int):
        """Load all smoothing levels for a specific patient/MRA"""
        
        base_name = f"{patient_id:02d}_MRA{mra_num}_seg"
        
        files = {
            'original': self.original_dir / f"{base_name}.nii.gz",
            'basic_smoothed': self.smoothed_dir / f"{base_name}_smoothed.nii.gz",
            'moderate': self.more_smoothed_dir / f"{base_name}_smoothed_moderate.nii.gz",
            'aggressive': self.more_smoothed_dir / f"{base_name}_smoothed_aggressive.nii.gz",
            'ultra': self.more_smoothed_dir / f"{base_name}_smoothed_ultra.nii.gz"
        }
        
        data = {}
        stats = {}
        
        for level, file_path in files.items():
            if file_path.exists():
                img = nib.load(file_path)
                vol_data = img.get_fdata()
                data[level] = vol_data
                
                # Calculate statistics
                volume = np.sum(vol_data > 0)
                if volume > 0:
                    # Calculate surface roughness (edge detection)
                    from scipy import ndimage
                    edges = ndimage.sobel(vol_data.astype(float))
                    roughness = np.sum(edges > 0.1) / volume if volume > 0 else 0
                else:
                    roughness = 0
                
                stats[level] = {
                    'volume': volume,
                    'roughness': roughness,
                    'shape': vol_data.shape
                }
                
                logger.debug(f"Loaded {level}: {volume} voxels, roughness: {roughness:.4f}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return data, stats
    
    def create_comparison_visualization(self, patient_id: int, mra_num: int):
        """Create side-by-side comparison visualization"""
        
        data, stats = self.load_smoothing_series(patient_id, mra_num)
        
        if len(data) < 2:
            logger.error(f"Not enough data for patient {patient_id} MRA {mra_num}")
            return
        
        # Get central slice for visualization
        sample_data = list(data.values())[0]
        center_z = sample_data.shape[2] // 2
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Smoothing Progression - Patient {patient_id:02d} MRA{mra_num}', fontsize=16, fontweight='bold')
        
        # Define levels and their positions
        levels = ['original', 'basic_smoothed', 'moderate', 'aggressive', 'ultra']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        
        for i, level in enumerate(levels):
            if level in data and i < len(positions):
                row, col = positions[i]
                ax = axes[row, col]
                
                # Show central slice
                slice_data = data[level][:, :, center_z]
                im = ax.imshow(slice_data.T, cmap='gray', origin='lower')
                
                # Add title with statistics
                vol = stats[level]['volume']
                rough = stats[level]['roughness']
                title = f'{level.replace("_", " ").title()}\n{vol:,} voxels\nRoughness: {rough:.4f}'
                ax.set_title(title, fontsize=12)
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplot
        if len(positions) > len(levels):
            axes[1, 2].axis('off')
        
        # Add statistics table
        stats_ax = axes[1, 2]
        stats_ax.axis('off')
        
        # Create statistics text
        stats_text = "Volume Statistics:\n\n"
        if 'original' in stats:
            orig_vol = stats['original']['volume']
            stats_text += f"Original: {orig_vol:,}\n"
            
            for level in ['basic_smoothed', 'moderate', 'aggressive', 'ultra']:
                if level in stats:
                    vol = stats[level]['volume']
                    change = (vol - orig_vol) / orig_vol * 100 if orig_vol > 0 else 0
                    level_name = level.replace('_', ' ').title()
                    stats_text += f"{level_name}: {vol:,} ({change:+.1f}%)\n"
        
        stats_text += f"\nRoughness Reduction:\n"
        if 'original' in stats:
            orig_rough = stats['original']['roughness']
            for level in ['basic_smoothed', 'moderate', 'aggressive', 'ultra']:
                if level in stats:
                    rough = stats[level]['roughness']
                    reduction = (orig_rough - rough) / orig_rough * 100 if orig_rough > 0 else 0
                    level_name = level.replace('_', ' ').title()
                    stats_text += f"{level_name}: {reduction:.1f}% smoother\n"
        
        stats_ax.text(0.1, 0.9, stats_text, transform=stats_ax.transAxes, 
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"smoothing_comparison_patient_{patient_id:02d}_MRA{mra_num}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created comparison: {output_file}")
        
        return stats
    
    def create_summary_report(self):
        """Create comprehensive summary of all smoothing levels"""
        
        logger.info("Creating comprehensive smoothing summary...")
        
        # Test with a few representative patients
        test_cases = [(1, 1), (1, 2), (10, 1), (23, 1), (56, 2)]
        
        all_stats = []
        
        for patient_id, mra_num in test_cases:
            try:
                stats = self.create_comparison_visualization(patient_id, mra_num)
                if stats:
                    all_stats.append((patient_id, mra_num, stats))
            except Exception as e:
                logger.error(f"Error processing patient {patient_id} MRA {mra_num}: {e}")
        
        # Create overall summary
        if all_stats:
            self.create_overall_summary(all_stats)
        
        return len(all_stats)
    
    def create_overall_summary(self, all_stats):
        """Create an overall summary chart"""
        
        # Calculate average statistics across all test cases
        levels = ['original', 'basic_smoothed', 'moderate', 'aggressive', 'ultra']
        avg_volumes = {}
        avg_roughness = {}
        
        for level in levels:
            volumes = []
            roughness = []
            
            for patient_id, mra_num, stats in all_stats:
                if level in stats:
                    volumes.append(stats[level]['volume'])
                    roughness.append(stats[level]['roughness'])
            
            if volumes:
                avg_volumes[level] = np.mean(volumes)
                avg_roughness[level] = np.mean(roughness)
        
        # Create summary plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Volume progression
        if avg_volumes:
            levels_present = list(avg_volumes.keys())
            volumes = list(avg_volumes.values())
            
            ax1.bar(range(len(levels_present)), volumes, color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'])
            ax1.set_xlabel('Smoothing Level')
            ax1.set_ylabel('Average Volume (voxels)')
            ax1.set_title('Volume Progression Across Smoothing Levels')
            ax1.set_xticks(range(len(levels_present)))
            ax1.set_xticklabels([l.replace('_', '\n') for l in levels_present], rotation=45)
            
            # Add value labels
            for i, v in enumerate(volumes):
                ax1.text(i, v + max(volumes)*0.01, f'{int(v):,}', ha='center', va='bottom')
        
        # Roughness reduction
        if avg_roughness:
            levels_present = list(avg_roughness.keys())
            roughness = list(avg_roughness.values())
            
            ax2.plot(range(len(levels_present)), roughness, marker='o', linewidth=3, markersize=8, color='blue')
            ax2.set_xlabel('Smoothing Level')
            ax2.set_ylabel('Average Roughness')
            ax2.set_title('Surface Roughness Reduction')
            ax2.set_xticks(range(len(levels_present)))
            ax2.set_xticklabels([l.replace('_', '\n') for l in levels_present], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, r in enumerate(roughness):
                ax2.text(i, r + max(roughness)*0.02, f'{r:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save summary
        summary_file = self.output_dir / "smoothing_levels_summary.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created overall summary: {summary_file}")

def main():
    print("=== Comparing Smoothing Levels ===")
    
    comparator = SmoothingComparator()
    
    # Create comparison visualizations
    num_comparisons = comparator.create_summary_report()
    
    print(f"\n✅ Smoothing comparison complete!")
    print(f"Created {num_comparisons} comparison visualizations")
    print(f"Output directory: {comparator.output_dir}")
    print("\nFiles created:")
    print("- Individual patient comparisons (PNG)")
    print("- Overall smoothing summary chart")

if __name__ == "__main__":
    main() 