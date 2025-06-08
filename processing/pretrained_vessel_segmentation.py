#!/usr/bin/env python3
"""
Pre-trained Vessel Segmentation Models Integration

This module provides wrappers for various pre-trained vessel segmentation models
to improve kissing artifact detection and vessel segmentation quality.

Available models:
1. VesselFM - Foundation model for universal 3D vessel segmentation
2. DeepVesselNet - CNN-based vessel segmentation
3. Retina vessel models (can be adapted for brain vessels)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom, gaussian_filter
import requests
import os
from tqdm import tqdm


class PretrainedVesselSegmentation:
    """
    Wrapper for various pre-trained vessel segmentation models
    """
    
    def __init__(self, model_name: str = 'unet', device: str = 'auto'):
        """
        Initialize pre-trained model
        
        Parameters:
        -----------
        model_name : str
            Model to use: 'vesselfm', 'deepvesselnet', 'unet', 'retina_unet'
        device : str
            Device to run on: 'auto', 'cuda', 'cpu'
        """
        self.model_name = model_name.lower()
        self.device = self._setup_device(device)
        self.logger = self._setup_logger()
        
        # Model will be loaded on demand
        self.model = None
        self.model_loaded = False
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('PretrainedVessel')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self):
        """Load the specified pre-trained model"""
        if self.model_loaded:
            return
        
        self.logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == 'vesselfm':
            self.model = self._load_vesselfm()
        elif self.model_name == 'deepvesselnet':
            self.model = self._load_deepvesselnet()
        elif self.model_name == 'unet':
            self.model = self._load_simple_unet()
        elif self.model_name == 'retina_unet':
            self.model = self._load_retina_unet()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_vesselfm(self):
        """
        Load VesselFM foundation model
        Note: This is a placeholder - actual implementation would require
        downloading and loading the actual VesselFM weights
        """
        self.logger.warning("VesselFM integration not yet implemented - using fallback U-Net")
        # In actual implementation, you would:
        # 1. Download VesselFM weights
        # 2. Load the model architecture
        # 3. Load pre-trained weights
        return self._load_simple_unet()
    
    def _load_deepvesselnet(self):
        """
        Load DeepVesselNet model
        """
        # Simplified DeepVesselNet architecture
        class DeepVesselNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._conv_block(1, 32)
                self.enc2 = self._conv_block(32, 64)
                self.enc3 = self._conv_block(64, 128)
                
                # Decoder
                self.dec3 = self._conv_block(128, 64)
                self.dec2 = self._conv_block(128, 32)  # 64 + 64 from skip
                self.dec1 = self._conv_block(64, 32)   # 32 + 32 from skip
                
                self.final = nn.Conv3d(32, 1, 1)
                
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(F.max_pool3d(e1, 2))
                e3 = self.enc3(F.max_pool3d(e2, 2))
                
                # Decoder
                d3 = self.dec3(e3)
                d3_up = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=False)
                d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
                d2_up = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=False)
                d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
                
                return torch.sigmoid(self.final(d1))
        
        return DeepVesselNet()
    
    def _load_simple_unet(self):
        """
        Load a simple 3D U-Net for vessel segmentation
        """
        class SimpleUNet3D(nn.Module):
            def __init__(self, in_channels=1, out_channels=1):
                super().__init__()
                
                # Encoder
                self.enc1 = self._double_conv(in_channels, 32)
                self.enc2 = self._double_conv(32, 64)
                self.enc3 = self._double_conv(64, 128)
                self.enc4 = self._double_conv(128, 256)
                
                # Decoder
                self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
                self.dec3 = self._double_conv(256, 128)
                self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
                self.dec2 = self._double_conv(128, 64)
                self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
                self.dec1 = self._double_conv(64, 32)
                
                self.final = nn.Conv3d(32, out_channels, 1)
                
            def _double_conv(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(F.max_pool3d(e1, 2))
                e3 = self.enc3(F.max_pool3d(e2, 2))
                e4 = self.enc4(F.max_pool3d(e3, 2))
                
                # Decoder
                d3 = self.up3(e4)
                d3 = torch.cat([d3, e3], dim=1)
                d3 = self.dec3(d3)
                
                d2 = self.up2(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d1 = self.up1(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                
                return torch.sigmoid(self.final(d1))
        
        return SimpleUNet3D()
    
    def _load_retina_unet(self):
        """
        Load a 2D U-Net trained on retinal vessels (adapted for 3D)
        """
        # For now, use the simple U-Net
        # In practice, you would load actual retina-trained weights
        self.logger.info("Loading retina-trained U-Net (using simple U-Net as placeholder)")
        return self._load_simple_unet()
    
    def download_pretrained_weights(self, model_name: str, save_dir: str = './weights'):
        """
        Download pre-trained weights for the specified model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # URLs for pre-trained weights (these are placeholders)
        weight_urls = {
            'vesselfm': 'https://example.com/vesselfm_weights.pth',
            'deepvesselnet': 'https://example.com/deepvesselnet_weights.pth',
            'retina_unet': 'https://example.com/retina_unet_weights.pth'
        }
        
        if model_name not in weight_urls:
            self.logger.warning(f"No pre-trained weights URL for {model_name}")
            return None
        
        weight_path = os.path.join(save_dir, f"{model_name}_weights.pth")
        
        if os.path.exists(weight_path):
            self.logger.info(f"Weights already downloaded: {weight_path}")
            return weight_path
        
        # Download weights
        self.logger.info(f"Downloading {model_name} weights...")
        try:
            response = requests.get(weight_urls[model_name], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(weight_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"Weights downloaded: {weight_path}")
            return weight_path
            
        except Exception as e:
            self.logger.error(f"Failed to download weights: {e}")
            return None
    
    def segment_vessels(self, input_nifti: str, output_nifti: str,
                       threshold: float = 0.5,
                       patch_size: int = 64,
                       overlap: int = 16,
                       enhance_thin: bool = True) -> Dict:
        """
        Segment vessels using the pre-trained model
        
        Parameters:
        -----------
        input_nifti : str
            Input NIfTI file
        output_nifti : str
            Output segmentation file
        threshold : float
            Probability threshold for vessel classification
        patch_size : int
            Size of patches for processing (to handle memory)
        overlap : int
            Overlap between patches
        enhance_thin : bool
            Apply thin vessel enhancement post-processing
        
        Returns:
        --------
        dict : Segmentation statistics
        """
        # Load model if not already loaded
        if not self.model_loaded:
            self.load_model()
        
        self.logger.info(f"Segmenting vessels in {input_nifti}")
        
        # Load input
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        
        # Normalize volume
        volume = self._normalize_volume(volume)
        
        # Segment using patches
        segmentation = self._segment_with_patches(
            volume, patch_size, overlap
        )
        
        # Post-processing
        if enhance_thin:
            segmentation = self._enhance_thin_vessels(segmentation)
        
        # Apply threshold
        binary_seg = segmentation > threshold
        
        # Remove small components
        binary_seg = self._remove_small_components(binary_seg, min_size=100)
        
        # Save result
        seg_nifti = nib.Nifti1Image(binary_seg.astype(np.float32), affine)
        nib.save(seg_nifti, output_nifti)
        
        # Compute statistics
        stats = {
            'input_shape': volume.shape,
            'vessel_voxels': int(binary_seg.sum()),
            'vessel_percentage': 100 * binary_seg.sum() / binary_seg.size,
            'threshold_used': threshold
        }
        
        self.logger.info(f"Segmentation complete: {stats['vessel_voxels']} vessel voxels "
                        f"({stats['vessel_percentage']:.2f}%)")
        
        return stats
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range"""
        # Clip extreme values
        p1, p99 = np.percentile(volume[volume > 0], [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize
        volume = (volume - p1) / (p99 - p1)
        
        return volume
    
    def _segment_with_patches(self, volume: np.ndarray, 
                            patch_size: int, overlap: int) -> np.ndarray:
        """
        Segment volume using overlapping patches
        """
        self.logger.info(f"Processing with patches of size {patch_size}")
        
        # Pad volume to handle edges
        pad_width = patch_size // 2
        padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        
        # Initialize output
        output_volume = np.zeros_like(padded_volume)
        weight_volume = np.zeros_like(padded_volume)
        
        # Create patches
        stride = patch_size - overlap
        patches_processed = 0
        
        with torch.no_grad():
            for z in range(0, padded_volume.shape[0] - patch_size + 1, stride):
                for y in range(0, padded_volume.shape[1] - patch_size + 1, stride):
                    for x in range(0, padded_volume.shape[2] - patch_size + 1, stride):
                        # Extract patch
                        patch = padded_volume[
                            z:z+patch_size,
                            y:y+patch_size,
                            x:x+patch_size
                        ]
                        
                        # Convert to tensor
                        patch_tensor = torch.from_numpy(patch).float()
                        patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
                        patch_tensor = patch_tensor.to(self.device)
                        
                        # Predict
                        pred = self.model(patch_tensor)
                        pred = pred.squeeze().cpu().numpy()
                        
                        # Add to output with Gaussian weighting
                        weight = self._gaussian_weight(patch_size)
                        output_volume[
                            z:z+patch_size,
                            y:y+patch_size,
                            x:x+patch_size
                        ] += pred * weight
                        
                        weight_volume[
                            z:z+patch_size,
                            y:y+patch_size,
                            x:x+patch_size
                        ] += weight
                        
                        patches_processed += 1
        
        # Normalize by weights
        output_volume = np.divide(
            output_volume,
            weight_volume,
            where=weight_volume > 0,
            out=np.zeros_like(output_volume)
        )
        
        # Remove padding
        output_volume = output_volume[
            pad_width:-pad_width,
            pad_width:-pad_width,
            pad_width:-pad_width
        ]
        
        self.logger.info(f"Processed {patches_processed} patches")
        
        return output_volume
    
    def _gaussian_weight(self, size: int) -> np.ndarray:
        """Create 3D Gaussian weight for patch blending"""
        sigma = size / 4
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        z = np.arange(size) - size // 2
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        weight = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
        
        return weight / weight.max()
    
    def _enhance_thin_vessels(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Enhance thin vessels using Frangi filter
        """
        from skimage.filters import frangi
        
        # Apply Frangi filter
        enhanced = frangi(
            segmentation,
            sigmas=range(1, 4),
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=False
        )
        
        # Combine with original
        combined = np.maximum(segmentation, enhanced)
        
        return combined
    
    def _remove_small_components(self, binary_mask: np.ndarray, 
                               min_size: int = 100) -> np.ndarray:
        """Remove small connected components"""
        from scipy.ndimage import label
        
        labeled, num_features = label(binary_mask)
        
        # Count component sizes
        component_sizes = np.bincount(labeled.ravel())
        
        # Keep only large components
        large_components = component_sizes > min_size
        large_components[0] = False  # Background
        
        return large_components[labeled]
    
    def refine_segmentation(self, input_nifti: str, 
                          initial_seg_nifti: str,
                          output_nifti: str,
                          iterations: int = 3) -> Dict:
        """
        Refine an existing segmentation using the pre-trained model
        
        Parameters:
        -----------
        input_nifti : str
            Original intensity image
        initial_seg_nifti : str
            Initial segmentation to refine
        output_nifti : str
            Output refined segmentation
        iterations : int
            Number of refinement iterations
        
        Returns:
        --------
        dict : Refinement statistics
        """
        # Load model if not already loaded
        if not self.model_loaded:
            self.load_model()
        
        self.logger.info(f"Refining segmentation with {iterations} iterations")
        
        # Load data
        nifti = nib.load(input_nifti)
        volume = nifti.get_fdata()
        affine = nifti.affine
        
        seg_nifti = nib.load(initial_seg_nifti)
        segmentation = seg_nifti.get_fdata()
        
        # Normalize
        volume = self._normalize_volume(volume)
        
        # Iterative refinement
        for i in range(iterations):
            self.logger.info(f"Refinement iteration {i+1}/{iterations}")
            
            # Use current segmentation as additional input channel
            combined_input = np.stack([volume, segmentation], axis=0)
            
            # Process
            with torch.no_grad():
                input_tensor = torch.from_numpy(combined_input).float()
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # For single channel models, use only intensity
                if self.model.enc1.in_channels == 1:
                    input_tensor = input_tensor[:, 0:1, :, :, :]
                
                output = self.model(input_tensor)
                segmentation = output.squeeze().cpu().numpy()
        
        # Final threshold
        binary_seg = segmentation > 0.5
        
        # Save
        refined_nifti = nib.Nifti1Image(binary_seg.astype(np.float32), affine)
        nib.save(refined_nifti, output_nifti)
        
        # Statistics
        initial_voxels = (seg_nifti.get_fdata() > 0.5).sum()
        refined_voxels = binary_seg.sum()
        
        stats = {
            'initial_voxels': int(initial_voxels),
            'refined_voxels': int(refined_voxels),
            'voxel_change': int(refined_voxels - initial_voxels),
            'percent_change': 100 * (refined_voxels - initial_voxels) / initial_voxels
        }
        
        self.logger.info(f"Refinement complete: {stats['voxel_change']:+d} voxels "
                        f"({stats['percent_change']:+.1f}% change)")
        
        return stats


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pre-trained vessel segmentation"
    )
    parser.add_argument('input_nifti', help='Input NIfTI file')
    parser.add_argument('output_nifti', help='Output segmentation')
    parser.add_argument('--model', default='unet',
                       choices=['vesselfm', 'deepvesselnet', 'unet', 'retina_unet'],
                       help='Model to use')
    parser.add_argument('--device', default='auto',
                       help='Device: auto, cuda, cpu')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Vessel probability threshold')
    parser.add_argument('--refine', help='Refine existing segmentation')
    parser.add_argument('--download-weights', action='store_true',
                       help='Download pre-trained weights')
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = PretrainedVesselSegmentation(
        model_name=args.model,
        device=args.device
    )
    
    if args.download_weights:
        # Download weights
        weight_path = segmenter.download_pretrained_weights(args.model)
        print(f"Weights saved to: {weight_path}")
        return
    
    if args.refine:
        # Refine existing segmentation
        stats = segmenter.refine_segmentation(
            args.input_nifti,
            args.refine,
            args.output_nifti
        )
    else:
        # Segment from scratch
        stats = segmenter.segment_vessels(
            args.input_nifti,
            args.output_nifti,
            threshold=args.threshold
        )
    
    print("\nSegmentation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 