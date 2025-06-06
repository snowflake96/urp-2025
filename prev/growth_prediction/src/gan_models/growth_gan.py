"""
3D GAN model for aneurysm growth prediction.
Combines shape and stress field generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
import trimesh
from torch.utils.data import Dataset, DataLoader


class AneurysmGrowthDataset(Dataset):
    """Dataset for time-series aneurysm data."""
    
    def __init__(self, data_dir: str, voxel_size: int = 64):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing processed aneurysm data
            voxel_size: Size of voxel grid for 3D representation
        """
        self.data_dir = data_dir
        self.voxel_size = voxel_size
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load all time-series samples."""
        # This is a placeholder - actual implementation would load real data
        samples = []
        # Load preprocessed voxelized aneurysm data
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Return current state, next state, and time delta
        return {
            'current_shape': sample['current_shape'],
            'current_stress': sample['current_stress'],
            'next_shape': sample['next_shape'],
            'next_stress': sample['next_stress'],
            'time_delta': sample['time_delta'],
            'clinical_features': sample['clinical_features']
        }


class ResidualBlock3D(nn.Module):
    """3D Residual block for generator and discriminator."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class GrowthGenerator(nn.Module):
    """Generator network for aneurysm growth prediction."""
    
    def __init__(self, latent_dim: int = 128, clinical_dim: int = 10,
                 voxel_size: int = 64):
        """
        Initialize generator.
        
        Args:
            latent_dim: Dimension of latent noise vector
            clinical_dim: Dimension of clinical features
            voxel_size: Output voxel grid size
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.clinical_dim = clinical_dim
        self.voxel_size = voxel_size
        
        # Encoder for current state
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 64, 4, 2, 1),  # 2 channels: shape + stress
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            ResidualBlock3D(64, 128, 2),
            ResidualBlock3D(128, 256, 2),
            ResidualBlock3D(256, 512, 2),
        )
        
        # Temporal processor
        self.temporal_fc = nn.Sequential(
            nn.Linear(clinical_dim + 1, 128),  # +1 for time delta
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True)
        )
        
        # Decoder for future state
        self.decoder_input = nn.Linear(512 * 4 * 4 * 4 + 256 + latent_dim, 
                                      512 * 4 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            ResidualBlock3D(256, 256),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            ResidualBlock3D(128, 128),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            ResidualBlock3D(64, 64),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
        )
        
        # Separate heads for shape and stress
        self.shape_head = nn.Conv3d(32, 1, 3, 1, 1)
        self.stress_head = nn.Conv3d(32, 1, 3, 1, 1)
        
    def forward(self, current_shape: torch.Tensor, current_stress: torch.Tensor,
                clinical_features: torch.Tensor, time_delta: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate future aneurysm state.
        
        Args:
            current_shape: Current shape voxel grid [B, 1, D, H, W]
            current_stress: Current stress field [B, 1, D, H, W]
            clinical_features: Clinical features [B, clinical_dim]
            time_delta: Time difference to predict [B, 1]
            noise: Optional noise vector [B, latent_dim]
            
        Returns:
            Tuple of (predicted_shape, predicted_stress)
        """
        batch_size = current_shape.size(0)
        
        # Concatenate current state
        current_state = torch.cat([current_shape, current_stress], dim=1)
        
        # Encode current state
        encoded = self.encoder(current_state)
        encoded = encoded.view(batch_size, -1)
        
        # Process temporal information
        temporal_input = torch.cat([clinical_features, time_delta], dim=1)
        temporal_features = self.temporal_fc(temporal_input)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=current_shape.device)
            
        # Combine all features
        combined = torch.cat([encoded, temporal_features, noise], dim=1)
        
        # Decode to future state
        decoded = self.decoder_input(combined)
        decoded = decoded.view(batch_size, 512, 4, 4, 4)
        decoded = self.decoder(decoded)
        
        # Generate shape and stress predictions
        predicted_shape = torch.sigmoid(self.shape_head(decoded))
        predicted_stress = self.stress_head(decoded)
        
        return predicted_shape, predicted_stress


class GrowthDiscriminator(nn.Module):
    """Discriminator network for distinguishing real vs generated growth."""
    
    def __init__(self, clinical_dim: int = 10):
        """
        Initialize discriminator.
        
        Args:
            clinical_dim: Dimension of clinical features
        """
        super().__init__()
        self.clinical_dim = clinical_dim
        
        # 3D CNN for processing voxel data
        self.voxel_processor = nn.Sequential(
            nn.Conv3d(4, 64, 4, 2, 1),  # 4 channels: before/after shape/stress
            nn.LeakyReLU(0.2, True),
            ResidualBlock3D(64, 128, 2),
            ResidualBlock3D(128, 256, 2),
            ResidualBlock3D(256, 512, 2),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Clinical feature processor
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim + 1, 128),  # +1 for time delta
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, True)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, before_shape: torch.Tensor, before_stress: torch.Tensor,
                after_shape: torch.Tensor, after_stress: torch.Tensor,
                clinical_features: torch.Tensor, time_delta: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake growth patterns.
        
        Args:
            before_shape: Shape before growth [B, 1, D, H, W]
            before_stress: Stress before growth [B, 1, D, H, W]
            after_shape: Shape after growth [B, 1, D, H, W]
            after_stress: Stress after growth [B, 1, D, H, W]
            clinical_features: Clinical features [B, clinical_dim]
            time_delta: Time difference [B, 1]
            
        Returns:
            Discrimination scores [B, 1]
        """
        # Concatenate all voxel data
        voxel_input = torch.cat([before_shape, before_stress, 
                                after_shape, after_stress], dim=1)
        
        # Process voxel data
        voxel_features = self.voxel_processor(voxel_input)
        voxel_features = voxel_features.view(voxel_features.size(0), -1)
        
        # Process clinical data
        clinical_input = torch.cat([clinical_features, time_delta], dim=1)
        clinical_features = self.clinical_processor(clinical_input)
        
        # Combine and classify
        combined = torch.cat([voxel_features, clinical_features], dim=1)
        output = self.classifier(combined)
        
        return output


class AneurysmGrowthGAN:
    """Complete GAN model for aneurysm growth prediction."""
    
    def __init__(self, device: str = 'cuda', learning_rate: float = 0.0002):
        """
        Initialize the GAN model.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizers
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = GrowthGenerator().to(self.device)
        self.discriminator = GrowthDiscriminator().to(self.device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                           lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                           lr=learning_rate, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of losses
        """
        # Move data to device
        current_shape = batch['current_shape'].to(self.device)
        current_stress = batch['current_stress'].to(self.device)
        next_shape = batch['next_shape'].to(self.device)
        next_stress = batch['next_stress'].to(self.device)
        clinical_features = batch['clinical_features'].to(self.device)
        time_delta = batch['time_delta'].to(self.device)
        
        batch_size = current_shape.size(0)
        
        # Labels for adversarial loss
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_pred = self.discriminator(current_shape, current_stress,
                                      next_shape, next_stress,
                                      clinical_features, time_delta)
        d_real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake data
        with torch.no_grad():
            fake_shape, fake_stress = self.generator(current_shape, current_stress,
                                                    clinical_features, time_delta)
        
        fake_pred = self.discriminator(current_shape, current_stress,
                                      fake_shape, fake_stress,
                                      clinical_features, time_delta)
        d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_shape, fake_stress = self.generator(current_shape, current_stress,
                                                clinical_features, time_delta)
        
        # Adversarial loss
        fake_pred = self.discriminator(current_shape, current_stress,
                                      fake_shape, fake_stress,
                                      clinical_features, time_delta)
        g_adv_loss = self.adversarial_loss(fake_pred, real_labels)
        
        # Reconstruction loss
        g_recon_loss = self.reconstruction_loss(fake_shape, next_shape) + \
                      self.reconstruction_loss(fake_stress, next_stress)
        
        g_loss = g_adv_loss + 10.0 * g_recon_loss  # Weight reconstruction more
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_recon_loss': g_recon_loss.item()
        }
        
    def predict(self, current_shape: np.ndarray, current_stress: np.ndarray,
                clinical_features: np.ndarray, time_delta: float,
                num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict future aneurysm state with uncertainty.
        
        Args:
            current_shape: Current shape voxel grid
            current_stress: Current stress field
            clinical_features: Clinical features
            time_delta: Time to predict forward
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (mean_shape, mean_stress, uncertainty)
        """
        self.generator.eval()
        
        # Convert to tensors
        current_shape_t = torch.FloatTensor(current_shape).unsqueeze(0).unsqueeze(0)
        current_stress_t = torch.FloatTensor(current_stress).unsqueeze(0).unsqueeze(0)
        clinical_features_t = torch.FloatTensor(clinical_features).unsqueeze(0)
        time_delta_t = torch.FloatTensor([time_delta]).unsqueeze(0)
        
        # Move to device
        current_shape_t = current_shape_t.to(self.device)
        current_stress_t = current_stress_t.to(self.device)
        clinical_features_t = clinical_features_t.to(self.device)
        time_delta_t = time_delta_t.to(self.device)
        
        # Generate multiple samples
        shape_samples = []
        stress_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred_shape, pred_stress = self.generator(
                    current_shape_t, current_stress_t,
                    clinical_features_t, time_delta_t
                )
                shape_samples.append(pred_shape.cpu().numpy())
                stress_samples.append(pred_stress.cpu().numpy())
                
        # Compute statistics
        shape_samples = np.array(shape_samples).squeeze()
        stress_samples = np.array(stress_samples).squeeze()
        
        mean_shape = np.mean(shape_samples, axis=0)
        mean_stress = np.mean(stress_samples, axis=0)
        
        # Uncertainty as standard deviation
        shape_std = np.std(shape_samples, axis=0)
        stress_std = np.std(stress_samples, axis=0)
        uncertainty = np.sqrt(shape_std**2 + stress_std**2)
        
        return mean_shape, mean_stress, uncertainty


def train_growth_gan(data_loader: DataLoader, num_epochs: int = 1000,
                    checkpoint_dir: str = './checkpoints'):
    """
    Train the growth prediction GAN.
    
    Args:
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    gan = AneurysmGrowthGAN()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {'d_loss': 0, 'g_loss': 0, 
                       'g_adv_loss': 0, 'g_recon_loss': 0}
        
        for batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            losses = gan.train_step(batch)
            
            for key, value in losses.items():
                epoch_losses[key] += value
                
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(data_loader)
            
        # Print progress
        print(f"Epoch {epoch+1}: D_loss={epoch_losses['d_loss']:.4f}, "
              f"G_loss={epoch_losses['g_loss']:.4f}, "
              f"G_adv={epoch_losses['g_adv_loss']:.4f}, "
              f"G_recon={epoch_losses['g_recon_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': gan.generator.state_dict(),
                'discriminator_state_dict': gan.discriminator.state_dict(),
                'g_optimizer_state_dict': gan.g_optimizer.state_dict(),
                'd_optimizer_state_dict': gan.d_optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))


if __name__ == "__main__":
    # Example usage
    print("AneurysmGrowthGAN module loaded successfully") 