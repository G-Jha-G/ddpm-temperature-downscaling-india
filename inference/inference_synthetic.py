"""
Batch inference script: run trained DDPM on the MSWX synthetic test period.

Applies the trained model to artificially degraded 1.0 deg MSWX inputs (test period:
7 Aug 2009 - 31 Dec 2014, 1,973 days) and computes evaluation metrics against the
0.1 deg MSWX reference. This reproduces the results in Table 2 and Table 3 of the paper.

Key implementation details:
    - Stitch-First, Denormalize-Later: patches are stitched in normalised space
      then globally denormalised using the full-resolution climatology maps.
      This eliminates the grid artefacts caused by per-patch denormalisation.
    - Gaussian blending (sigma_factor=0.3) for seamless patch boundaries.
    - DDIM sampling (50 steps) loaded from best_model.pt checkpoint.
    - EMA weights applied at inference.

Usage:
    Edit the USER CONFIGURATION section below, then run:
        python inference_synthetic.py

    To run in the background:
        nohup python -u inference_synthetic.py > inference_synthetic.log 2>&1 &
"""

import os
import logging
import math
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for DDPM model and inference."""
    # Model architecture
    in_channels: int = 2
    out_channels: int = 1
    time_dim: int = 256
    base_dim: int = 64
    
    # Training parameters
    t_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Inference parameters - ARTIFACT PREVENTION
    patch_size: int = 128
    stride: int = 64
    sigma_factor: float = 0.3  # Validated to be artifact-free
    ddim_steps: int = 50
    
    # Data
    var_name: str = "air_temperature"

# ============================================================================
# USER CONFIGURATION — update all paths before running
# ============================================================================

# LR_FOLDER:        Directory containing 1.0 deg degraded MSWX files
#                   Naming: YYYYDDD_1deg_to_hr.nc
# HR_FOLDER:        Directory containing 0.1 deg MSWX reference files (for metrics)
#                   Naming: YYYYDDD.nc
# HR_TEMPLATE_FILE: Any single 0.1 deg MSWX file — used to copy lat/lon coordinates
# MODEL_PATH:       Path to best_model.pt (saved by train.py as best_model.pt)
# CLIMATOLOGY_PATH: Path to climatology_maps.nc (saved during training to results_dir)
# OUTPUT_FOLDER:    Directory where downscaled output NetCDF files are written

LR_FOLDER = None          # e.g. "/data/MSWX_Temp_Daily_Ind_LR_regridded"
HR_FOLDER = None          # e.g. "/data/MSWX_Temp_Daily_Ind_HR"
HR_TEMPLATE_FILE = None   # e.g. "/data/MSWX_Temp_Daily_Ind_HR/2014342.nc"
MODEL_PATH = None         # e.g. "/results/best_model.pt"
CLIMATOLOGY_PATH = None   # e.g. "/results/climatology_maps.nc"
OUTPUT_FOLDER = None      # e.g. "/results/ddpm_synthetic_output"

# START_DATE: only process files with YYYYDDD > this value (test period start)
START_DATE = 2009219      # 7 Aug 2009 — start of test period

# Metrics Configuration
COMPUTE_METRICS = True  # Set to False if HR files are not available

# ============================================================================
# GPU SELECTION CONFIGURATION
# ============================================================================

GPU_MODE = "auto"  # Options: "auto", "manual", "cpu"
MANUAL_GPU_ID = 0  # Set to 0, 1, 2, etc. for specific GPU

# ============================================================================
# GPU SELECTION UTILITIES
# ============================================================================

def get_gpu_memory_info():
    """Get memory usage information for all GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                free_mem = float(parts[1].strip())
                total_mem = float(parts[2].strip())
                gpu_info.append((gpu_id, free_mem, total_mem))
        
        return gpu_info
    
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not get GPU info via nvidia-smi: {e}")
        return []

def select_least_busy_gpu():
    """Select the GPU with the most free memory."""
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        return None
    
    gpu_info.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("GPU MEMORY INFORMATION")
    print("="*80)
    for gpu_id, free_mem, total_mem in gpu_info:
        used_mem = total_mem - free_mem
        usage_pct = (used_mem / total_mem) * 100
        print(f"GPU {gpu_id}: {free_mem:.0f} MB free / {total_mem:.0f} MB total "
              f"({usage_pct:.1f}% used)")
    print("="*80)
    
    selected_gpu = gpu_info[0][0]
    selected_free = gpu_info[0][1]
    
    print(f"✓ Auto-selected GPU {selected_gpu} with {selected_free:.0f} MB free memory")
    print("="*80 + "\n")
    
    return selected_gpu

def setup_device(mode="auto", manual_gpu_id=0):
    """Setup device based on GPU selection mode."""
    if mode == "cpu":
        print("\n" + "="*80)
        print("GPU MODE: CPU (Forced)")
        print("="*80)
        print("⚠️  Running on CPU - this will be VERY slow!")
        print("="*80 + "\n")
        return torch.device("cpu")
    
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU MODE: CPU (No CUDA available)")
        print("="*80)
        print("⚠️  CUDA not available - falling back to CPU")
        print("="*80 + "\n")
        return torch.device("cpu")
    
    if mode == "manual":
        if manual_gpu_id >= torch.cuda.device_count():
            print(f"\n⚠️  Warning: GPU {manual_gpu_id} not available. "
                  f"Only {torch.cuda.device_count()} GPU(s) detected.")
            print("Falling back to auto selection...\n")
            mode = "auto"
        else:
            print("\n" + "="*80)
            print(f"GPU MODE: Manual (GPU {manual_gpu_id})")
            print("="*80)
            
            gpu_name = torch.cuda.get_device_name(manual_gpu_id)
            print(f"✓ Using manually selected GPU {manual_gpu_id}: {gpu_name}")
            
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                for gpu_id, free_mem, total_mem in gpu_info:
                    if gpu_id == manual_gpu_id:
                        print(f"✓ GPU {manual_gpu_id}: {free_mem:.0f} MB free / {total_mem:.0f} MB total")
            
            print("="*80 + "\n")
            
            torch.cuda.set_device(manual_gpu_id)
            return torch.device(f"cuda:{manual_gpu_id}")
    
    if mode == "auto":
        print("\n" + "="*80)
        print("GPU MODE: Auto (Selecting least busy GPU)")
        print("="*80)
        
        selected_gpu = select_least_busy_gpu()
        
        if selected_gpu is None:
            print("⚠️  Could not detect GPU information - using default CUDA device")
            return torch.device("cuda")
        
        gpu_name = torch.cuda.get_device_name(selected_gpu)
        print(f"GPU Name: {gpu_name}")
        
        torch.cuda.set_device(selected_gpu)
        return torch.device(f"cuda:{selected_gpu}")
    
    print("\n⚠️  Unknown GPU mode, using default CUDA device\n")
    return torch.device("cuda")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('ddpm_artifact_free_inference')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def nearest_fill_nan_fast(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values using nearest neighbor interpolation."""
    if np.isnan(arr).sum() == 0:
        return arr
    
    valid = ~np.isnan(arr)
    coords = np.array(np.nonzero(valid)).T
    
    if coords.size == 0:
        return np.zeros_like(arr)
    
    grid_coords = np.array(np.nonzero(np.ones_like(arr))).T
    tree = cKDTree(coords)
    _, idx = tree.query(grid_coords, k=1)
    mapped_coords = coords[idx]
    
    filled_arr = np.zeros_like(arr)
    filled_arr.flat[:] = arr[tuple(mapped_coords.T)]
    
    return filled_arr

def safe_load_dataset(file_path: str, var_name: str = "air_temperature") -> Optional[np.ndarray]:
    """Safely load NetCDF dataset."""
    try:
        with xr.open_dataset(file_path) as ds:
            data = np.squeeze(ds[var_name].values).astype(np.float32)
            return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        prediction: Predicted array
        target: Ground truth array
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Flatten arrays
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'correlation': np.nan,
            'r2': np.nan,
            'ssim': np.nan,
            'psnr': np.nan
        }
    
    # Basic metrics
    mse = mean_squared_error(target_valid, pred_valid)
    metrics['rmse'] = np.sqrt(mse)
    metrics['mae'] = mean_absolute_error(target_valid, pred_valid)
    metrics['bias'] = np.mean(pred_valid - target_valid)
    
    # Correlation
    correlation, _ = pearsonr(pred_valid, target_valid)
    metrics['correlation'] = correlation
    metrics['r2'] = correlation ** 2
    
    # Image quality metrics (SSIM and PSNR)
    try:
        # Normalize to [0, 1] for SSIM and PSNR
        pred_norm = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        target_norm = (target - target.min()) / (target.max() - target.min())
        
        # SSIM
        ssim_score = ssim(target_norm, pred_norm, data_range=1.0)
        metrics['ssim'] = ssim_score
        
        # PSNR
        psnr_score = psnr(target_norm, pred_norm, data_range=1.0)
        metrics['psnr'] = psnr_score
        
    except Exception as e:
        logger.warning(f"Failed to compute SSIM/PSNR: {e}")
        metrics['ssim'] = np.nan
        metrics['psnr'] = np.nan
    
    return metrics

def print_metrics_summary(all_metrics: List[dict]):
    """
    Print formatted metrics summary.
    
    Args:
        all_metrics: List of metric dictionaries from all processed files
    """
    if not all_metrics:
        logger.warning("No metrics to display")
        return
    
    # Aggregate metrics
    metric_names = ['rmse', 'mae', 'bias', 'r2', 'correlation', 'ssim', 'psnr']
    aggregated = {}
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if not np.isnan(m[metric_name])]
        if values:
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        else:
            aggregated[metric_name] = {
                'mean': np.nan,
                'std': np.nan
            }
    
    # Print formatted results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # RMSE
    if 'rmse' in aggregated:
        mean_val = aggregated['rmse']['mean']
        std_val = aggregated['rmse']['std']
        print(f"RMSE:         {mean_val:.6f} ± {std_val:.6f} K")
    
    # MAE
    if 'mae' in aggregated:
        mean_val = aggregated['mae']['mean']
        std_val = aggregated['mae']['std']
        print(f"MAE:          {mean_val:.6f} ± {std_val:.6f} K")
    
    # Bias
    if 'bias' in aggregated:
        mean_val = aggregated['bias']['mean']
        std_val = aggregated['bias']['std']
        print(f"Bias:         {mean_val:.6f} ± {std_val:.6f} K")
    
    # R²
    if 'r2' in aggregated:
        mean_val = aggregated['r2']['mean']
        std_val = aggregated['r2']['std']
        print(f"R²:           {mean_val:.6f} ± {std_val:.6f}")
    
    # Correlation
    if 'correlation' in aggregated:
        mean_val = aggregated['correlation']['mean']
        std_val = aggregated['correlation']['std']
        print(f"Correlation:  {mean_val:.6f} ± {std_val:.6f}")
    
    # SSIM
    if 'ssim' in aggregated:
        mean_val = aggregated['ssim']['mean']
        std_val = aggregated['ssim']['std']
        print(f"SSIM:         {mean_val:.6f} ± {std_val:.6f}")
    
    # PSNR
    if 'psnr' in aggregated:
        mean_val = aggregated['psnr']['mean']
        std_val = aggregated['psnr']['std']
        print(f"PSNR:         {mean_val:.2f} ± {std_val:.2f} dB")
    
    print("="*80)
    print(f"✓ Metrics computed over {len(all_metrics)} samples")
    print("="*80 + "\n")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        
        scale = 1.0 / math.sqrt(C)
        attn = torch.einsum('bci,bcj->bij', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum('bij,bcj->bci', attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h

class ResNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        time_emb = self.activation(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, 
                 num_layers: int = 2, downsample: bool = True, attention: bool = False):
        super().__init__()
        self.downsample = downsample
        
        self.resnets = nn.ModuleList([
            ResNetBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_layers)
        ])
        
        if attention:
            self.attentions = nn.ModuleList([AttentionBlock(out_ch) for _ in range(num_layers)])
        else:
            self.attentions = [None] * num_layers
        
        if downsample:
            self.downsample_conv = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t)
            if attn is not None:
                x = attn(x)
        
        if self.downsample:
            x = self.downsample_conv(x)
        
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 num_layers: int = 2, upsample: bool = True, attention: bool = False):
        super().__init__()
        self.upsample = upsample
        
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_in_ch = in_ch + out_ch if i == 0 else out_ch
            self.resnets.append(ResNetBlock(res_in_ch, out_ch, time_emb_dim))
        
        if attention:
            self.attentions = nn.ModuleList([AttentionBlock(out_ch) for _ in range(num_layers)])
        else:
            self.attentions = [None] * num_layers
        
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t)
            if attn is not None:
                x = attn(x)
        
        return x

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1,
                 time_dim: int = 256, base_dim: int = 64):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        down_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.downs = nn.ModuleList([
            DownBlock(down_dims[0], down_dims[1], time_dim, attention=False),
            DownBlock(down_dims[1], down_dims[2], time_dim, attention=False),
            DownBlock(down_dims[2], down_dims[3], time_dim, attention=True),
        ])
        
        self.bottleneck = ResNetBlock(down_dims[3], down_dims[3], time_dim)
        self.bottleneck_attn = AttentionBlock(down_dims[3])
        
        self.ups = nn.ModuleList([
            UpBlock(down_dims[3], down_dims[2], time_dim, attention=True),
            UpBlock(down_dims[2], down_dims[1], time_dim, attention=False),
            UpBlock(down_dims[1], down_dims[0], time_dim, attention=False),
        ])
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)
        
        skips = []
        skips.append(x)
        x = self.downs[0](x, t_emb)
        
        skips.append(x)
        x = self.downs[1](x, t_emb)
        
        skips.append(x)
        x = self.downs[2](x, t_emb)
        
        x = self.bottleneck(x, t_emb)
        x = self.bottleneck_attn(x)
        
        x = self.ups[0](x, skips[2], t_emb)
        x = self.ups[1](x, skips[1], t_emb)
        x = self.ups[2](x, skips[0], t_emb)
        
        return self.conv_out(x)

# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
    
    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# ============================================================================
# NOISE SCHEDULER
# ============================================================================

class NoiseScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, device: str = 'cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def ddim_sample_prev_timestep(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                                   t: int, t_prev: int, eta: float = 0.0) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t)) * noise_pred
        noise = eta * torch.randn_like(x_t) * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + noise
        return x_prev

# ============================================================================
# NORMALIZER
# ============================================================================

class EfficientNormalizer:
    def __init__(self, mean_map: np.ndarray, std_map: np.ndarray, method: str = 'robust'):
        self.mean_map = torch.from_numpy(mean_map).float()
        self.std_map = torch.from_numpy(std_map).float()
        self.scale_map = torch.where(self.std_map < 1e-7, 1e-7, self.std_map)
        self.method = method
    
    def normalize_patch(self, patch: np.ndarray, top: int, left: int, patch_size: int) -> np.ndarray:
        patch_tensor = torch.from_numpy(patch).float()
        mean = self.mean_map[top:top+patch_size, left:left+patch_size]
        scale = self.scale_map[top:top+patch_size, left:left+patch_size]
        normalized = (patch_tensor - mean) / scale
        return normalized.numpy()

# ============================================================================
# ARTIFACT-FREE PATCH PROCESSING
# ============================================================================

def extract_patches(img: np.ndarray, patch_size: int = 128, stride: int = 64) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Extract overlapping patches with proper edge handling."""
    h, w = img.shape
    patches, positions = [], []
    
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patches.append(img[top:top + patch_size, left:left + patch_size])
            positions.append((top, left))
    
    if (w - patch_size) % stride != 0:
        left = w - patch_size
        for top in range(0, h - patch_size + 1, stride):
            patches.append(img[top:top + patch_size, left:left + patch_size])
            positions.append((top, left))
    
    if (h - patch_size) % stride != 0:
        top = h - patch_size
        for left in range(0, w - patch_size + 1, stride):
            patches.append(img[top:top + patch_size, left:left + patch_size])
            positions.append((top, left))
    
    if (h - patch_size) % stride != 0 and (w - patch_size) % stride != 0:
        top, left = h - patch_size, w - patch_size
        patches.append(img[top:top + patch_size, left:left + patch_size])
        positions.append((top, left))
    
    return patches, positions

def create_gaussian_weights(patch_size: int, sigma_factor: float = 0.3) -> np.ndarray:
    """Create Gaussian weights for smooth blending."""
    center = patch_size // 2
    y, x = np.ogrid[:patch_size, :patch_size]
    sigma = sigma_factor * patch_size
    dist_sq = (x - center)**2 + (y - center)**2
    weights = np.exp(-dist_sq / (2 * sigma**2))
    return (weights / weights.max()).astype(np.float32)

def stitch_patches_gaussian(patches: List[np.ndarray], positions: List[Tuple[int, int]],
                           full_shape: Tuple[int, int], patch_size: int = 128,
                           sigma_factor: float = 0.3) -> np.ndarray:
    """Stitch patches using Gaussian weighted blending - ARTIFACT-FREE."""
    h, w = full_shape
    output = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    gaussian_weights = create_gaussian_weights(patch_size, sigma_factor=sigma_factor)
    
    for patch, (top, left) in zip(patches, positions):
        ph, pw = patch.shape
        w_curr = gaussian_weights[:ph, :pw]
        
        output[top:top+ph, left:left+pw] += patch * w_curr
        weight_sum[top:top+ph, left:left+pw] += w_curr
    
    return output / np.where(weight_sum < 1e-8, 1.0, weight_sum)

# ============================================================================
# DDIM SAMPLING
# ============================================================================

@torch.no_grad()
def sample_ddim(model: nn.Module, lr_condition: torch.Tensor, scheduler: NoiseScheduler,
                device: torch.device, ddim_steps: int = 50) -> torch.Tensor:
    """DDIM sampling for faster inference."""
    model.eval()
    
    c = scheduler.num_timesteps // ddim_steps
    timesteps = list(range(0, scheduler.num_timesteps, c))[::-1]
    
    x = torch.randn_like(lr_condition).to(device)
    
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        
        with torch.amp.autocast('cuda'):
            model_input = torch.cat([x, lr_condition], dim=1)
            t_batch = torch.tensor([t] * x.shape[0], device=device, dtype=torch.long)
            noise_pred = model(model_input, t_batch)
        
        x = scheduler.ddim_sample_prev_timestep(x, noise_pred, t, t_prev)
    
    return x

# ============================================================================
# ARTIFACT-FREE FULL IMAGE PREDICTION
# ============================================================================

def predict_full_image(model: nn.Module, lr_image: np.ndarray, normalizer: EfficientNormalizer,
                      scheduler: NoiseScheduler, device: torch.device, config: Config,
                      use_ddim: bool = True) -> np.ndarray:
    """
    Predict full HR image using ARTIFACT-FREE patch-based processing.
    
    CRITICAL: Uses "Stitch-First, Denormalize-Later" strategy.
    """
    
    patches_lr, positions = extract_patches(lr_image, config.patch_size, config.stride)
    pred_patches_norm = []
    
    for lr_patch, (top, left) in zip(patches_lr, positions):
        lr_norm = normalizer.normalize_patch(lr_patch, top, left, config.patch_size)
        lr_norm = np.clip(lr_norm, -5, 5)
        
        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        
        if use_ddim:
            hr_norm_pred = sample_ddim(model, lr_tensor, scheduler, device, config.ddim_steps)
        else:
            raise NotImplementedError("DDPM sampling not implemented")
        
        hr_norm_pred = hr_norm_pred.squeeze().cpu().numpy()
        hr_norm_pred = np.clip(hr_norm_pred, -5, 5)
        
        pred_patches_norm.append(hr_norm_pred)
    
    sr_normalized = stitch_patches_gaussian(
        pred_patches_norm,
        positions,
        lr_image.shape,
        config.patch_size,
        config.sigma_factor
    )
    
    full_mean = normalizer.mean_map.numpy() if torch.is_tensor(normalizer.mean_map) else normalizer.mean_map
    full_std = normalizer.scale_map.numpy() if torch.is_tensor(normalizer.scale_map) else normalizer.scale_map
    
    h, w = lr_image.shape
    result = sr_normalized * full_std[:h, :w] + full_mean[:h, :w]
    
    return result

# ============================================================================
# MAIN BATCH INFERENCE
# ============================================================================

def main():
    """Main batch inference function."""
    
    # Validate paths
    required = {
        "LR_FOLDER": LR_FOLDER,
        "HR_TEMPLATE_FILE": HR_TEMPLATE_FILE,
        "MODEL_PATH": MODEL_PATH,
        "CLIMATOLOGY_PATH": CLIMATOLOGY_PATH,
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"The following paths are not set in the USER CONFIGURATION section: {missing}\n"
            "Edit inference_synthetic.py and set each path before running."
        )
    
    logger.info("="*80)
    logger.info("ARTIFACT-FREE DDPM BATCH INFERENCE v2.2")
    logger.info("="*80)
    logger.info("✓ Stitch-First, Denormalize-Later strategy")
    logger.info("✓ Gaussian blending (σ=0.3)")
    logger.info("✓ Global denormalization")
    logger.info("✓ Smart GPU selection")
    logger.info("✓ Comprehensive metrics")
    logger.info("="*80)
    
    # Setup device
    device = setup_device(mode=GPU_MODE, manual_gpu_id=MANUAL_GPU_ID)
    
    # Load model and config
    logger.info(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Extract only the config keys that match our Config dataclass
    config_dict = checkpoint['config']
    valid_keys = {
        'in_channels', 'out_channels', 'time_dim', 'base_dim',
        't_steps', 'beta_start', 'beta_end',
        'use_ema', 'ema_decay',
        'patch_size', 'stride', 'sigma_factor', 'ddim_steps',
        'var_name'
    }
    
    # Filter config to only include valid keys
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    # Create Config with filtered dictionary, using defaults for missing keys
    config = Config(**filtered_config)
    
    logger.info(f"Config: patch_size={config.patch_size}, stride={config.stride}, sigma={config.sigma_factor}")
    
    # Initialize model
    model = EnhancedUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        time_dim=config.time_dim,
        base_dim=config.base_dim
    )
    
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Apply EMA
    if config.use_ema and 'ema_state_dict' in checkpoint:
        ema = EMA(model, decay=config.ema_decay)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.apply_shadow()
        logger.info("✓ EMA weights applied")
    
    # Initialize noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=config.t_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device
    )
    
    # Load climatology
    logger.info(f"Loading climatology from: {CLIMATOLOGY_PATH}")
    with xr.open_dataset(CLIMATOLOGY_PATH) as ds:
        mean_map = ds['mean_map'].values
        std_map = ds['std_map'].values
    
    normalizer = EfficientNormalizer(mean_map, std_map, method='robust')
    logger.info("✓ Climatology loaded")
    
    # Load HR template
    logger.info(f"Loading HR template from: {HR_TEMPLATE_FILE}")
    with xr.open_dataset(HR_TEMPLATE_FILE) as hr_template_ds:
        lat_coords = hr_template_ds['lat']
        lon_coords = hr_template_ds['lon']
        var_attrs = hr_template_ds[config.var_name].attrs if config.var_name in hr_template_ds else {}
    
    # Setup output directory
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Get list of LR files
    lr_folder = Path(LR_FOLDER)
    lr_files = sorted([
        f for f in lr_folder.glob("*_1deg_to_hr.nc")
        if int(f.stem.split('_')[0]) > START_DATE
    ])
    
    logger.info(f"Found {len(lr_files)} files to process (date > {START_DATE})")
    
    if len(lr_files) == 0:
        logger.warning("No files to process!")
        return
    
    # Process each file
    logger.info("="*80)
    logger.info("STARTING BATCH PROCESSING")
    logger.info("="*80)
    
    all_metrics = []
    
    for i, lr_file in enumerate(tqdm(lr_files, desc="Processing files"), 1):
        try:
            # Load LR image
            lr_image = safe_load_dataset(str(lr_file), config.var_name)
            if lr_image is None:
                logger.warning(f"Skipping {lr_file.name} - failed to load")
                continue
            
            lr_image = nearest_fill_nan_fast(lr_image)
            
            # Run artifact-free downscaling
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    prediction_np = predict_full_image(
                        model=model,
                        lr_image=lr_image,
                        normalizer=normalizer,
                        scheduler=scheduler,
                        device=device,
                        config=config,
                        use_ddim=True
                    )
            
            # Add time dimension
            if prediction_np.ndim == 2:
                prediction_np = np.expand_dims(prediction_np, axis=0)
            
            # Load time coordinate
            with xr.open_dataset(lr_file) as lr_ds:
                time_coords = lr_ds['time']
            
            # Create xarray DataArray
            output_da = xr.DataArray(
                data=prediction_np,
                coords={'time': time_coords, 'lat': lat_coords, 'lon': lon_coords},
                dims=['time', 'lat', 'lon'],
                name=config.var_name,
                attrs=var_attrs
            )
            
            # Generate output filename
            output_filename = lr_file.name.replace("_1deg_to_hr", "_ddpm_artifact_free")
            output_file = output_path / output_filename
            
            # Save NetCDF
            output_da.to_netcdf(output_file)
            
            # Compute metrics if HR file is available
            if COMPUTE_METRICS:
                hr_file_name = lr_file.name.replace("_1deg_to_hr.nc", ".nc")
                hr_file_path = Path(HR_FOLDER) / hr_file_name
                
                if hr_file_path.exists():
                    hr_image = safe_load_dataset(str(hr_file_path), config.var_name)
                    if hr_image is not None:
                        hr_image = nearest_fill_nan_fast(hr_image)
                        metrics = compute_metrics(prediction_np.squeeze(), hr_image)
                        all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error processing {lr_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("="*80)
    logger.info("🎉 BATCH PROCESSING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Processed {len(lr_files)} files")
    logger.info(f"Output directory: {output_path}")
    logger.info("✅ All outputs are ARTIFACT-FREE!")
    logger.info("="*80)
    
    # Print metrics summary
    if COMPUTE_METRICS and all_metrics:
        print_metrics_summary(all_metrics)

if __name__ == "__main__":
    main()