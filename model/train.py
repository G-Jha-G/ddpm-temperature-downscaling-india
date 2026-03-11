"""
DDPM-based climate downscaling: model architecture, training pipeline, and utilities.

Implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for 10x
spatial super-resolution of daily near-surface temperature over India (1.0 deg -> 0.1 deg).

Architecture:
    - Conditional U-Net with self-attention (channels: [64, 128, 256, 512])
    - Sinusoidal time embeddings
    - Exponential Moving Average (EMA) of weights
    - DDIM sampling (50 steps) for fast inference
    - Gaussian patch stitching (128x128, stride 64) for artefact-free full-domain inference

Usage:
    python train.py --gpu 0 --epochs 100 --batch-size 64
    python train.py --gpu 0 --epochs 100 --resume-checkpoint /path/to/checkpoint.pt

Configuration:
    Edit the USER CONFIGURATION section in Config below, or pass arguments via CLI.
    All paths default to None and must be set before training.
"""

import os
import math
import re
import random
import json
import logging
import warnings
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Union, Any
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import stats, ndimage
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance
import seaborn as sns

# Optional: Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")

warnings.filterwarnings('ignore')

# -------------------------
# Performance Optimizations
# -------------------------
def optimize_pytorch_settings():
    """Set optimal PyTorch environment variables for performance."""
    # Optimize for multi-core CPU
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # Enable tensor core usage (for supported GPUs)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Optimize CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Enable optimized attention (if available)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    print("Applied PyTorch performance optimizations")

# -------------------------
# Type Definitions
# -------------------------
TensorType = Union[torch.Tensor, torch.FloatTensor]
ArrayType = npt.NDArray[np.float32]

# -------------------------
# Optimized Configuration
# -------------------------
@dataclass
class Config:
    """Enhanced configuration with performance optimizations."""
    
    # =========================================================================
    # USER CONFIGURATION — set these paths before running
    # =========================================================================
    # hr_dir:       Directory containing high-resolution MSWX 0.1 deg NetCDF files
    #               File naming: YYYYDDD.nc (e.g. 1979001.nc)
    # lr_dir:       Directory containing spatially degraded 1.0 deg NetCDF files
    #               File naming: YYYYDDD_1deg_to_hr.nc
    # results_dir:  Directory where checkpoints, logs, and climatology_maps.nc are saved
    # var_name:     NetCDF variable name for temperature (default: air_temperature)
    # =========================================================================
    hr_dir: Optional[str] = field(default=None)
    lr_dir: Optional[str] = field(default=None)
    results_dir: Optional[str] = field(default=None)
    var_name: str = field(default="air_temperature")
    
    # Data processing
    train_fraction: float = field(default=0.70)
    val_fraction: float = field(default=0.15)
    patch_size: int = field(default=128)
    stride: int = field(default=64)
    sample_stats: int = field(default=2500)
    
    # Training parameters - OPTIMIZED
    batch_size: int = field(default=64) # Increased from 32
    inference_batch_size: int = field(default=128) # New: Batch size for prediction
    epochs: int = field(default=100)
    patience: int = field(default=15)
    learning_rate: float = field(default=2e-4) # Slightly increased
    weight_decay: float = field(default=1e-5)
    gradient_clip: float = field(default=1.0)
    
    # Performance optimizations
    cache_data: bool = field(default=True)
    use_memory_format: bool = field(default=True)
    compile_model: bool = field(default=True)
    mixed_precision: bool = field(default=True)
    max_cache_size_gb: float = field(default=30.0)
    
    # Gaussian weighting
    gaussian_sigma_factor: float = field(default=0.3)
    edge_fade_width: int = field(default=16)
    
    # Diffusion parameters
    t_steps: int = field(default=1000)
    beta_start: float = field(default=0.0001)
    beta_end: float = field(default=0.02)
    ddim_steps: int = field(default=50)
    
    # Model parameters
    base_dim: int = field(default=64)
    time_dim: int = field(default=256)
    use_ema: bool = field(default=True)
    ema_decay: float = field(default=0.9999)
    
    # System parameters - OPTIMIZED
    num_workers: int = field(default=8)
    prefetch_factor: int = field(default=4)
    pin_memory: bool = field(default=True)
    persistent_workers: bool = field(default=True)
    
    # Logging
    use_wandb: bool = field(default=False)
    resume_checkpoint: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hr_dir is None or self.lr_dir is None or self.results_dir is None:
            raise ValueError(
                "hr_dir, lr_dir, and results_dir must be set before training. "
                "Edit the Config defaults or pass --hr-dir, --lr-dir, --results-dir as CLI arguments."
            )
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(self.hr_dir):
            raise ValueError(f"HR directory not found: {self.hr_dir}")
        if not os.path.exists(self.lr_dir):
            raise ValueError(f"LR directory not found: {self.lr_dir}")
        
        if self.batch_size % 8 != 0:
            warnings.warn("Batch size not divisible by 8, may affect performance")
        if self.train_fraction + self.val_fraction > 0.95:
            warnings.warn("Less than 5% data for testing")
    
    def save_config(self, path: str):
        """Save configuration to JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# -------------------------
# Logging Setup
# -------------------------
def setup_logging(results_dir: str) -> logging.Logger:
    """Setup comprehensive logging with both file and console output."""
    log_file = os.path.join(results_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('ddpm_climate')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# -------------------------
# Device Management
# -------------------------
import sys
import argparse
import subprocess

def get_least_busy_gpu() -> int:
    """Return the index of the GPU with the most free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        memory_free = [int(x) for x in result.stdout.strip().split("\n")]
        return max(range(len(memory_free)), key=lambda i: memory_free[i])
    except Exception as e:
        print(f"Could not auto-detect least busy GPU: {e}")
        return 0

def setup_device() -> torch.device:
    """Setup optimal device configuration."""
    parser = argparse.ArgumentParser(description="DDPM climate downscaling training")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use (-1 for auto-select)")
    
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if args.gpu == -1:
            chosen_gpu = get_least_busy_gpu()
            print(f"Auto-selected GPU {chosen_gpu} with the most free memory.")
        else:
            chosen_gpu = args.gpu
            if chosen_gpu >= gpu_count:
                print(f"Warning: GPU index {chosen_gpu} is out of range. Using GPU 0.")
                chosen_gpu = 0
        
        device = torch.device(f"cuda:{chosen_gpu}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU. This will be slow.")
    
    print(f"Using device: {device}")
    return device

# -------------------------
# Utility Functions
# -------------------------
def nearest_fill_nan_fast(arr: ArrayType) -> ArrayType:
    """Fast NaN filling using scipy."""
    if np.isnan(arr).sum() == 0:
        return arr
    
    mask = np.isnan(arr)
    arr = arr.copy()
    arr[mask] = 0
    
    # Distance transform approach (faster than KDTree for dense arrays)
    dist, idx = ndimage.distance_transform_edt(
        mask, return_distances=True, return_indices=True
    )
    arr[mask] = arr[tuple(idx[:, mask])]
    return arr

def sorted_alphanumeric(lst: List[str]) -> List[str]:
    """Sort list alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)

def safe_load_dataset(file_path: str, var_name: str, logger: logging.Logger) -> Optional[ArrayType]:
    """Safely load NetCDF with error handling."""
    try:
        with xr.open_dataset(file_path) as ds:
            if var_name not in ds:
                raise KeyError(f"Variable {var_name} not found in {file_path}")
            return np.squeeze(ds[var_name].values).astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def list_pairs(hr_dir: str, lr_dir: str) -> List[Tuple[str, str]]:
    """List matching HR-LR file pairs."""
    hr_files = sorted_alphanumeric([f for f in os.listdir(hr_dir) if f.endswith(".nc")])
    lr_files_set = set(os.listdir(lr_dir))
    pairs = []
    
    for hr in hr_files:
        base = os.path.splitext(hr)[0]
        expected_lr = f"{base}_1deg_to_hr.nc"
        if expected_lr in lr_files_set:
            pairs.append((os.path.join(hr_dir, hr), os.path.join(lr_dir, expected_lr)))
    
    return pairs

def split_pairs_by_fraction(
    pairs: List[Tuple[str, str]], 
    train_frac: float, 
    val_frac: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split pairs chronologically by fraction."""
    n_total = len(pairs)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    return train_pairs, val_pairs, test_pairs

# -------------------------
# Exponential Moving Average
# -------------------------
class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict[str, TensorType]:
        """Get state dict for saving."""
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: Dict[str, TensorType]):
        """Load state dict."""
        self.shadow = state_dict

# -------------------------
# Enhanced Normalization
# -------------------------
class EfficientNormalizer:
    """Memory-efficient normalization for climate data."""
    
    def __init__(self, mean_map: ArrayType, std_map: ArrayType, method: str = 'robust'):
        self.mean_map = torch.from_numpy(mean_map).float()
        self.std_map = torch.from_numpy(std_map).float()
        self.method = method
        
        if method == 'robust':
            self.scale_map = torch.where(self.std_map < 1e-7, 1e-7, self.std_map)
        elif method == 'minmax':
            self.data_min = self.mean_map - 3 * self.std_map
            self.data_max = self.mean_map + 3 * self.std_map
    
    def normalize_patch(self, patch: ArrayType, top: int, left: int, patch_size: int) -> ArrayType:
        """Efficiently normalize a patch using precomputed maps."""
        patch_tensor = torch.from_numpy(patch).float()
        mean = self.mean_map[top:top+patch_size, left:left+patch_size]
        
        if self.method == 'standard' or self.method == 'robust':
            scale = self.scale_map[top:top+patch_size, left:left+patch_size]
            normalized = (patch_tensor - mean) / scale
        elif self.method == 'minmax':
            data_min = self.data_min[top:top+patch_size, left:left+patch_size]
            data_max = self.data_max[top:top+patch_size, left:left+patch_size]
            normalized = 2 * (patch_tensor - data_min) / (data_max - data_min + 1e-7) - 1
        
        return normalized.numpy()
    
    def denormalize_patch(self, patch: ArrayType, top: int, left: int, patch_size: int) -> ArrayType:
        """Denormalize a patch back to original scale."""
        patch_tensor = torch.from_numpy(patch).float()
        mean = self.mean_map[top:top+patch_size, left:left+patch_size]
        
        if self.method == 'standard' or self.method == 'robust':
            scale = self.scale_map[top:top+patch_size, left:left+patch_size]
            denormalized = patch_tensor * scale + mean
        elif self.method == 'minmax':
            data_min = self.data_min[top:top+patch_size, left:left+patch_size]
            data_max = self.data_max[top:top+patch_size, left:left+patch_size]
            denormalized = (patch_tensor + 1) / 2 * (data_max - data_min) + data_min
        
        return denormalized.numpy()

# -------------------------
# Climatology Management
# -------------------------
class ClimatologyManager:
    """Manage climatological statistics for normalization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.climatology_file = os.path.join(config.results_dir, "climatology_maps.nc")
    
    def compute_climatology(
        self, 
        train_pairs: List[Tuple[str, str]], 
        var_name: str, 
        sample_size: int
    ) -> Tuple[ArrayType, ArrayType, Dict[str, ArrayType]]:
        """Compute pointwise climatology from training data."""
        print("Computing climatology from training data sample...")
        train_hr_files = [p[0] for p in train_pairs]
        
        if len(train_hr_files) <= sample_size:
            sample_files = train_hr_files
        else:
            sample_files = list(np.random.choice(train_hr_files, sample_size, replace=False))
        
        sum_arr, sumsq_arr, count = None, None, 0
        
        for f in tqdm(sample_files, desc="Computing climatology"):
            with xr.open_dataset(f) as ds:
                arr = np.squeeze(ds[var_name].values).astype(np.float64)
                arr = nearest_fill_nan_fast(arr)
                
                if sum_arr is None:
                    sum_arr = np.zeros_like(arr)
                    sumsq_arr = np.zeros_like(arr)
                
                sum_arr += arr
                sumsq_arr += arr**2
                count += 1
        
        mean_map = sum_arr / count
        std_map = np.sqrt(np.maximum((sumsq_arr / count) - (mean_map**2), 1e-7))
        
        # Compute percentiles for robust statistics
        percentiles = [5, 25, 50, 75, 95]
        perc_maps = {}
        
        for p in percentiles:
            perc_vals = []
            for f in sample_files[:100]: # Use subset for percentiles
                with xr.open_dataset(f) as ds:
                    arr = np.squeeze(ds[var_name].values).astype(np.float64)
                    arr = nearest_fill_nan_fast(arr)
                    perc_vals.append(arr)
            
            perc_vals = np.stack(perc_vals, axis=0)
            perc_maps[f'p{p}'] = np.percentile(perc_vals, p, axis=0)
        
        return mean_map.astype(np.float32), std_map.astype(np.float32), perc_maps
    
    def get_or_compute_climatology(
        self, 
        train_pairs: List[Tuple[str, str]]
    ) -> Tuple[ArrayType, ArrayType, Dict[str, ArrayType]]:
        """Get cached climatology or compute new one."""
        if os.path.exists(self.climatology_file):
            print(f"Loading cached climatology from {self.climatology_file}...")
            with xr.open_dataset(self.climatology_file) as ds:
                mean_map = ds['mean_map'].values
                std_map = ds['std_map'].values
                perc_maps = {var: ds[var].values for var in ds.data_vars if var.startswith('p')}
        else:
            mean_map, std_map, perc_maps = self.compute_climatology(
                train_pairs, self.config.var_name, self.config.sample_stats
            )
            
            # Save climatology
            ds_out = xr.Dataset({
                "mean_map": (("lat", "lon"), mean_map),
                "std_map": (("lat", "lon"), std_map),
                **{k: (("lat", "lon"), v) for k, v in perc_maps.items()}
            })
            
            print(f"Saving climatology to {self.climatology_file}...")
            ds_out.to_netcdf(self.climatology_file)
        
        return mean_map, std_map, perc_maps

# -------------------------
# Loss Functions
# -------------------------
class SimpleMSELoss(nn.Module):
    """Simple MSE loss for DDPM training."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        noise_pred: TensorType, 
        true_noise: TensorType, 
        hr_target: Optional[TensorType] = None
    ) -> Tuple[TensorType, Dict[str, float]]:
        mse_loss = F.mse_loss(noise_pred, true_noise)
        
        loss_dict = {
            'mse': mse_loss.item(),
            'total': mse_loss.item()
        }
        
        return mse_loss, loss_dict

# -------------------------
# Noise Scheduler
# -------------------------
class NoiseScheduler:
    """DDPM noise scheduler with linear beta schedule."""
    
    def __init__(
        self, 
        num_timesteps: int = 1000, 
        beta_start: float = 0.0001, 
        beta_end: float = 0.02, 
        device: str = 'cpu'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
    
    def add_noise(
        self, 
        x_0: TensorType, 
        t: TensorType, 
        noise: Optional[TensorType] = None
    ) -> Tuple[TensorType, TensorType]:
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        if t.dim() == 0:
            t = t.repeat(x_0.shape[0])
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise
    
    def sample_prev_timestep(
        self, 
        x_t: TensorType, 
        noise_pred: TensorType, 
        t: TensorType
    ) -> TensorType:
        """Reverse diffusion step."""
        if t.dim() == 0:
            t = t.repeat(x_t.shape[0])
        
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(-1, t).view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        posterior_variance_t = self.posterior_variance.gather(-1, t).view(-1, 1, 1, 1)
        betas_t = self.betas.gather(-1, t).view(-1, 1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * sqrt_recipm1_alphas_cumprod_t * noise_pred
        )
        
        noise = torch.randn_like(x_t)
        noise_mask = (t > 0).float().view(-1, 1, 1, 1)
        
        return model_mean + noise_mask * torch.sqrt(posterior_variance_t) * noise
    
    def ddim_sample_prev_timestep(
        self, 
        x_t: TensorType, 
        noise_pred: TensorType, 
        t: int, 
        t_prev: int, 
        eta: float = 0.0
    ) -> TensorType:
        """DDIM sampling step."""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t)) * noise_pred
        
        noise = eta * torch.randn_like(x_t) * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + noise
        
        return x_prev

# -------------------------
# U-Net Architecture
# -------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: TensorType) -> TensorType:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class AttentionBlock(nn.Module):
    """Self-attention block for better spatial modeling."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: TensorType) -> TensorType:
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
    """ResNet block with time embedding."""
    
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: TensorType, t: TensorType) -> TensorType:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding
        time_emb = self.activation(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + self.skip(x)

class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        time_emb_dim: int, 
        num_layers: int = 2, 
        downsample: bool = True, 
        attention: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.downsample = downsample
        
        self.resnets = nn.ModuleList([
            ResNetBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_layers)
        ])
        
        if attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_ch) for _ in range(num_layers)
            ])
        else:
            self.attentions = [None] * num_layers
        
        if downsample:
            self.downsample_conv = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x: TensorType, t: TensorType) -> TensorType:
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t)
            if attn is not None:
                x = attn(x)
        
        if self.downsample:
            x = self.downsample_conv(x)
        
        return x

class UpBlock(nn.Module):
    """Upsampling block with fixed channel handling."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        time_emb_dim: int, 
        num_layers: int = 2, 
        upsample: bool = True, 
        attention: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.upsample = upsample
        
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                res_in_ch = in_ch + out_ch
            else:
                res_in_ch = out_ch
            self.resnets.append(ResNetBlock(res_in_ch, out_ch, time_emb_dim))
        
        if attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_ch) for _ in range(num_layers)
            ])
        else:
            self.attentions = [None] * num_layers
        
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
    
    def forward(self, x: TensorType, skip: TensorType, t: TensorType) -> TensorType:
        if self.upsample:
            x = self.upsample_conv(x)

        x = torch.cat([x, skip], dim=1)
        
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t)
            if attn is not None:
                x = attn(x)
        
        return x

class EnhancedUNet(nn.Module):
    """Enhanced U-Net with attention and improved architecture."""
    
    def __init__(
        self, 
        in_channels: int = 2, 
        out_channels: int = 1, 
        time_dim: int = 256, 
        base_dim: int = 64
    ):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Down path
        down_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.downs = nn.ModuleList([
            DownBlock(down_dims[0], down_dims[1], time_dim, attention=False),
            DownBlock(down_dims[1], down_dims[2], time_dim, attention=False),
            DownBlock(down_dims[2], down_dims[3], time_dim, attention=True),
        ])
        
        # Bottleneck
        self.bottleneck = ResNetBlock(down_dims[3], down_dims[3], time_dim)
        self.bottleneck_attn = AttentionBlock(down_dims[3])
        
        # Up path
        self.ups = nn.ModuleList([
            UpBlock(down_dims[3], down_dims[2], time_dim, attention=True),
            UpBlock(down_dims[2], down_dims[1], time_dim, attention=False),
            UpBlock(down_dims[1], down_dims[0], time_dim, attention=False),
        ])
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )
    
    def forward(self, x: TensorType, t: TensorType) -> TensorType:
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Down path - store skip connections
        skips = []
        
        skips.append(x)
        x = self.downs[0](x, t_emb)
        
        skips.append(x)
        x = self.downs[1](x, t_emb)
        
        skips.append(x)
        x = self.downs[2](x, t_emb)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)
        x = self.bottleneck_attn(x)
        
        # Up path
        x = self.ups[0](x, skips[2], t_emb)
        x = self.ups[1](x, skips[1], t_emb)
        x = self.ups[2](x, skips[0], t_emb)
        
        # Output
        return self.conv_out(x)

# -------------------------
# Gaussian Patch Processing
# -------------------------
def create_gaussian_weights(patch_size: int, sigma_factor: float = 0.3) -> np.ndarray:
    """Create 2D Gaussian weights for patch blending."""
    center = patch_size // 2
    y, x = np.ogrid[:patch_size, :patch_size]
    
    distance_sq = (x - center) ** 2 + (y - center) ** 2
    
    sigma = sigma_factor * patch_size
    weights = np.exp(-distance_sq / (2 * sigma ** 2))
    
    weights = weights / np.max(weights)
    
    return weights.astype(np.float32)

def extract_patches_improved(
    img: np.ndarray, 
    patch_size: int = 128, 
    stride: int = 64,
    ensure_coverage: bool = True
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Extract overlapping patches with improved edge handling."""
    h, w = img.shape
    patches, positions = [], []
    
    top_positions = list(range(0, h - patch_size + 1, stride))
    left_positions = list(range(0, w - patch_size + 1, stride))
    
    if ensure_coverage:
        if top_positions and top_positions[-1] != h - patch_size:
            top_positions.append(h - patch_size)
        if left_positions and left_positions[-1] != w - patch_size:
            left_positions.append(w - patch_size)
    
    for top in top_positions:
        for left in left_positions:
            top = min(top, h - patch_size)
            left = min(left, w - patch_size)
            
            patch = img[top:top + patch_size, left:left + patch_size]
            patches.append(patch.copy())
            positions.append((top, left))
    
    return patches, positions

def stitch_patches_gaussian(
    patches: List[np.ndarray], 
    positions: List[Tuple[int, int]], 
    full_shape: Tuple[int, int], 
    patch_size: int = 128,
    sigma_factor: float = 0.3,
    edge_fade: bool = True
) -> np.ndarray:
    """Stitch patches using Gaussian weighting for seamless blending."""
    h, w = full_shape
    output = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    gaussian_weights = create_gaussian_weights(patch_size, sigma_factor)
    
    for patch, (top, left) in zip(patches, positions):
        end_top = min(top + patch_size, h)
        end_left = min(left + patch_size, w)
        actual_h = end_top - top
        actual_w = end_left - left
        
        weights = gaussian_weights[:actual_h, :actual_w].copy()
        
        if edge_fade:
            weights = apply_edge_fading(weights, top, left, actual_h, actual_w, h, w)
        
        patch_region = patch[:actual_h, :actual_w]
        
        output[top:end_top, left:end_left] += patch_region * weights
        weight_sum[top:end_top, left:end_left] += weights
    
    weight_sum = np.where(weight_sum < 1e-8, 1.0, weight_sum)
    output = output / weight_sum
    
    return output

def apply_edge_fading(
    weights: np.ndarray, 
    top: int, 
    left: int, 
    patch_h: int, 
    patch_w: int,
    img_h: int, 
    img_w: int, 
    fade_width: int = 16
) -> np.ndarray:
    """Apply edge fading to reduce weights near image boundaries."""
    weights = weights.copy()
    
    dist_to_top = top
    dist_to_left = left
    dist_to_bottom = img_h - (top + patch_h)
    dist_to_right = img_w - (left + patch_w)
    
    for i in range(patch_h):
        for j in range(patch_w):
            pixel_dist_top = dist_to_top + i
            pixel_dist_left = dist_to_left + j
            pixel_dist_bottom = dist_to_bottom + (patch_h - 1 - i)
            pixel_dist_right = dist_to_right + (patch_w - 1 - j)
            
            min_edge_dist = min(pixel_dist_top, pixel_dist_left, 
                                pixel_dist_bottom, pixel_dist_right)
            
            if min_edge_dist < fade_width:
                fade_factor = min_edge_dist / fade_width
                weights[i, j] *= fade_factor
    
    return weights

# -------------------------
# Optimized Dataset with Caching
# -------------------------
class CachedPatchDataset(Dataset):
    """High-performance dataset with intelligent caching."""
    
    def __init__(
        self, 
        file_pairs: List[Tuple[str, str]], 
        normalizer: EfficientNormalizer, 
        config: Config, 
        augment: bool = True
    ):
        self.normalizer = normalizer
        self.config = config
        self.augment = augment
        self.cache = {}
        self.cache_size_bytes = 0
        self.max_cache_bytes = int(config.max_cache_size_gb * 1024**3)
        
        self.patch_index = []
        self._build_patch_index_with_cache(file_pairs)
        
        print(f"Dataset initialized with {len(self.patch_index)} patches")
        print(f"Cache contains {len(self.cache)} files ({self.cache_size_bytes / 1024**3:.2f} GB)")
    
    def _build_patch_index_with_cache(self, file_pairs: List[Tuple[str, str]]):
        """Build index and cache files that fit in memory."""
        print("Building optimized patch index with caching...")
        patch_size = self.config.patch_size
        stride = self.config.stride
        
        for hr_file, lr_file in tqdm(file_pairs, desc="Processing files"):
            hr_size = os.path.getsize(hr_file)
            lr_size = os.path.getsize(lr_file)
            total_size = hr_size + lr_size
            
            should_cache = (
                self.config.cache_data and 
                self.cache_size_bytes + total_size < self.max_cache_bytes and
                total_size < 500 * 1024 * 1024
            )
            
            if should_cache:
                try:
                    with xr.open_dataset(hr_file) as ds_hr:
                        hr_data = np.squeeze(ds_hr[self.config.var_name].values).astype(np.float32)
                    with xr.open_dataset(lr_file) as ds_lr:
                        lr_data = np.squeeze(ds_lr[self.config.var_name].values).astype(np.float32)
                    
                    hr_data = nearest_fill_nan_fast(hr_data)
                    lr_data = nearest_fill_nan_fast(lr_data)
                    
                    self.cache[hr_file] = hr_data
                    self.cache[lr_file] = lr_data
                    self.cache_size_bytes += total_size
                    
                except Exception as e:
                    print(f"Failed to cache {hr_file}: {e}")
                    should_cache = False
            
            if should_cache:
                h, w = hr_data.shape
            else:
                with xr.open_dataset(hr_file) as ds:
                    h, w = ds[self.config.var_name].shape[-2:]
            
            if h < patch_size or w < patch_size:
                continue # Skip images smaller than the patch size

            top_positions = list(range(0, h - patch_size + 1, stride))
            if not top_positions or top_positions[-1] != h - patch_size:
                top_positions.append(h - patch_size)

            left_positions = list(range(0, w - patch_size + 1, stride))
            if not left_positions or left_positions[-1] != w - patch_size:
                left_positions.append(w - patch_size)
            
            for top in top_positions:
                for left in left_positions:
                    self.patch_index.append((hr_file, lr_file, top, left, should_cache))

    def __len__(self) -> int:
        return len(self.patch_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_file, lr_file, top, left, is_cached = self.patch_index[idx]
        
        if is_cached:
            hr_data = self.cache[hr_file]
            lr_data = self.cache[lr_file]
        else:
            hr_data, lr_data = self._load_patch_region(hr_file, lr_file, top, left)
        
        hr_patch = hr_data[top:top + self.config.patch_size, left:left + self.config.patch_size].copy()
        lr_patch = lr_data[top:top + self.config.patch_size, left:left + self.config.patch_size].copy()
        
        hr_norm = self.normalizer.normalize_patch(hr_patch, top, left, self.config.patch_size)
        lr_norm = self.normalizer.normalize_patch(lr_patch, top, left, self.config.patch_size)
        
        hr_norm = np.clip(hr_norm, -5, 5)
        lr_norm = np.clip(lr_norm, -5, 5)
        
        if self.augment:
            hr_norm, lr_norm = self._augment_patches_fast(hr_norm, lr_norm)
        
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_norm, dtype=np.float32)).unsqueeze(0)
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_norm, dtype=np.float32)).unsqueeze(0)
        
        return lr_tensor, hr_tensor
    
    def _load_patch_region(self, hr_file: str, lr_file: str, top: int, left: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load only the required region from NetCDF files."""
        patch_size = self.config.patch_size
        
        with xr.open_dataset(hr_file) as ds_hr, xr.open_dataset(lr_file) as ds_lr:
            hr_patch = np.squeeze(ds_hr[self.config.var_name].isel(
                time=0,
                lat=slice(top, top + patch_size),
                lon=slice(left, left + patch_size)
            ).values).astype(np.float32)
            
            lr_patch = np.squeeze(ds_lr[self.config.var_name].isel(
                time=0,
                lat=slice(top, top + patch_size),
                lon=slice(left, left + patch_size)
            ).values).astype(np.float32)
        
        hr_patch = nearest_fill_nan_fast(hr_patch)
        lr_patch = nearest_fill_nan_fast(lr_patch)
        
        return hr_patch, lr_patch
    
    def _augment_patches_fast(self, hr_patch: np.ndarray, lr_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized data augmentation."""
        rand_vals = np.random.random(3)
        
        if rand_vals[0] > 0.5:
            hr_patch = hr_patch[:, ::-1]
            lr_patch = lr_patch[:, ::-1]
        
        if rand_vals[1] > 0.5:
            hr_patch = hr_patch[::-1, :]
            lr_patch = lr_patch[::-1, :]
        
        k = int(rand_vals[2] * 4)
        if k > 0:
            hr_patch = np.rot90(hr_patch, k)
            lr_patch = np.rot90(lr_patch, k)
        
        return hr_patch, lr_patch

# -------------------------
# Optimized DataLoader
# -------------------------
def create_optimized_dataloader(
    dataset: Dataset, 
    config: Config, 
    shuffle: bool = True
) -> DataLoader:
    """Create optimized DataLoader with best performance settings."""
    
    def optimized_collate(batch):
        lr_tensors, hr_tensors = zip(*batch)
        
        lr_batch = torch.stack(lr_tensors, dim=0)
        hr_batch = torch.stack(hr_tensors, dim=0)
        
        if config.use_memory_format:
            lr_batch = lr_batch.to(memory_format=torch.channels_last)
            hr_batch = hr_batch.to(memory_format=torch.channels_last)
        
        return lr_batch, hr_batch
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        collate_fn=optimized_collate,
        drop_last=True,
        multiprocessing_context='fork' if os.name != 'nt' else None,
    )

# -------------------------
# Model Setup with Optimizations
# -------------------------
def setup_optimized_model(model: nn.Module, config: Config, device: torch.device) -> nn.Module:
    """Setup model with performance optimizations."""
    
    model = model.to(device)
    
    if config.use_memory_format and device.type == 'cuda':
        try:
            model = model.to(memory_format=torch.channels_last)
            print("Enabled channels_last memory format")
        except Exception as e:
            print(f"Could not enable channels_last: {e}")
    
    if config.compile_model:
        try:
            model = torch.compile(
                model, 
                mode='reduce-overhead',
                dynamic=False
            )
            print("Model compiled successfully")
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    return model

# -------------------------
# Optimized Trainer
# -------------------------
class OptimizedTrainer:
    """High-performance trainer with optimizations."""
    
    def __init__(self, model: nn.Module, config: Config, device: torch.device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        self.loss_fn = SimpleMSELoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=5, 
            factor=0.5, 
            verbose=True
        )
        
        self.use_amp = config.mixed_precision and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler(
                init_scale=2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
        else:
            self.scaler = None
        
        if config.use_ema:
            self.ema = EMA(model, decay=config.ema_decay)
        else:
            self.ema = None
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_step_optimized(
        self, 
        dataloader: DataLoader, 
        scheduler: NoiseScheduler, 
        epoch: int
    ) -> Tuple[float, dict]:
        """Optimized training step with maximum performance."""
        self.model.train()
        total_loss = 0
        loss_components = {'mse': 0, 'total': 0}
        
        # Pre-generate random timesteps for better performance
        total_batches = len(dataloader)
        timesteps_epoch = torch.randint(
            0, scheduler.num_timesteps, 
            (total_batches * self.config.batch_size,), 
            device=self.device, 
            dtype=torch.long
        )
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(pbar):
            # Fast data transfer
            if self.config.use_memory_format:
                lr_batch = lr_batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                hr_batch = hr_batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            else:
                lr_batch = lr_batch.to(self.device, non_blocking=True)
                hr_batch = hr_batch.to(self.device, non_blocking=True)
            
            # Use pre-generated timesteps
            batch_start = batch_idx * self.config.batch_size
            batch_end = batch_start + lr_batch.shape[0]
            t = timesteps_epoch[batch_start:batch_end]
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast():
                    noisy_hr, true_noise = scheduler.add_noise(hr_batch, t)
                    model_input = torch.cat([noisy_hr, lr_batch], dim=1)
                    noise_pred = self.model(model_input, t)
                    loss, loss_dict = self.loss_fn(noise_pred, true_noise)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noisy_hr, true_noise = scheduler.add_noise(hr_batch, t)
                model_input = torch.cat([noisy_hr, lr_batch], dim=1)
                noise_pred = self.model(model_input, t)
                loss, loss_dict = self.loss_fn(noise_pred, true_noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            if self.ema is not None:
                self.ema.update()
            
            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mse': f"{loss_dict.get('mse', 0):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    def val_step(
        self, 
        dataloader: DataLoader, 
        scheduler: NoiseScheduler
    ) -> Tuple[float, Dict[str, float]]:
        """Validation step."""
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0
        loss_components = {'mse': 0, 'total': 0}
        
        with torch.no_grad():
            for lr_batch, hr_batch in tqdm(dataloader, desc='Validation'):
                if self.config.use_memory_format:
                    lr_batch = lr_batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    hr_batch = hr_batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    lr_batch = lr_batch.to(self.device, non_blocking=True)
                    hr_batch = hr_batch.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        t = torch.randint(
                            0, scheduler.num_timesteps,
                            (hr_batch.shape[0],),
                            device=self.device,
                            dtype=torch.long
                        )
                        
                        noisy_hr, true_noise = scheduler.add_noise(hr_batch, t)
                        model_input = torch.cat([noisy_hr, lr_batch], dim=1)
                        noise_pred = self.model(model_input, t)
                        loss, loss_dict = self.loss_fn(noise_pred, true_noise)
                else:
                    t = torch.randint(
                        0, scheduler.num_timesteps,
                        (hr_batch.shape[0],),
                        device=self.device,
                        dtype=torch.long
                    )
                    
                    noisy_hr, true_noise = scheduler.add_noise(hr_batch, t)
                    model_input = torch.cat([noisy_hr, lr_batch], dim=1)
                    noise_pred = self.model(model_input, t)
                    loss, loss_dict = self.loss_fn(noise_pred, true_noise)
                
                total_loss += loss.item()
                for key in loss_components:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key]
        
        if self.ema is not None:
            self.ema.restore()
        
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if is_best:
            save_path = os.path.join(self.config.results_dir, 'best_model.pt')
            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved best model checkpoint to {save_path}")
        
        latest_path = os.path.join(self.config.results_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        return checkpoint.get('epoch', 0)
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        scheduler: NoiseScheduler
    ):
        """Main training loop with resumption support."""
        start_epoch = 1
        
        if self.config.resume_checkpoint and os.path.exists(self.config.resume_checkpoint):
            start_epoch = self.resume_from_checkpoint(self.config.resume_checkpoint) + 1
        
        self.logger.info(f"Starting training from epoch {start_epoch}...")
        
        for epoch in range(start_epoch, self.config.epochs + 1):
            # Training
            train_loss, train_components = self.train_step_optimized(train_loader, scheduler, epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_components = self.val_step(val_loader, scheduler)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.lr_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, LR: {current_lr:.6f}"
            )
            
            # Wandb logging
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    **{f'train_{k}': v for k, v in train_components.items()},
                    **{f'val_{k}': v for k, v in val_components.items()}
                })
            
            # Early stopping and checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"New best validation loss: {val_loss:.5f}")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best=False)
            
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping after {self.config.patience} epochs without improvement")
                break
        
        self.logger.info("Training completed!")

# -------------------------
# Inference Functions
# -------------------------
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, 
    lr_condition: TensorType, 
    scheduler: NoiseScheduler, 
    device: torch.device, 
    progress: bool = True
) -> TensorType:
    """Sample from DDPM model using standard reverse diffusion."""
    model.eval()
    x = torch.randn_like(lr_condition).to(device)
    
    timesteps = range(scheduler.num_timesteps)
    if progress:
        timesteps = tqdm(reversed(timesteps), desc="DDPM Sampling", total=scheduler.num_timesteps)
    else:
        timesteps = reversed(timesteps)
    
    for i in timesteps:
        t = torch.tensor([i] * x.shape[0], device=device, dtype=torch.long)
        
        with torch.cuda.amp.autocast():
            model_input = torch.cat([x, lr_condition], dim=1)
            noise_pred = model(model_input, t)
        
        x = scheduler.sample_prev_timestep(x, noise_pred, t)
    
    return x

@torch.no_grad()
def sample_ddim(
    model: nn.Module, 
    lr_condition: TensorType, 
    scheduler: NoiseScheduler, 
    device: torch.device, 
    ddim_steps: int = 50, 
    eta: float = 0.0, 
    progress: bool = True
) -> TensorType:
    """Faster deterministic sampling using DDIM."""
    model.eval()
    
    c = scheduler.num_timesteps // ddim_steps
    timesteps = list(range(0, scheduler.num_timesteps, c))[::-1]
    
    x = torch.randn_like(lr_condition).to(device)
    
    iterator = enumerate(timesteps)
    if progress:
        iterator = tqdm(iterator, desc="DDIM Sampling", total=len(timesteps))
    
    for i, t in iterator:
        t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        
        with torch.cuda.amp.autocast():
            model_input = torch.cat([x, lr_condition], dim=1)
            t_batch = torch.tensor([t] * x.shape[0], device=device, dtype=torch.long)
            noise_pred = model(model_input, t_batch)
        
        x = scheduler.ddim_sample_prev_timestep(x, noise_pred, t, t_prev, eta)
    
    return x

def predict_full_image(model, lr_image, normalizer, scheduler, device, config, use_ddim=True):
    """
    Predict full HR image using ARTIFACT-FREE patch-based processing.
    
    CRITICAL: Uses "Stitch-First, Denormalize-Later" strategy to prevent grid artifacts.
    
    Processing Order:
    1. Normalize LR patches using local climatology
    2. Predict HR patches in NORMALIZED space
    3. Store NORMALIZED predictions (no per-patch denormalization)
    4. Stitch NORMALIZED patches using Gaussian blending
    5. Denormalize GLOBALLY using full-resolution climatology maps
    
    Args:
        model: DDPM model
        lr_image: Low-resolution input image (H, W)
        normalizer: EfficientNormalizer instance
        scheduler: NoiseScheduler instance
        device: torch device
        config: Config object with patch_size, stride, sigma_factor
        use_ddim: Whether to use DDIM sampling (faster)
    
    Returns:
        High-resolution prediction (H, W) in original temperature units, artifact-free
    """
    
    # Extract patches
    patches_lr, positions = extract_patches(lr_image, config.patch_size, config.stride)
    pred_patches_norm = []  # ✅ Store NORMALIZED predictions
    
    for lr_patch, (top, left) in zip(patches_lr, positions):
        # Normalize LR patch using local climatology
        lr_norm = normalizer.normalize_patch(lr_patch, top, left, config.patch_size)
        lr_norm = np.clip(lr_norm, -5, 5)
        
        # Convert to tensor
        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Generate HR prediction in NORMALIZED space
        if use_ddim:
            hr_norm_pred = sample_ddim(model, lr_tensor, scheduler, device, config.ddim_steps)
        else:
            hr_norm_pred = sample_ddpm(model, lr_tensor, scheduler, device)
        
        hr_norm_pred = hr_norm_pred.squeeze().cpu().numpy()
        hr_norm_pred = np.clip(hr_norm_pred, -5, 5)
        
        # ✅ CRITICAL: Store NORMALIZED prediction (no denormalization yet!)
        pred_patches_norm.append(hr_norm_pred)
    
    # ✅ CRITICAL: Stitch patches in NORMALIZED space using Gaussian blending
    sr_normalized = stitch_patches_gaussian(
        pred_patches_norm,  # ✅ These are NORMALIZED
        positions,
        lr_image.shape,
        config.patch_size,
        config.sigma_factor
    )
    
    # ✅ CRITICAL: Global denormalization using full-resolution climatology maps
    full_mean = normalizer.mean_map.numpy() if torch.is_tensor(normalizer.mean_map) else normalizer.mean_map
    full_std = normalizer.scale_map.numpy() if torch.is_tensor(normalizer.scale_map) else normalizer.scale_map
    
    h, w = lr_image.shape
    result = sr_normalized * full_std[:h, :w] + full_mean[:h, :w]
    
    return result  # ✅ Returns artifact-free result

# -------------------------
# Evaluation
# -------------------------
class ClimateEvaluator:
    """Comprehensive evaluation for climate downscaling."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def compute_metrics(self, predictions: ArrayType, targets: ArrayType) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        if len(pred_valid) == 0:
            return {}
        
        mse = mean_squared_error(target_valid, pred_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_valid, pred_valid)
        
        r_coeff, _ = stats.pearsonr(pred_valid, target_valid)
        r2 = r_coeff ** 2
        
        bias = np.mean(pred_valid - target_valid)
        std_ratio = np.std(pred_valid) / np.std(target_valid)
        
        ks_stat, ks_pvalue = ks_2samp(pred_valid, target_valid)
        wasserstein_dist = wasserstein_distance(pred_valid, target_valid)
        
        percentile_bias_95 = (np.percentile(pred_valid, 95) - np.percentile(target_valid, 95))
        percentile_bias_5 = (np.percentile(pred_valid, 5) - np.percentile(target_valid, 5))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correlation': r_coeff,
            'bias': bias,
            'std_ratio': std_ratio,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein_distance': wasserstein_dist,
            'percentile_bias_95': percentile_bias_95,
            'percentile_bias_5': percentile_bias_5
        }
    
    def evaluate_test_set(
        self, 
        model: nn.Module, 
        test_pairs: List[Tuple[str, str]], 
        normalizer: EfficientNormalizer, 
        scheduler: NoiseScheduler, 
        device: torch.device, 
        ema: Optional[EMA] = None, 
        max_samples: int = 10
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        self.logger.info("Starting test set evaluation...")
        
        if ema is not None:
            ema.apply_shadow()
        
        all_metrics = []
        all_predictions = []
        all_targets = []
        all_lr_inputs = []
        
        for i, (hr_file, lr_file) in enumerate(tqdm(test_pairs[:max_samples], desc="Evaluating")):
            try:
                hr_image = safe_load_dataset(hr_file, self.config.var_name, self.logger)
                lr_image = safe_load_dataset(lr_file, self.config.var_name, self.logger)
                
                if hr_image is None or lr_image is None:
                    continue
                
                hr_image = nearest_fill_nan_fast(hr_image)
                lr_image = nearest_fill_nan_fast(lr_image)
                
                prediction = predict_full_image(
                    model, lr_image, normalizer, scheduler, device, 
                    self.config, use_ddim=True
                )
                
                metrics = self.compute_metrics(prediction, hr_image)
                all_metrics.append(metrics)
                
                if i < 5:
                    all_predictions.append(prediction)
                    all_targets.append(hr_image)
                    all_lr_inputs.append(lr_image)
                
            except Exception as e:
                self.logger.error(f"Error processing {hr_file}: {e}")
                continue
        
        if ema is not None:
            ema.restore()
        
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
                if values:
                    aggregated_metrics[f'{key}_mean'] = np.mean(values)
                    aggregated_metrics[f'{key}_std'] = np.std(values)
        
        return {
            'aggregated_metrics': aggregated_metrics,
            'individual_metrics': all_metrics,
            'sample_predictions': all_predictions,
            'sample_targets': all_targets,
            'sample_lr_inputs': all_lr_inputs
        }
    
    def plot_results(self, results: Dict[str, Any]):
        """Create comprehensive result plots."""
        predictions = results['sample_predictions']
        targets = results['sample_targets']
        lr_inputs = results['sample_lr_inputs']
        
        if not predictions:
            self.logger.warning("No predictions to plot")
            return
        
        n_samples = min(3, len(predictions))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # LR input
            im1 = axes[i, 0].imshow(lr_inputs[i], cmap='viridis', aspect='auto')
            axes[i, 0].set_title('LR Input')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
            
            # DDPM prediction
            im2 = axes[i, 1].imshow(predictions[i], cmap='viridis', aspect='auto')
            axes[i, 1].set_title('DDPM Prediction')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
            
            # HR target
            im3 = axes[i, 2].imshow(targets[i], cmap='viridis', aspect='auto')
            axes[i, 2].set_title('HR Target')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
            
            # Difference
            diff = predictions[i] - targets[i]
            vmax = np.abs(diff).max()
            im4 = axes[i, 3].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            axes[i, 3].set_title('Difference (Pred - Target)')
            axes[i, 3].axis('off')
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.results_dir, 'evaluation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# -------------------------
# Benchmarking Function
# -------------------------
def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 50):
    """Benchmark dataloader performance."""
    print(f"Benchmarking dataloader with {num_batches} batches...")
    
    start_time = time.time()
    for i, (lr_batch, hr_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
        _ = lr_batch.mean()
        _ = hr_batch.mean()
    
    end_time = time.time()
    total_time = end_time - start_time
    batches_per_sec = num_batches / total_time
    
    print(f"Dataloader speed: {batches_per_sec:.2f} batches/sec")
    print(f"Estimated full epoch time: {len(dataloader) / batches_per_sec / 60:.1f} minutes")
    
    return batches_per_sec

# -------------------------
# Main Pipeline
# -------------------------
def main():
    """Main training and evaluation pipeline with optimizations."""
    # Apply performance optimizations first
    optimize_pytorch_settings()
    
    # Parse path arguments
    import argparse
    parser = argparse.ArgumentParser(description="DDPM climate downscaling training")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hr-dir", type=str, default=None, help="Path to MSWX 0.1 deg HR directory")
    parser.add_argument("--lr-dir", type=str, default=None, help="Path to 1.0 deg LR directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    config = Config(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        results_dir=args.results_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    device = setup_device()
    
    # Setup logging
    logger = setup_logging(config.results_dir)
    logger.info("Starting Optimized DDPM Climate Downscaling Pipeline")
    
    # Save configuration
    config.save_config(os.path.join(config.results_dir, 'config.json'))
    
    try:
        # Data preparation
        logger.info("Preparing data...")
        all_pairs = list_pairs(config.hr_dir, config.lr_dir)
        if not all_pairs:
            raise RuntimeError("No matching file pairs found.")
        
        train_pairs, val_pairs, test_pairs = split_pairs_by_fraction(
            all_pairs, config.train_fraction, config.val_fraction
        )
        
        logger.info(f"Data split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")
        
        # Climatology
        climatology_manager = ClimatologyManager(config)
        mean_map, std_map, perc_maps = climatology_manager.get_or_compute_climatology(train_pairs)
        normalizer = EfficientNormalizer(mean_map, std_map, method='robust')
        
        # Datasets and dataloaders with caching
        logger.info("Creating optimized datasets...")
        train_dataset = CachedPatchDataset(train_pairs, normalizer, config, augment=True)
        val_dataset = CachedPatchDataset(val_pairs, normalizer, config, augment=False)
        
        train_loader = create_optimized_dataloader(train_dataset, config, shuffle=True)
        val_loader = create_optimized_dataloader(val_dataset, config, shuffle=False)
        
        # Benchmark dataloader performance
        logger.info("Benchmarking dataloader performance...")
        benchmark_dataloader(train_loader, num_batches=20)
        
        # Model and scheduler
        logger.info("Initializing optimized model and scheduler...")
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.t_steps, 
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=device
        )
        
        model = EnhancedUNet(
            in_channels=2, 
            out_channels=1, 
            time_dim=config.time_dim,
            base_dim=config.base_dim
        )
        
        # Apply optimizations to model
        model = setup_optimized_model(model, config, device)
        
        # Training with optimized trainer
        trainer = OptimizedTrainer(model, config, device, logger)
        trainer.train(train_loader, val_loader, noise_scheduler)
        
        # Evaluation
        if test_pairs:
            logger.info("Starting evaluation...")
            
            checkpoint_path = os.path.join(config.results_dir, 'best_model.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model_state = checkpoint['model_state_dict']
                
                if hasattr(model, '_orig_mod'):
                    model._orig_mod.load_state_dict(model_state)
                else:
                    model.load_state_dict(model_state)
                
                ema = None
                if config.use_ema and 'ema_state_dict' in checkpoint:
                    ema = EMA(model, decay=config.ema_decay)
                    ema.load_state_dict(checkpoint['ema_state_dict'])
                
                logger.info("Loaded best model for evaluation")
            
            evaluator = ClimateEvaluator(config, logger)
            results = evaluator.evaluate_test_set(
                model, test_pairs, normalizer, noise_scheduler, device, ema
            )
            
            # Save results
            results_file = os.path.join(config.results_dir, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json_results = {
                    'aggregated_metrics': results['aggregated_metrics'],
                    'individual_metrics': results['individual_metrics']
                }
                json.dump(json_results, f, indent=2, default=str)
            
            evaluator.plot_results(results)
            
            logger.info(f"Evaluation completed. Results saved to {config.results_dir}")
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(trainer.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(trainer.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        if trainer.val_losses:
            best_epoch = np.argmin(trainer.val_losses)
            plt.plot(trainer.val_losses, 'o-', alpha=0.8)
            plt.plot(best_epoch, trainer.val_losses[best_epoch], 'r*', markersize=15, label='Best')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # Performance metrics plot
        plt.text(0.1, 0.8, f"Final Train Loss: {trainer.train_losses[-1]:.5f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"Best Val Loss: {trainer.best_val_loss:.5f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"Total Epochs: {len(trainer.train_losses)}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f"Batch Size: {config.batch_size}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"Cached Data: {config.cache_data}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f"Mixed Precision: {config.mixed_precision}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f"Model Compiled: {config.compile_model}", transform=plt.gca().transAxes)
        plt.title('Training Configuration')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.results_dir, 'training_curves.png'), dpi=300)
        plt.close()
        
        logger.info("Optimized pipeline completed successfully!")
        
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()