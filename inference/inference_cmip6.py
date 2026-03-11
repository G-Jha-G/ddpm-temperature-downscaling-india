"""
Batch inference script: apply trained DDPM to all 10 QDM-corrected CMIP6 models.

Takes bias-corrected CMIP6 inputs (output of QDM pipeline at 1.0 deg) and produces
downscaled 0.1 deg daily temperature NetCDF files for 2015-2100 across SSP2-4.5,
SSP3-7.0, and SSP5-8.5. Reproduces the multi-model pipeline results in Section 3.9
and Supplementary Table 6 of the paper.

The script imports model components from model/train.py. Ensure train.py is on
the Python path (or in the same directory) before running.

Usage:
    python inference_cmip6.py

    To run in the background:
        nohup python -u inference_cmip6.py > inference_cmip6.log 2>&1 &

Configuration:
    Edit the BatchInferenceConfig class below:
        training_results_dir  — directory containing best_model.pt, config.json,
                                 and climatology_maps.nc (output of training)
        input_dir             — directory containing QDM bias-corrected 1.0 deg
                                 CMIP6 NetCDF files (all models, all SSPs)
        output_dir            — directory where downscaled 0.1 deg files are written
        target_model          — CMIP6 model name to filter (process one model at a time;
                                 run once per model to produce all 10 outputs)

Supported models (Table 1 of paper):
    CESM2, CESM2-WACCM, EC-Earth3-AerChem, EC-Earth3-CC,
    HadGEM3-GC31-LL, HadGEM3-GC31-MM, INM-CM4-8, INM-CM5-0,
    MPI-ESM1-2-HR, NorESM2-MM
"""

# nohup python -u inference_cmip6.py > inference_cmip6.log 2>&1 &

import os
import sys
import json
import warnings
import logging
import psutil
import gc
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_inference_cesm2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Import model components from train.py ---
# train.py must be importable. Either run this script from the repo root
# (where model/train.py exists) or add the model/ directory to PYTHONPATH:
#   export PYTHONPATH=/path/to/repo/model:$PYTHONPATH
try:
    from train import (
        Config,
        EnhancedUNet,
        NoiseScheduler,
        EfficientNormalizer,
        ClimatologyManager,
        setup_device,
        setup_optimized_model,
        nearest_fill_nan_fast,
        extract_patches_improved,
        stitch_patches_gaussian,
        sample_ddim,
        EMA,
        ArrayType,
        sorted_alphanumeric,
    )
    logger.info("Successfully imported DDPM components from train.py")
except ImportError as e:
    logger.error(f"Failed to import from train.py: {e}")
    logger.error("Ensure model/train.py is on your PYTHONPATH.")
    sys.exit(1)

class BatchInferenceConfig:
    """Configuration for batch inference with validation."""
    
    def __init__(self):
        # =====================================================================
        # USER CONFIGURATION — set these three paths before running
        # =====================================================================
        # training_results_dir: directory containing best_model.pt, config.json,
        #                        and climatology_maps.nc (output of training run)
        # input_dir:            directory with QDM bias-corrected 1.0 deg CMIP6 files
        #                        naming: YYYY-MM-DD_{MODEL_NAME}.nc
        # output_dir:           directory where downscaled 0.1 deg files are written
        # =====================================================================
        self.training_results_dir = None  # e.g. "/results/DDPM_enhanced_results_mse"
        self.input_dir = None             # e.g. "/data/Top10_QDM_BC_CMIP6/All_in_1_0p1"
        self.output_dir = None            # e.g. "/results/DDPM_downscaled_CESM2"
        
        # Performance settings
        self.inference_batch_size = 32  # Will be auto-adjusted based on GPU memory
        self.auto_adjust_batch_size = True
        self.max_memory_usage = 0.8  # Use up to 80% of GPU memory
        
        # Processing options
        # ref_nc_path: a single MSWX 0.1 deg file used to copy reference lat/lon coords
        # e.g. "/data/MSWX_Temp_Daily_Ind_HR/1979001.nc"
        # If None, coordinates are calculated from the LR input (less precise).
        self.ref_nc_path = None           # e.g. "/data/MSWX_Temp_Daily_Ind_HR/1979001.nc"
        
        self.skip_existing = True
        self.backup_failed = True
        self.max_retries = 3
        
        # target_model: filter input directory for files ending with _{MODEL_NAME}.nc
        # Change this to the model you want to process. Run once per model.
        # Supported: CESM2, CESM2-WACCM, EC-Earth3-AerChem, EC-Earth3-CC,
        #            HadGEM3-GC31-LL, HadGEM3-GC31-MM, INM-CM4-8, INM-CM5-0,
        #            MPI-ESM1-2-HR, NorESM2-MM
        self.target_model = "CESM2"  # Change to target model name
        
        # Validation
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate all required paths exist."""
        if self.training_results_dir is None:
            raise ValueError("training_results_dir is not set. Edit BatchInferenceConfig.")
        if self.input_dir is None:
            raise ValueError("input_dir is not set. Edit BatchInferenceConfig.")
        if self.output_dir is None:
            raise ValueError("output_dir is not set. Edit BatchInferenceConfig.")
            
        if not Path(self.training_results_dir).exists():
            raise FileNotFoundError(f"Training results directory not found: {self.training_results_dir}")
        
        if not Path(self.input_dir).exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check for required files
        required_files = ['config.json', 'best_model.pt', 'climatology_maps.nc']
        for file in required_files:
            file_path = Path(self.training_results_dir) / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

class MemoryManager:
    """Manages GPU memory and batch size optimization."""
    
    def __init__(self, device: torch.device, max_usage: float = 0.8):
        self.device = device
        self.max_usage = max_usage
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            available = (total_memory - allocated_memory) / (1024**3)
            return available * self.max_usage
        return psutil.virtual_memory().available / (1024**3) * 0.5  # Conservative for CPU
    
    def estimate_optimal_batch_size(self, config: Config, model: torch.nn.Module) -> int:
        """Estimate optimal batch size based on available memory."""
        available_gb = self.get_available_memory()
        
        # Rough estimation: each patch uses ~20MB with mixed precision
        patch_memory_mb = 20 if hasattr(model, 'config') and getattr(model.config, 'mixed_precision', False) else 40
        max_patches = int((available_gb * 1024) / patch_memory_mb)
        
        # Conservative estimate with safety margin
        optimal_batch_size = max(4, min(max_patches // 2, 64))
        
        logger.info(f"Available GPU memory: {available_gb:.1f}GB")
        logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

class BatchInference:
    """Main batch inference class with improved error handling."""
    
    def __init__(self, config: BatchInferenceConfig):
        self.config = config
        self.device = setup_device()
        self.memory_manager = MemoryManager(self.device, config.max_memory_usage)
        
        # Initialize components
        self._load_training_config()
        self._setup_model_and_normalizer()
        
        # Track processing stats
        self.processed_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        self.filtered_files = 0
    
    def _load_training_config(self):
        """Load configuration from training results."""
        config_path = Path(self.config.training_results_dir) / 'config.json'
        logger.info(f"Loading training configuration from {config_path}")
        
        self.train_config = Config.load_config(str(config_path))
        logger.info("Training configuration loaded successfully")
    
    def _setup_model_and_normalizer(self):
        """Initialize model, normalizer, and scheduler."""
        logger.info("Setting up model and normalizer...")
        
        # Load climatology
        climatology_manager = ClimatologyManager(self.train_config)
        mean_map, std_map, _ = climatology_manager.get_or_compute_climatology([])
        self.normalizer = EfficientNormalizer(mean_map, std_map, method='robust')
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=self.train_config.t_steps,
            beta_start=self.train_config.beta_start,
            beta_end=self.train_config.beta_end,
            device=self.device
        )
        
        # Initialize and load model
        model = EnhancedUNet(
            in_channels=2,
            out_channels=1,
            time_dim=self.train_config.time_dim,
            base_dim=self.train_config.base_dim
        )
        
        self.model = setup_optimized_model(model, self.train_config, self.device)
        self._load_model_weights()
        
        # Auto-adjust batch size if enabled
        if self.config.auto_adjust_batch_size:
            self.config.inference_batch_size = self.memory_manager.estimate_optimal_batch_size(
                self.train_config, self.model
            )
        
        logger.info("Model setup completed successfully")
    
    def _load_model_weights(self):
        """Load model weights with validation."""
        model_path = Path(self.config.training_results_dir) / 'best_model.pt'
        logger.info(f"Loading model weights from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'ema_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("Loaded EMA model weights")
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded regular model weights")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model weights (direct state dict)")
                
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def _filter_cesm2_files(self, files: List[str]) -> List[str]:
        """Filter files to only include CESM2 model files."""
        cesm2_files = []
        
        for filename in files:
            # Check if filename contains the target model name
            if filename.endswith('_EC-Earth3-CC.nc'): # Change here
                cesm2_files.append(filename)
            else:
                self.filtered_files += 1
        
        logger.info(f"Found {len(cesm2_files)} {self.config.target_model} files out of {len(files)} total files")
        logger.info(f"Filtered out {self.filtered_files} non-{self.config.target_model} files")
        
        return cesm2_files
    
    def _load_cmip_file(self, file_path: Path) -> Tuple[Optional[ArrayType], Optional[str]]:
        """Load CMIP6 NetCDF file and return temperature data."""
        try:
            logger.debug(f"Loading file: {file_path.name}")
            
            with xr.open_dataset(file_path) as ds:
                # Try to find temperature variable
                temp_vars = ['tas', 'temperature', 'T', 'temp']
                var_name = None
                
                for var in temp_vars:
                    if var in ds.variables:
                        var_name = var
                        break
                
                if var_name is None:
                    available_vars = list(ds.data_vars.keys())
                    logger.error(f"No temperature variable found in {file_path.name}. Available variables: {available_vars}")
                    return None, None
                
                # Load data
                data = ds[var_name].values
                
                # Handle different dimensions
                if data.ndim == 4:  # (time, level, lat, lon)
                    data = data[:, 0, :, :]  # Take first level
                elif data.ndim == 3:  # (time, lat, lon)
                    pass  # Already correct format
                elif data.ndim == 2:  # (lat, lon)
                    data = data[np.newaxis, :, :]  # Add time dimension
                else:
                    logger.error(f"Unexpected data dimensions in {file_path.name}: {data.shape}")
                    return None, None
                
                # Take first time step if multiple
                if data.shape[0] > 1:
                    logger.warning(f"Multiple time steps found, using first one for {file_path.name}")
                
                lr_image = data[0, :, :]
                
                # Handle NaN values
                if np.any(np.isnan(lr_image)):
                    logger.info(f"Filling NaN values in {file_path.name}")
                    lr_image = nearest_fill_nan_fast(lr_image)
                
                # Validate data range
                if np.any(lr_image < -100) or np.any(lr_image > 150):
                    logger.warning(f"Unusual temperature range in {file_path.name}: [{lr_image.min():.1f}, {lr_image.max():.1f}]")
                
                return lr_image, var_name
                
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None, None
    
    def _save_netcdf(self, prediction: ArrayType, input_path: Path, output_path: Path, var_name: str):
        """Save prediction as NetCDF file using HARDCODED reference grid."""
        
        # --- 1. REFERENCE COORDINATE FILE ---
        # A single MSWX 0.1 deg file used to copy exact lat/lon coordinate arrays.
        # Set REF_NC_PATH in BatchInferenceConfig or provide via environment variable.
        # This ensures output coordinates match the MSWX reference grid exactly (float32).
        REF_NC_PATH = getattr(self.config, 'ref_nc_path', None)
        
        try:
            # --- 2. Fix Dimensions: Ensure 3D (time, lat, lon) ---
            if prediction.ndim == 2:
                # Add time dimension: (lat, lon) -> (1, lat, lon)
                prediction = prediction[np.newaxis, :, :]

            coords = {}
            use_reference = False

            # --- 3. Fix Coordinates: Copy from Reference ---
            if os.path.exists(REF_NC_PATH):
                with xr.open_dataset(REF_NC_PATH) as ref_ds:
                    # Check if prediction shape matches reference shape
                    # Prediction is (1, lat, lon), Reference is (time, lat, lon)
                    if prediction.shape[1:] == ref_ds['air_temperature'].shape[1:]:
                        coords['lat'] = ref_ds['lat'].values  # Copies exact float32 values
                        coords['lon'] = ref_ds['lon'].values  # Copies exact float32 values
                        use_reference = True
                    else:
                        logger.warning(f"Prediction shape {prediction.shape} does not match Reference {REF_NC_PATH}. Falling back to calculation.")

            # Fallback: Calculate coordinates if reference file is missing or shape mismatches
            if not use_reference:
                with xr.open_dataset(input_path) as original_ds:
                    lat_lr = original_ds['lat'].values
                    lon_lr = original_ds['lon'].values
                    upscale = prediction.shape[1] // len(lat_lr)
                    
                    # Force float32 to minimize precision errors
                    lat_hr = np.linspace(lat_lr.min(), lat_lr.max(), len(lat_lr) * upscale).astype(np.float32)
                    lon_hr = np.linspace(lon_lr.min(), lon_lr.max(), len(lon_lr) * upscale).astype(np.float32)
                    
                    coords['lat'] = lat_hr
                    coords['lon'] = lon_hr

            # --- 4. Handle Time Variable ---
            with xr.open_dataset(input_path) as original_ds:
                if 'time' in original_ds.variables or 'time' in original_ds.coords:
                    t_val = original_ds['time'].values
                    # Ensure time is an array of size 1
                    if np.ndim(t_val) == 0:
                        coords['time'] = np.array([t_val])
                    else:
                        coords['time'] = t_val[0:1] # Take first element as array
                else:
                    coords['time'] = np.array([0]) # Dummy time if missing

            # --- 5. Create Dataset ---
            # Explicitly define dimensions as ['time', 'lat', 'lon']
            hr_ds = xr.Dataset(
                {var_name: (['time', 'lat', 'lon'], prediction)},
                coords=coords
            )
            
            # Metadata
            hr_ds.attrs['source'] = f'DDPM downscaled from {input_path.name}'
            hr_ds.attrs['upscale_factor'] = prediction.shape[1] // len(coords['lat']) if not use_reference else "Ref Grid"
            
            # Save
            hr_ds.to_netcdf(output_path)
            logger.debug(f"Saved prediction: {output_path.name}")
                
        except Exception as e:
            logger.error(f"Error saving {output_path.name}: {e}")
            raise
    
    @torch.no_grad()
    def _predict_image(self, lr_image: ArrayType) -> ArrayType:
        """Predict full HR image using batched inference."""
        self.model.eval()
        
        # Extract patches
        patches_lr, positions = extract_patches_improved(
            lr_image, 
            self.train_config.patch_size, 
            self.train_config.patch_size // 2, 
            ensure_coverage=True
        )
        
        pred_patches = [None] * len(positions)
        num_patches = len(patches_lr)
        batch_size = self.config.inference_batch_size
        
        # Process patches in batches
        pbar = tqdm(
            range(0, num_patches, batch_size),
            desc="    Processing patches",
            leave=False,
            ncols=80
        )
        
        for i in pbar:
            batch_slice = slice(i, min(i + batch_size, num_patches))
            current_patches = patches_lr[batch_slice]
            current_positions = positions[batch_slice]
            
            # Normalize patches
            norm_patches = []
            for patch, (top, left) in zip(current_patches, current_positions):
                lr_norm = self.normalizer.normalize_patch(
                    patch, top, left, self.train_config.patch_size
                )
                lr_norm = np.clip(lr_norm, -5, 5)
                norm_patches.append(torch.from_numpy(lr_norm).float())
            
            # Create batch tensor and predict
            lr_batch = torch.stack(norm_patches).unsqueeze(1).to(self.device)
            
            try:
                hr_pred_batch = sample_ddim(
                    self.model,
                    lr_batch,
                    self.noise_scheduler,
                    self.device,
                    ddim_steps=self.train_config.ddim_steps,
                    progress=False
                )
                
                # Denormalize and store results
                hr_pred_cpu = hr_pred_batch.squeeze(1).cpu().numpy()
                
                for patch_idx in range(len(current_positions)):
                    pred_patches[i + patch_idx] = hr_pred_cpu[patch_idx]
                
                # Clean up GPU memory periodically
                if (i // batch_size) % 10 == 0:
                    self.memory_manager.cleanup()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("GPU out of memory, reducing batch size")
                    self.config.inference_batch_size = max(4, batch_size // 2)
                    raise
                else:
                    raise
        
        # Stitch patches in NORMALIZED space
        sr_normalized = stitch_patches_gaussian(
            pred_patches,
            positions,
            lr_image.shape,
            self.train_config.patch_size,
            sigma_factor=self.train_config.gaussian_sigma_factor,
            edge_fade=True
        )

        # Global denormalization using full-resolution climatology maps
        full_mean = self.normalizer.mean_map.numpy() if torch.is_tensor(self.normalizer.mean_map) else self.normalizer.mean_map
        full_std = self.normalizer.scale_map.numpy() if torch.is_tensor(self.normalizer.scale_map) else self.normalizer.scale_map

        h, w = lr_image.shape
        result = sr_normalized * full_std[:h, :w] + full_mean[:h, :w]

        return result
    
    def process_single_file(self, input_path: Path, output_path: Path) -> bool:
        """Process a single file with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                # Load input file
                lr_image, var_name = self._load_cmip_file(input_path)
                if lr_image is None:
                    return False
                
                # Generate prediction
                prediction = self._predict_image(lr_image)
                
                # Save result
                self._save_netcdf(prediction, input_path, output_path, var_name)
                
                self.processed_files += 1
                return True
                
            except RuntimeError as e:
                if "out of memory" in str(e) and attempt < self.config.max_retries - 1:
                    logger.warning(f"OOM error on attempt {attempt + 1}, retrying with smaller batch")
                    self.memory_manager.cleanup()
                    continue
                else:
                    logger.error(f"Failed to process {input_path.name} after {attempt + 1} attempts: {e}")
                    self.failed_files += 1
                    return False
            except Exception as e:
                logger.error(f"Error processing {input_path.name} (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    self.failed_files += 1
                    return False
        
        return False
    
    def run_batch_inference(self):
        """Main inference loop."""
        logger.info("Starting batch inference process for CESM2 model files")
        
        # Get all input files
        all_files = [
            f for f in os.listdir(self.config.input_dir) 
            if f.endswith('.nc')
        ]
        
        if not all_files:
            logger.error("No .nc files found in input directory")
            return
        
        logger.info(f"Found {len(all_files)} total .nc files in directory")
        
        # Filter for CESM2 files only
        input_files = self._filter_cesm2_files(all_files)
        
        if not input_files:
            logger.error(f"No {self.config.target_model} files found in input directory")
            return
        
        # Sort files alphanumerically
        input_files = sorted_alphanumeric(input_files)
        
        logger.info(f"Processing {len(input_files)} {self.config.target_model} files")
        
        # Process files
        progress_bar = tqdm(input_files, desc=f"Processing {self.config.target_model} files", ncols=100)
        
        for filename in progress_bar:
            input_path = Path(self.config.input_dir) / filename
            output_path = Path(self.config.output_dir) / filename
            
            # Skip existing files if configured
            if self.config.skip_existing and output_path.exists():
                self.skipped_files += 1
                progress_bar.set_postfix({
                    'processed': self.processed_files,
                    'failed': self.failed_files,
                    'skipped': self.skipped_files,
                    'filtered': self.filtered_files
                })
                continue
            
            # Process file
            success = self.process_single_file(input_path, output_path)
            
            # Update progress display
            progress_bar.set_postfix({
                'processed': self.processed_files,
                'failed': self.failed_files,
                'skipped': self.skipped_files,
                'filtered': self.filtered_files
            })
            
            # Backup failed files if configured
            if not success and self.config.backup_failed:
                self._backup_failed_file(input_path)
        
        # Final cleanup
        self.memory_manager.cleanup()
        
        # Print summary
        logger.info("Batch inference completed")
        logger.info(f"Total files in directory: {len(all_files)}")
        logger.info(f"Non-{self.config.target_model} files filtered out: {self.filtered_files}")
        logger.info(f"Successfully processed: {self.processed_files}")
        logger.info(f"Failed: {self.failed_files}")
        logger.info(f"Skipped: {self.skipped_files}")
        logger.info(f"Output directory: {self.config.output_dir}")
    
    def _backup_failed_file(self, input_path: Path):
        """Backup failed files for later inspection."""
        backup_dir = Path(self.config.output_dir) / "failed_files"
        backup_dir.mkdir(exist_ok=True)
        
        try:
            import shutil
            shutil.copy2(input_path, backup_dir / input_path.name)
            logger.info(f"Backed up failed file: {input_path.name}")
        except Exception as e:
            logger.warning(f"Failed to backup {input_path.name}: {e}")

def main():
    """Main entry point."""
    try:
        # Initialize configuration
        config = BatchInferenceConfig()
        
        # Validate that input directory is set
        if not config.input_dir:
            logger.error("Please update the input_dir path in the BatchInferenceConfig class")
            logger.error("Set the input_dir variable to your directory containing CMIP6 model files")
            sys.exit(1)
        
        # Create and run inference
        inference = BatchInference(config)
        inference.run_batch_inference()
        
        logger.info("CESM2 batch inference pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()