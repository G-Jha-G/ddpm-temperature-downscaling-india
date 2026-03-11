# DDPM Temperature Downscaling — India

Code for the paper:

> Jha, S.K., Gupta, V., et al. (2026). *Not published yet ;)

## What this repo contains

| Path | Purpose |
|---|---|
| `model/train.py` | Model architecture + full training pipeline |
| `inference/inference_synthetic.py` | Run inference on MSWX synthetic test period → Table 2, Table 3 |
| `inference/inference_ensemble.py` | Generate probabilistic ensembles → Figure 8, Figure 9 |
| `inference/inference_cmip6.py` | Apply model to all 10 CMIP6 models across SSPs → Figures 11–13, Supp. Table 6 |
| `analysis/01_cmip6_historical_bias.ipynb` | CMIP6 historical evaluation → Figure 10, Supp. Tables 7A–C |
| `analysis/02_seasonal_bias_maps.ipynb` | Seasonal bias maps → Figure 5, Figure 6 |
| `analysis/03_case_studies_projections.ipynb` | Case studies + SSP projections → Figures 7, 11, 12, 13 |
| `analysis/04_trend_analysis.ipynb` | IMD zone RMSE + trend analysis |

## Architecture

- **Model:** Conditional U-Net with self-attention, channel dimensions [64, 128, 256, 512]
- **Diffusion:** Linear β schedule, T=1000 timesteps (β₁=1×10⁻⁴, β_T=0.02)
- **Sampling:** DDIM, 50 steps (deterministic by default; stochastic for ensemble)
- **Inference:** Gaussian patch stitching (128×128 patches, stride 64, σ=0.3)
- **Normalisation:** Pointwise climatological standardisation from training data
- **EMA:** Exponential Moving Average weights used exclusively at inference

## Data

| Dataset | Role | Source |
|---|---|---|
| MSWX 0.1° daily | Training reference + test target | http://www.gloh2o.org/mswx/ |
| MSWX 1.0° (degraded) | Training input (synthetic) | Generated from MSWX via block averaging |
| CMIP6 (10 models, QDM-corrected) | Inference input for projections | https://esgf-node.llnl.gov/ |
| Downscaled dataset (output) | 0.1° daily, 2015–2100, 10 models × 3 SSPs | Zenodo: [INSERT DOI] |

## Hardware

All experiments ran on:
- AMD EPYC 7742, 330 GB RAM
- 2× NVIDIA RTX A5000, 24 GB VRAM each
- Single GPU used for training and inference

## Setup

```bash
git clone https://github.com/[yourusername]/ddpm-temperature-downscaling-india.git
cd ddpm-temperature-downscaling-india
pip install -r requirements.txt
```

Python 3.9+ required. PyTorch 2.0+ recommended for `torch.compile` support.

## Training

Edit the path arguments in `model/train.py` (Config class, USER CONFIGURATION section), then:

```bash
python model/train.py \
    --hr-dir /path/to/MSWX_HR \
    --lr-dir /path/to/MSWX_LR_1deg \
    --results-dir /path/to/output \
    --gpu 0 \
    --epochs 100
```

Training produces:
- `best_model.pt` — best checkpoint (EMA weights + model weights + config)
- `latest_model.pt` — latest checkpoint (for resuming)
- `climatology_maps.nc` — pointwise mean/std maps used for normalisation
- `config.json` — full training configuration

To resume from a checkpoint:
```bash
python model/train.py --hr-dir ... --lr-dir ... --results-dir ... \
    --resume-checkpoint /path/to/output/latest_model.pt
```

## Inference

### Synthetic test period (reproduces Table 2 & 3)

Edit the USER CONFIGURATION section in `inference/inference_synthetic.py`, then:
```bash
python inference/inference_synthetic.py
```

### Ensemble uncertainty (reproduces Figure 8 & 9)

Edit paths in `inference/inference_ensemble.py`. The four case study dates from the paper are pre-configured as `MANUAL_EVENT_DATES`. Then:
```bash
python inference/inference_ensemble.py
```

### CMIP6 future projections (reproduces Figures 11–13)

Edit `BatchInferenceConfig` in `inference/inference_cmip6.py`. Set `target_model` to one of the 10 CMIP6 model names. Run once per model:
```bash
# Add model/ to path so train.py is importable
export PYTHONPATH=$(pwd)/model:$PYTHONPATH
python inference/inference_cmip6.py
```

## Reproducibility

All experiments use `random seed 42`. Set in `Config.sample_stats` and passed to all numpy/torch random calls. The ensemble seeds follow the sequence `{42, 142, 242, ..., 42+(N-1)×100}`.

## Citation

If you use this code, please cite:

```bibtex
@article{jha2026ddpm,
  title     = {A Generative {AI} Framework for High-Fidelity Probabilistic Downscaling
               of {CMIP6} Daily Temperature Projections over India},
  author    = {Jha, Shailesh Kumar and Gupta, Vivek and others},
  journal   = {Communications Earth \& Environment},
  year      = {2026},
  doi       = {INSERT PAPER DOI}
}
```

## License

MIT — see [LICENSE](LICENSE).
