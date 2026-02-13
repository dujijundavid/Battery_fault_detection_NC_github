# DyAD: Battery Fault Detection via Dynamical Deep Learning

**Official code repository for our Nature Communications paper** - "Realistic fault detection of Li-ion battery via dynamical deep learning approach"

## Quick Links

| Documentation | Description |
|--------------|-------------|
| [CLAUDE.md](CLAUDE.md) | Complete project documentation for AI assistants |
| [docs/](docs/) | Tutorials, API reference, and technical guides |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Quick start guide |

## Repository Structure

```
Battery_fault_detection_NC_github/
├── dyad/                    # DyAD: Main method (Nature Communications)
├── baselines/                 # Comparison/baseline methods
│   ├── auto_encoder_svdd/ # AutoEncoder + Deep SVDD
│   ├── lstm_ae/           # LSTM Autoencoder
│   ├── gdn/               # Graph Deviation Network
│   └── gaussian_process/  # Gaussian Process
├── data/                      # Dataset
│   ├── battery_brand*/     # Raw data by brand
│   ├── label/             # Labels
│   └── splits/           # Train/test splits
├── evaluation/                # Evaluation notebooks (AUROC calculation)
├── docs/                      # Documentation
│   ├── tutorials/        # Step-by-step tutorials
│   ├── reference/        # Technical reference
│   └── domain/           # Domain knowledge
├── infrastructure/            # Development tools
│   └── databricks/        # Databricks GPU setup
├── databricks.yml            # Databricks configuration
├── requirement.txt           # Python dependencies (CUDA 10.2)
└── requirements_relaxed.txt  # Alternative dependencies (flexible TF versions)
```

## Environment Setup

```bash
# Requirements: CUDA 10.2, Python 3.6
pip install torch==1.5.1
pip install torch-geometric==1.5.0
pip install -r requirement.txt
```

## Training Examples

```bash
# DyAD (main method)
cd dyad
python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num 0

# Baseline: AutoEncoder + SVDD
cd baselines/auto_encoder_svdd
python traditional_methods.py --method auto_encoder --normalize --fold_num 0

# Baseline: LSTM-AE
cd baselines/lstm_ae
python main.py configs/config_lstm_ae_battery_0.json

# Baseline: GDN
cd baselines/gdn
bash run_battery.sh 3 battery 1 0 20

# Baseline: Gaussian Process
cd baselines/gaussian_process
python gaussian_process.py 3 battery 1 0
```

## Citation

If you use this code, please cite:
> **D. Duan et al., "Realistic fault detection of Li-ion battery via dynamical deep learning approach", *Nature Communications*, 2024
