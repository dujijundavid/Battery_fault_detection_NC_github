# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **DyAD (Dynamic Variational Autoencoder)** for battery fault detection, published in Nature Communications. The project implements multiple anomaly detection methods for electric vehicle battery fault detection using charging time-series data.

**Core Methods Implemented:**
- **DyAD** - The main method using bidirectional RNN encoder with conditional VAE decoder
- **AutoEncoder + SVDD** - Traditional baseline methods
- **LSTM-AE** - Recurrent autoencoder baseline
- **GDN** - Graph deviation network baseline

## Common Commands

### Environment Setup

```bash
# Install dependencies (requires CUDA 10.2, Python 3.6)
pip install torch==1.5.1
pip install torch-geometric==1.5.0
pip install -r requirement.txt
```

### Data Preparation

```bash
cd data
# Run the notebook to generate five-fold split files
jupyter notebook five_fold_train_test_split.ipynb
# Generates: five_fold_utils/all_car_dict.npz.npy and ind_odd_dict*.npz.npy
```

### Training DyAD (Main Method)

```bash
cd DyAD
# Single fold training
python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num 0

# Complete five-fold cross-validation
for fold in {0..4}; do
    python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num $fold
done
```

### Training Comparison Methods

```bash
# AutoEncoder
cd AE_and_SVDD
python traditional_methods.py --method auto_encoder --normalize --fold_num 0

# Deep SVDD
python traditional_methods.py --method deepsvdd --normalize --fold_num 0

# LSTM-AE
cd Recurrent-Autoencoder-modify
python main.py configs/config_lstm_ae_battery_0.json

# GDN
cd GDN_battery
bash run_battery.sh 3 battery 1 0 20
```

### Evaluation

```bash
# AUROC calculation is done via Jupyter notebooks in notebooks/ directory
# Each method has corresponding evaluation notebooks:
# - dyad_eval_fivefold-threshold.ipynb (robust scores)
# - dyad_eval_fivefold-threshold_no.ipynb (average scores)
# Note: Update paths in notebooks before running
```

## Code Architecture

### DyAD Module Structure

```
DyAD/
├── main_five_fold.py          # Entry point: orchestrates train → extract → evaluate
├── train.py                  # Training loop with KL annealing
├── extract.py                # Feature extraction from trained model
├── evaluate.py               # Anomaly scoring using reconstruction error
├── utils.py                  # Utility functions (to_var, normalization)
├── model/
│   ├── dynamic_vae.py         # DynamicVAE model definition
│   ├── dataset.py             # Five-fold cross-validation data loader
│   └── tasks.py              # Feature selection per battery brand
└── model_params_battery_brand*.json  # Hyperparameter configs
```

### Key Architectural Patterns

**1. Five-Fold Cross-Validation**
- Normal (IND) samples split into 5 folds
- Train: 4/5 of IND samples
- Test: 1/5 of IND + ALL OOD (abnormal) samples
- Fold selection controlled by `fold_num` parameter (0-4)
- Split definitions stored in `five_fold_utils/ind_odd_dict*.npz.npy`

**2. Multi-Brand Support**
- Each battery brand (1/2/3) has separate config and data split files
- To switch brands, modify `ind_ood_car_dict_path` in:
  - `DyAD/model/dataset.py`
  - Corresponding files in other method directories

**3. DynamicVAE Model Flow**
```
Input (batch, seq, 7 features)
    ↓
Encoder Filter (select 6-7 features depending on brand)
    ↓
Bidirectional RNN Encoder
    ↓
Hidden States → [μ (mean), logσ² (log variance)]
    ↓
Reparameterization: z = μ + σ·ε (training) or z = μ (inference)
    ↓
Decoder Initial Hidden from z
    ↓
Conditional Decoder (SOC + Current as conditions)
    ↓
Reconstruction (batch, seq, 3-5 output features)
```

**4. Feature Selection by Brand**
- Brand 1: 7 encoder features (soc, current, min_temp, max_single_volt, min_single_volt, volt, max_temp), 2 decoder conditions, 5 outputs
- Brand 2: 7 encoder features, 4 decoder conditions, 3 outputs
- Brand 3: 6 encoder features, 2 decoder conditions, 4 outputs
- Defined in `tasks.py` via `BatterybrandaTask`, `BatterybrandbTask`, `EvTask`

### Loss Function Components

1. **NLL Loss** - Reconstruction error (MSE between input and output)
2. **KL Divergence** - Regularizes latent space (with annealing)
3. **Label Loss** - Auxiliary mileage prediction task

Total Loss = NLL + β(t)·KL + Label

**KL Annealing** prevents posterior collapse:
- Linear: `kl_weight = anneal0 * min(1, step / x0)`
- Logistic: `kl_weight = anneal0 / (1 + exp(-k*(step - x0)))`

### Data Format

Each `.pkl` file contains a tuple:
1. **Time series tensor** - Charging data with shape (seq_len, num_features)
2. **Metadata dict** - Contains:
   - `label`: Fault label (0=normal, 1=abnormal)
   - `car_number`: Vehicle identifier
   - `charge_segment_number`: Which charging session
   - `mileage`: Vehicle mileage (used for auxiliary task)

Column names stored in `data/battery_brand*/column.pkl`

## Output Structure

After training, outputs are saved to timestamped directories:

```
dyad_vae_save/YYYY-MM-DD-HH-MM-SS_fold0/
├── model/
│   ├── model.torch           # Trained model weights
│   └── model_params.json     # Full configuration snapshot
├── feature/                # Training set extracted features
├── mean/                   # Test set extracted features
├── loss/                   # Loss curve plots
└── result/
    └── test_segment_scores.csv  # car_id, label, rec_error
```

### Key Files for Modification

| File | When to Modify |
|------|----------------|
| `model_params_battery_brand*.json` | Tuning hyperparameters (latent_size, hidden_size, learning_rate, loss weights) |
| `model/dynamic_vae.py` | Changing model architecture (new layers, different RNN types) |
| `model/tasks.py` | Adding new battery brands or modifying feature selection |
| `model/dataset.py` | Changing data loading logic or fold splitting |
| `train.py` | Modifying loss function or training loop behavior |

## Important Implementation Notes

1. **GPU Requirements**: Code expects CUDA. For CPU-only, set `CUDA_VISIBLE_DEVICES=""` environment variable
2. **Version Constraints**: Requires specific PyTorch 1.5.1 + CUDA 10.2 for torch-geometric compatibility
3. **Path Configuration**: All paths in config files use relative paths from the method's directory
4. **Brand Selection**: Must manually change dictionary paths in code when switching brands (not parameterized)
5. **Variable Length Handling**: Model supports `pack_padded_sequence` for variable-length sequences

## Documentation

- **Quick Start**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Architecture Reference**: [docs/reference/Architecture_Reference.md](docs/reference/Architecture_Reference.md)
- **Training Guide**: [docs/reference/Training_and_Evaluation.md](docs/reference/Training_and_Evaluation.md)
- **Tutorial Series**: [docs/tutorials/](docs/tutorials/) - Step-by-step learning guides
- **Index**: [docs/INDEX.md](docs/INDEX.md) - Complete documentation navigation
