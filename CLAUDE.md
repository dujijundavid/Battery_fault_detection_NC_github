# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **official implementation** of **DyAD (Dynamic Variational Autoencoder)** for battery fault detection, published in Nature Communications. The project implements multiple anomaly detection methods for electric vehicle battery fault detection using charging time-series data.

**Core Methods Implemented:**
- **DyAD** - The main method using bidirectional RNN encoder with conditional VAE decoder
- **AutoEncoder + SVDD** - Traditional baseline methods
- **LSTM-AE** - Recurrent autoencoder baseline
- **GDN** - Graph deviation network baseline
- **Gaussian Process** - Non-parametric baseline

## Repository Structure

```
Battery_fault_detection_NC_github/
├── dyad/                      # DyAD: Main method implementation
├── baselines/                 # Comparison/baseline methods
│   ├── auto_encoder_svdd/ # AutoEncoder + Deep SVDD
│   ├── lstm_ae/           # LSTM Autoencoder
│   ├── gdn/               # Graph Deviation Network
│   └── gaussian_process/  # Gaussian Process
├── data/                      # Dataset
│   ├── battery_brand*/     # Raw data by brand
│   ├── label/             # Labels
│   └── splits/           # Five-fold cross-validation splits
├── evaluation/                # Evaluation notebooks (AUROC calculation)
├── docs/                      # Comprehensive documentation
│   ├── tutorials/        # Step-by-step tutorials
│   ├── reference/        # Technical reference
│   └── domain/           # Domain knowledge
├── databricks.yml            # Databricks configuration
├── requirement.txt           # Python dependencies (CUDA 10.2)
└── requirements_relaxed.txt  # Alternative dependencies (flexible TF versions)
```

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
```
**Output:** Generates `five_fold_utils/all_car_dict.npz.npy` and `ind_odd_dict*.npz.npy` files

### Training DyAD (Main Method)

```bash
cd dyad

# Single fold training
python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num 0

# Complete five-fold cross-validation
for fold in {0..4}; do
    python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num $fold
done
```

**Brand switching:** Currently hardcoded to brand 1. To use other brands, manually modify `ind_ood_car_dict_path` in `model/dataset.py`:
- Brand 2: `../data/splits/ind_odd_dict2.npz.npy`
- Brand 3: `../data/splits/ind_odd_dict3.npz.npy`
- All brands: `../data/splits/all_car_dict.npz.npy`

### Training Comparison Methods

#### AutoEncoder + Deep SVDD

```bash
cd baselines/auto_encoder_svdd

# Train AutoEncoder
python traditional_methods.py --method auto_encoder --normalize --fold_num 0

# Train Deep SVDD
python traditional_methods.py --method deepsvdd --normalize --fold_num 0
```

#### LSTM-AE

```bash
cd baselines/lstm_ae

# Single fold training
python main.py configs/config_lstm_ae_battery_0.json

# Five-fold cross-validation (run 5 times with different config files)
# Config files: config_lstm_ae_battery_{0..4}.json
```

#### GDN (Graph Deviation Network)

```bash
cd baselines/gdn

# Train on brand 1, fold 0, 20 epochs
bash run_battery.sh 3 battery 1 0 20
```

#### Gaussian Process

```bash
cd baselines/gaussian_process

# Train on brand 1, fold 0
python gaussian_process.py 3 battery 1 0
```

### Evaluation

**Note:** Evaluation is done via Jupyter notebooks in the `evaluation/` directory. Each method has corresponding evaluation notebooks:

- `dyad_eval_fivefold-threshold.ipynb` - DyAD robust scores
- `dyad_eval_fivefold-threshold_no.ipynb` - DyAD average scores
- `traditional_methods_eval-threshold.ipynb` - AE+SVDD robust scores
- `traditional_methods_eval-threshold_no.ipynb` - AE+SVDD average scores
- `lstmad_eval_fivefold-threshold.ipynb` - LSTM-AE robust scores
- `lstmad_eval_fivefold-threshold_no.ipynb` - LSTM-AE average scores
- `gdn_eval_five_fold-threshold.ipynb` - GDN robust scores
- `gdn_eval_five_fold-threshold_no.ipynb` - GDN average scores
- `gaussian_process_nature_version_robust.ipynb` - GP robust scores
- `gaussian_process_nature_version_robust-no.ipynb` - GP average scores

**Important:** Before running evaluation notebooks, update the data/model paths to match the new directory structure.

## Code Architecture

### DyAD Module Structure

```
dyad/
├── main_five_fold.py          # Entry point: orchestrates train → extract → evaluate
├── train.py                    # Training class with KL annealing
├── extract.py                  # Feature extraction from trained model
├── evaluate.py                 # Anomaly evaluation using reconstruction error
├── utils.py                    # Utilities (to_var, normalization)
├── model/
│   ├── dynamic_vae.py         # DynamicVAE model definition
│   ├── dataset.py             # Five-fold cross-validation data loader
│   └── tasks.py               # Feature selection per battery brand
└── model_params_battery_brand*.json  # Hyperparameter configs per brand
```

### Key Architectural Patterns

**1. Five-Fold Cross-Validation**
- Normal (IND) samples split into 5 folds
- Train: 4/5 of IND samples
- Test: 1/5 of IND + ALL OOD (abnormal) samples
- Fold selection controlled by `fold_num` parameter (0-4)
- Split definitions stored in `data/splits/` as `ind_odd_dict*.npz.npy`

**2. Multi-Brand Support**
- Each battery brand (1/2/3) has separate configs
- To switch brands, modify `ind_ood_car_dict_path` in `model/dataset.py`

**3. DynamicVAE Model Flow**
```
Input (batch, seq, features)
    ↓
Encoder Filter (selects features depending on brand)
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
Reconstruction (batch, seq, output features)
```

**4. Feature Selection by Brand**
- **Brand 1**: 7 encoder features, 2 decoder conditions, 5 outputs
- **Brand 2**: 7 encoder features, 4 decoder conditions, 3 outputs
- **Brand 3**: 6 encoder features, 2 decoder conditions, 4 outputs
- Defined in `model/tasks.py` via `BatteryBrand1Task`, `BatteryBrand2Task`, `BatteryBrand3Task`

### Loss Function Components

1. **NLL Loss** - Reconstruction error (MSE between input and output)
2. **KL Divergence** - Regularizes latent space (with annealing)
3. **Label Loss** - Auxiliary mileage prediction task

**Total Loss = NLL + β(t)·KL + Label**

**KL Annealing** prevents posterior collapse:
- Linear: `kl_weight = anneal0 * min(1, step / x0)`
- Logistic: `kl_weight = anneal0 / (1 + exp(-k*(step - x0)))`

## Data Format

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
dyad_vae_save/YYYY-MM-DD-HH-MM-SS_fold{num}/
├── model/
│   └── model.torch           # Trained model weights
├── feature/                # Training set extracted features
├── mean/                   # Test set extracted features
├── loss/                   # Loss curve plots
└── result/
    └── test_segment_scores.csv  # car_id, label, rec_error
```

## Databricks Integration

- **Configuration:** `databricks.yml` at repository root
- **Setup:** See `infrastructure/databricks/SETUP_GPU_CLUSTER.md`
- **GPU Testing:** See `infrastructure/databricks/GPU_test.ipynb`
- **Troubleshooting:** See `infrastructure/databricks/TROUBLESHOOTING.md`

## Important Implementation Notes

1. **GPU Requirements**: Code expects CUDA. For CPU-only, set `CUDA_VISIBLE_DEVICES=""` environment variable
2. **Version Constraints**: Requires specific PyTorch 1.5.1 + CUDA 10.2 for torch-geometric compatibility
3. **Path Configuration**: All paths in config files use relative paths from the method's directory
4. **Brand Selection**: Currently hardcoded to brand 1; modify dataset.py for other brands
5. **Variable Length Handling**: Model supports `pack_padded_sequence` for variable-length sequences

## Documentation

- **Quick Start:** [docs/QUICKSTART.md](QUICKSTART.md)
- **Tutorials:** [docs/tutorials/](tutorials/)
- **Architecture Reference:** [docs/reference/Architecture_Reference.md](reference/Architecture_Reference.md)
- **Training Guide:** [docs/reference/Training_and_Evaluation.md](reference/Training_and_Evaluation.md)

## Dependencies

- **Strict (requirement.txt):** TensorFlow 2.6.2 (for CUDA 10.2 compatibility)
- **Relaxed (requirements_relaxed.txt):** TensorFlow >=2.10.0,<2.17 (for newer CUDA versions)

Use `requirements_relaxed.txt` if you have a different CUDA version or want to use newer TensorFlow.
