# DyAD: Dynamic Anomaly Detection for Batteries

This repository contains the refactored code for the DyAD model.

## Directory Structure

- `src/`: Core source code.
    - `model/`: Model definitions (`dynamic_vae.py`, `tasks.py`, `dataset.py`).
    - `train.py`: Training script.
    - `extract.py`: Feature extraction script.
    - `evaluate.py`: Evaluation script.
    - `utils.py`: Utility functions.
- `pipeline/`: Pipeline scripts.
    - `evaluate_fivefold.py`: Script for five-fold cross-validation evaluation.
- `notebooks/`: Jupyter notebooks for visualization and exploration.
- `data/`: Directory for input data (place your `.npy` and `.csv` files here).
- `docs/`: Documentation.

## Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup
Place your data files in the `data/` directory.
Required files:
- `all_car_dict.npz.npy`
- `ind_odd_dict1.npz.npy`
- Label CSVs in `data/battery_brand*/label/`

### Training
Run the training script from the root directory:
```bash
python -m src.train --config_path params.json
```

### Evaluation
Run the evaluation pipeline:
```bash
python pipeline/evaluate_fivefold.py --data_dir data --results_dir results
```

## Refactoring Notes
- Core code has been moved to `src/`.
- Imports have been updated to use absolute imports from the `src` package.
- Data paths in `dataset.py` now default to `data/`.
