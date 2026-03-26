# Annitia

Deep survival analysis for MASLD (Metabolic dysfunction-Associated Steatotic Liver Disease) using Mamba State Space Models.

[![Build](https://img.shields.io/badge/build-CMake-blue)](CMakeLists.txt)
[![Language](https://img.shields.io/badge/language-C11-blue.svg)](src/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**Annitia** is a high-performance C implementation of a dual-output survival model that predicts:
- **Hepatic event risk** — progression to severe liver outcomes
- **Death risk** — all-cause mortality

The model processes irregular time-series clinical data using **Mamba State Space Models** (via [k-mamba](https://github.com/goldensam777/k-mamba)) with:
- Selective state space layers for long-range dependencies
- Cox proportional hazards loss + differentiable ranking loss
- C-index evaluation for censored survival data

## Features

- **Architecture**: Mamba SSM with feature projection, temporal encoding, and dual survival heads
- **Input**: 18 clinical features (12 dynamic + 6 static) across up to 22 visits
- **Training**: Combined Cox + ranking loss with Adam optimizer
- **Evaluation**: Concordance index (C-index) for both hepatic and death endpoints
- **Data pipeline**: CSV → binary preprocessing for fast loading
- **Backend**: [optimatrix](https://github.com/goldensam777/optimatrix) (AVX2 assembly kernels via k-mamba submodule) + OpenBLAS for BLAS operations

## Requirements

- **Compiler**: GCC ≥ 11 or Clang ≥ 14
- **Build**: CMake ≥ 3.18, NASM ≥ 2.15 (for AVX2 kernels)
- **Math**: OpenBLAS
- **Optional**: CUDA ≥ 12.0 (for GPU acceleration)

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/goldensam777/annitia.kali.git
cd annitia.kali
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Build

```bash
# CPU-only build
cmake -B build -DANNITIA_BUILD_TESTS=ON
cmake --build build -j

# With CUDA support (optional)
cmake -B build -DANNITIA_BUILD_CUDA=ON -DANNITIA_BUILD_TESTS=ON
cmake --build build -j
```

### 3. Run Tests

```bash
ctest --test-dir build
```

### 4. Train a Model

```bash
# Preprocess CSV data to binary format
./build/annitia_preprocess data/train.csv data/train.bin data/norm.bin

# Train
./build/annitia_train \
    --train data/train.bin \
    --val data/val.bin \
    --out checkpoint.bin \
    --dim 64 --state 16 --layers 2 \
    --epochs 50 --batch 32 \
    --lr 1e-3 --wd 1e-4
```

### 5. Generate Predictions

```bash
./build/annitia_predict \
    --model checkpoint.bin \
    --data data/test.bin \
    --ids data/test_ids.bin \
    --out submission.csv
```

## CLI Reference

### annitia_preprocess

Converts CSV to optimized binary format for training.

```bash
./annitia_preprocess <input.csv> <output.bin> <norm.bin>
```

### annitia_train

| Flag | Description | Default |
|------|-------------|---------|
| `--train` | Training data (.bin) | required |
| `--val` | Validation data (.bin) | required |
| `--out` | Checkpoint output path | `checkpoint.bin` |
| `--dim D` | Model dimension | 64 |
| `--state N` | SSM state size | 16 |
| `--layers L` | Number of MambaBlocks | 2 |
| `--mimo R` | MIMO rank (0/1 = SISO) | 1 |
| `--epochs E` | Training epochs | 50 |
| `--batch B` | Batch size | 32 |
| `--lr` | Learning rate | 1e-3 |
| `--wd` | Weight decay | 1e-4 |
| `--seed` | Random seed | 42 |

### annitia_predict

```bash
./annitia_predict --model <checkpoint.bin> --data <test.bin> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Trained checkpoint | required |
| `--data` | Test data (.bin) | required |
| `--ids` | Patient IDs file | none |
| `--out` | Output CSV path | `submission.csv` |

## Data Format

### Input Features (18 total)

**Dynamic features** (12, measured at each visit):
- BMI, ALT, AST, Bilirubin, Cholesterol, GGT
- Fasting glucose, Platelets, Triglycerides
- Aixplorer, FibroTest, FibroScan

**Static features** (6, repeated at each timestep):
- Gender, T2DM, Hypertension, Dyslipidaemia
- Bariatric surgery, Bariatric surgery age

### Binary Format

The `.bin` files use a custom binary format with:
- Magic header (`MASL` = `0x4D41534Cu`)
- Normalized float32 features `[N, T, F]`
- Binary masks for missing values
- Time gaps between visits
- Survival targets (time + event indicators)

## Architecture

```
Input: [B, T, 18] — clinical features
    ↓
Feature Projection: Linear(18 → D)
    ↓
Temporal Encoding: time_gap → Linear(1 → D) → add to features
    ↓
MambaBlocks × L: Selective SSM layers (from k-mamba)
    ↓
Mean Pooling over time
    ↓
Dual Survival Heads:
    ├── Linear(D → 1) → risk_hepatic
    └── Linear(D → 1) → risk_death
```

## Loss Functions

- **Cox Loss**: Partial likelihood for proportional hazards
- **Ranking Loss**: Differentiable proxy for C-index
- **Combined**: `α × cox + (1-α) × ranking` (α = 0.5 default)

## Scoring

Final score: `0.7 × C-index_hepatic + 0.3 × C-index_death`

## Project Structure

```
annitia.kali/
├── include/
│   └── annitia.h           # Public API
├── src/
│   ├── annitia.c            # Model implementation
│   ├── survival_loss.c      # Cox + ranking loss
│   ├── masld_data.c         # Data loading
│   ├── mask_utils.c         # Masking utilities
│   ├── csv_parser.c         # CSV → binary converter
│   ├── train_main.c         # Training CLI
│   └── predict_main.c       # Inference CLI
├── k-mamba/                 # Git submodule (Mamba SSM library)
├── data/                    # Training/test data (binary)
├── tests/                   # Unit tests
├── CMakeLists.txt           # Build configuration
└── README.md
```

## Dependencies

- **[k-mamba](https://github.com/goldensam777/k-mamba)** — Mamba State Space Models with ND wavefront scan
  - Includes **[optimatrix](https://github.com/goldensam777/optimatrix)** submodule — AVX2 assembly kernels (GEMM, conv, activations, optimizers)
- **[OpenBLAS](https://www.openblas.net/)** — BLAS operations
- (Optional) **CUDA** — GPU kernels via optimatrix CUDA backend

## Citation

```bibtex
@software{annitia2024,
  author = {YEVI, Mawuli Peniel Samuel},
  title = {Annitia: Mamba-based Survival Analysis for MASLD},
  year = {2024},
  url = {https://github.com/goldensam777/annitia.kali}
}
```

## Author

**YEVI Mawuli Peniel Samuel** — IFRI-UAC (Bénin)

*"Optima, immo absoluta perfectio..."*

## License

MIT License — See [LICENSE](LICENSE) file for details.
