# ANNITIA — Prédiction de survie MASLD par K-Mamba SSM

**Challenge ANNITIA par TRUSTII / ICAN** — Prédire la progression de la maladie stéatosique hépatique métabolique (MASLD) à partir de données longitudinales cliniques.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Language: C11](https://img.shields.io/badge/Language-C11-blue.svg)](src/)
[![Build: CMake](https://img.shields.io/badge/Build-CMake-blue.svg)](CMakeLists.txt)

**Auteur :** YEVI Mawuli Peniel Samuel — IFRI-UAC (Bénin)

---

## Résultats

| Modèle | OOF C-hep | OOF C-dth | Score OOF |
|--------|-----------|-----------|-----------|
| Cox ElasticNet (baseline) | ~0.77 | ~0.72 | ~0.75 |
| XGBoost Cox multi-seed | 0.8851 | 0.8760 | 0.8823 |
| K-Mamba SSM mimo4 100ep | 0.8152 | 0.9193 | 0.8464 |
| **K-Mamba SSM 3-seed 200ep** | **0.8865** | **0.9284** | **0.8991** ✅ |

**Métrique :** `Score = 0.7 × C-index_hépatique + 0.3 × C-index_décès`

---

## Architecture

Ce dépôt implémente deux modèles complémentaires :

### 1. XGBoost Cox (features temporelles enrichies)

Chaque patient est représenté par **162 features** extraites de ses trajectoires longitudinales :

| Statistique | Description |
|-------------|-------------|
| `last`, `first` | Valeur à la dernière / première visite |
| `max`, `min`, `mean`, `std` | Statistiques globales |
| `slope` | Tendance linéaire (OLS) sur tout le suivi |
| `delta` | Variation absolue : dernière − première valeur |
| `slope_recent` | Tendance sur la 2ème moitié des visites |
| `accel` | Accélération : moyenne(2ème moitié) − moyenne(1ère moitié) |
| `obs_rate` | Taux de visites avec mesure renseignée |

**Ratios cliniques validés :** FIB-4, De Ritis (AST/ALT), GGT/ALT, FibroScan×FibroTest.

Entraînement : `survival:cox` XGBoost, 5-fold stratifié, 5 seeds → predictions moyennées.

### 2. K-Mamba SSM (trajectoires temporelles brutes)

Réseau neuronal séquentiel en **C pur** avec rétropropagation end-to-end, basé sur [k-mamba](https://github.com/goldensam777/k-mamba).

```
Patient [T ≤ 22 visites, 18 features]
         ↓
  W_feat [18 → 64]  +  W_time [1 → 64]   (encodage temporel)
         ↓
  hidden [T, 64]
         ↓
  MambaBlock #1  (état=16, MIMO rank=4)
         ↓
  MambaBlock #2
         ↓
  Pooling sur la dernière visite valide  (masque de visite)
         ↓
  W_hepatic [64 → 1]    W_death [64 → 1]
         ↓
  risk_hepatic,  risk_death
```

**Loss :** `L = α × ranking_loss + (1−α) × cox_loss`
- `α_hépatique = 0.2` (événements rares à 3.8%)
- `α_décès = 0.5`

**Entraînement multi-seed :** 3 seeds (42, 123, 777) × 5 folds × 200 epochs → moyenne rank-normalisée.

| Seed | OOF Score |
|------|-----------|
| 42 | 0.8422 |
| 123 | 0.8720 |
| 777 | 0.8531 |
| **Moyenne 3 seeds** | **0.8991** |

La variance inter-seed est importante → toujours moyenner plusieurs seeds.

---

## Structure du dépôt

```
annitia.kali/
├── k-mamba/                     # Submodule — bibliothèque SSM (C)
│   ├── src/mamba_block.c        # MambaBlock MIMO, scan, MUON optimizer
│   └── include/kmamba.h         # API publique
├── include/
│   └── annitia.h                # AnnitiaModel, AnnitiaConfig, SurvivalBatch
├── src/
│   ├── annitia.c                # Modèle principal (projection + SSM + têtes survie)
│   ├── survival_loss.c          # Cox partial likelihood + pairwise ranking loss
│   ├── masld_data.c             # Chargeur binaire MASL
│   └── mask_utils.c             # Gestion des valeurs manquantes
├── annitia/                     # Package Python (interface haut niveau)
│   ├── model.py                 # AnnitiaModel (wraps C via ctypes)
│   ├── dataset.py               # MasldDataset
│   ├── train.py                 # Boucle d'entraînement
│   ├── ssm_kfold.py             # K-fold OOF SSM
│   ├── xgb.py                   # XGBoost Cox pipeline
│   ├── metrics.py               # C-index
│   └── optimizer.py             # GP score surface 4D
├── data/
│   ├── train.bin                # 1253 patients (format MASL binaire)
│   ├── test.bin                 # 423 patients
│   ├── test_ssm_hep_200ep_3s.npy  # Prédictions SSM 3-seed (hépatique)
│   ├── test_ssm_dth_200ep_3s.npy  # Prédictions SSM 3-seed (décès)
│   ├── oof_ssm_hep_200ep_3s.npy   # OOF SSM 3-seed (train)
│   └── oof_ssm_dth_200ep_3s.npy
├── notebook_annitia.ipynb       # Notebook de soumission (auto-contenu)
├── RESULTS.md                   # Journal détaillé des expériences
└── CMakeLists.txt
```

---

## Utilisation

### Prérequis

```bash
sudo apt install cmake gcc libopenblas-dev nasm
pip install numpy pandas xgboost scikit-learn scipy matplotlib
```

### 1. Cloner avec sous-modules

```bash
git clone --recursive https://github.com/goldensam777/annitia.kali
cd annitia.kali
```

### 2. Compiler la bibliothèque C

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 3. Installer le package Python

```bash
pip install -e .
```

### 4. Prétraiter les données CSV

```bash
python -c "
from annitia.dataset import preprocess_csv
preprocess_csv(
    '/path/to/DB-train.csv',
    '/path/to/DB-test.csv',
    train_out='data/train.bin',
    test_out='data/test.bin',
    norm_out='data/norm.bin',
)
"
```

### 5. Entraîner le SSM (K-fold, multi-seed)

```bash
# Seed 42
python -c "
from annitia.ssm_kfold import train_ssm_kfold
import numpy as np
r = train_ssm_kfold('data/train.bin', 'data/test.bin',
                     epochs=200, seed=42, mimo_rank=4)
np.save('data/test_ssm_hep_200ep_s42.npy', r['test_hep'])
np.save('data/test_ssm_dth_200ep_s42.npy', r['test_dth'])
print(f'OOF score: {r[\"score\"]:.4f}')
"

# Idem pour seeds 123 et 777, puis moyenner
```

### 6. Notebook de soumission

Le notebook `notebook_annitia.ipynb` est auto-contenu :
- Clone ce dépôt via `subprocess.run(['git', 'clone', '--recursive', ...])`
- Charge les prédictions SSM pré-calculées (`data/test_ssm_*_200ep_3s.npy`)
- Entraîne XGBoost en direct (~5 minutes)
- Génère `submission_final.csv`

---

## Format binaire MASL

Les fichiers `.bin` utilisent un format binaire personnalisé :

```
Header : magic(u32=0x4D41534C) n_patients(u32) T(u32) F(u32) version(u32) pad(u32)
Par patient :
  features   [T × F]  float32   (zéro si manquant)
  mask       [T × F]  float32   (1=valide, 0=manquant)
  time_gaps  [T]      float32   (jours entre visites)
  n_visits   int32
  time_hep   float32
  event_hep  uint8 + pad uint8
  time_dth   float32
  event_dth  uint8 + pad[3] uint8
```

---

## Bug critique corrigé

**`k-mamba/src/mamba_block.c` ligne 354** (2026-03-26) :

```c
// AVANT (incorrect) — heap corruption avec mimo_rank > 1 :
block->b_B = calloc(config->state_size, sizeof(float));

// APRÈS (correct) :
block->b_B = calloc(config->state_size * R, sizeof(float));
```

Ce bug provoquait une corruption de heap avec `mimo_rank=4` (R=4, accès indices 0..63 sur 0..15 alloués). Tous les résultats `mimo_rank=4` avant le 2026-03-26 sont invalides.

---

## Citation

```bibtex
@software{annitia2026,
  author  = {YEVI, Mawuli Peniel Samuel},
  title   = {Annitia: K-Mamba State Space Model for MASLD Survival Prediction},
  year    = {2026},
  url     = {https://github.com/goldensam777/annitia.kali},
  note    = {Challenge ANNITIA par TRUSTII/ICAN}
}
```

---

*"Optima, immo absoluta perfectio."*

## Licence

MIT License — voir [LICENSE](LICENSE).
