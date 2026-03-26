# Paper Draft — Annitia : Mamba SSM pour la Survie MASLD

**Statut :** AVANT soumission Trustii (26 mars 2026)
**Version :** 0.1 — pré-résultats officiels

---

## Titre provisoire

> *Longitudinal Survival Analysis for MASLD using Mamba State Space Models:
> a Lightweight C Implementation with Gaussian Process Performance Modeling*

---

## Abstract (provisoire)

Nous présentons **Annitia**, un modèle de survie dual pour la MASLD
(Metabolic dysfunction-Associated Steatotic Liver Disease) combinant :

1. Un **State Space Model (SSM) de type Mamba** implémenté en C pur avec
   backpropagation end-to-end, entraîné sur les trajectoires longitudinales brutes
   (jusqu'à 22 visites, 18 features dynamiques et statiques).

2. Un **XGBoost Cox PH** avec feature engineering temporal enrichi (162 features :
   statistiques par visite, ratios cliniques validés FIB-4/De Ritis/GGT-ALT,
   patterns de données manquantes, accélération temporelle).

3. Un **optimiseur de surface de score** par Gaussian Process qui modélise
   `Score(w_ssm, epochs, mimo_rank)` et localise analytiquement le θ* optimal
   ainsi que le plafond de performance atteignable dans l'espace de paramètres courant.

**Score OOF estimé :** 0.882 (C-index hépatique 0.885, C-index décès 0.876)
**Score Trustii officiel :** ??? (à remplir après soumission)

---

## 1. Contexte et Problème

### 1.1 Challenge ANNITIA (TRUSTII / ICAN)

- **Données :** 1 676 patients synthétiques MASLD
  - Train : 1 253 patients, 47 événements hépatiques (3.8%), 76 décès (6.1%)
  - Test  : 423 patients
- **Features :** 12 dynamiques × 22 visites + 6 statiques = 18 features/timestep
- **Métrique :** `Score = 0.7 × C-index_hepatic + 0.3 × C-index_death`
- **Difficulté principale :** rareté extrême des événements hépatiques (EPV ≈ 1.3)

### 1.2 Pourquoi un SSM ?

Les données MASLD sont des séquences temporelles **irrégulières** :
- Intervalles entre visites variables (time gaps)
- Données manquantes structurelles (toutes les features ne sont pas mesurées à chaque visite)
- Dépendances à long terme entre les visites (fibrose progressive)

Un SSM traite naturellement ces trois caractéristiques via :
- Encodage des time gaps dans la projection temporelle
- Masque de visite pour le pooling (last-valid-timestep)
- Mémoire sélective des états cachés (mécanisme Mamba)

---

## 2. Architecture Annitia

### 2.1 Vue d'ensemble

```
Patient [T ≤ 22 visites, F = 18 features]
    ↓
W_feat [F=18 → dim=64]  +  W_time [1 → dim=64]  (projection + encodage temporel)
hidden [T, dim=64]
    ↓
MambaBlock #1  (state=16, MIMO rank=4, 9 696 params)
    ↓
MambaBlock #2
    ↓
Last-valid-timestep pooling  (masque de visite)
    ↓
W_hepatic [dim=64 → 1]    W_death [dim=64 → 1]
    ↓
risk_hepatic,  risk_death
```

**Paramètres totaux :** 19 393 (mimo_rank=4) — délibérément minimal pour CPU hospitalier

### 2.2 Loss de survie combinée

```
L = α × L_ranking + (1-α) × L_cox

L_cox     = -Σᵢ [riskᵢ - log(Σⱼ:tⱼ≥tᵢ exp(riskⱼ))]
L_ranking = Σᵢⱼ concordants sigmoid(riskⱼ - riskᵢ)
```

- `α_hepatic = 0.2` (événements rares → plus de ranking loss)
- `α_death = 0.5`

### 2.3 Implémentation C pur

- Backend : k-mamba (SSM) + optimatrix (AVX2 GEMM)
- Backpropagation end-to-end via `mamba_backward()` en ordre inverse des couches
- Pas de framework ML — déployable en environnement hospitalier sans GPU

---

## 3. Feature Engineering Temporal

### 3.1 Statistiques par feature dynamique (×12 features)

| Statistique | Description |
|---|---|
| `last`, `first` | Valeur dernière/première visite |
| `max`, `min`, `mean`, `std` | Statistiques globales |
| `slope` | Tendance OLS sur tout le suivi |
| `delta` | Variation last − first |
| `slope_recent` | Tendance sur la 2e moitié des visites |
| `accel` | mean(2e moitié) − mean(1ère moitié) |
| `obs_rate` | Proportion de visites avec mesure |

### 3.2 Ratios cliniques MASLD validés

| Feature | Formule | Signification |
|---|---|---|
| **FIB-4** | (âge × AST) / (plt × √ALT) | Marqueur fibrose standard |
| **De Ritis** | AST / ALT | Fibrose avancée si > 1 |
| **GGT/ALT** | GGT / ALT | Cholestase vs cytolyse |
| **FibroScan×FibroTest** | fibro × ftest | Concordance mesures fibrose |
| **bili×AST** | bili × AST | Insuffisance hépatique |

### 3.3 Total : 162 features (vs 35-44 features flat baseline)

**Top features XGBoost (gain) :**
`follow_up` > `age_v1` > `FibroScan_max` > `AST_ALT_ratio` > `FIB4` >
`BMI_slope` > `FibroScan_accel` > `ALT_slope` > `FibroTest_delta`

---

## 4. Résultats Expérimentaux

### 4.1 Comparaison des modèles (OOF, train split)

| Modèle | C-idx Hép | C-idx Décès | Score | Notes |
|---|---|---|---|---|
| Cox ElasticNet (baseline) | 0.77 | 0.72 | 0.75 | Features flat |
| RSF (baseline) | ~0.91 | ~0.93 | ~0.91 | **Train score — overfit** |
| LightGBM binary | 0.78 | 0.64 | 0.73 | Objectif non adapté |
| CatBoost Cox | 0.711 | 0.745 | 0.721 | Instable |
| SSM K-Mamba (30ep, mimo1) | — | — | 0.782 | Sous-entraîné |
| SSM K-Mamba (50ep, mimo4) | 0.765 | 0.910 | 0.808 | |
| SSM K-Mamba (100ep, mimo4) | **0.824** | **0.882** | **0.842** | |
| **XGBoost Cox multi-seed** | **0.885** | **0.876** | **0.882** | **→ Soumission 1** |
| Ensemble 30%SSM+70%XGB | ??? | ??? | ??? | **→ Soumission 2** |

### 4.2 Score Trustii officiel

| Soumission | Contenu | Score OOF | Score Trustii |
|---|---|---|---|
| submission_1.csv | XGBoost Cox 5 seeds | 0.882 | ??? |
| submission_2.csv | 30% SSM + 70% XGBoost | ~0.866* | ??? |

*estimé par GP, incertitude ±0.021

### 4.3 Progression SSM avec les epochs

| Epochs | mimo_rank | Score OOF | Fold 3 hép | Notes |
|---|---|---|---|---|
| 30 | 1 | 0.782 | 0.653 | Sous-entraîné |
| 50 | 4 | 0.808 | 0.697 | En progression |
| 100 | 4 | **0.842** | **0.896** | Non saturé |
| 150* | 4 | ~0.858 | — | *extrapolation GP |

---

## 5. Optimiseur de Surface de Score (contribution originale)

### 5.1 Formulation

On modélise le score comme une fonction paramétrée :

```
Score : θ = (w_ssm, epochs, mimo_rank) → [0, 1]
```

Fittée par **Gaussian Process** (kernel Matérn ν=2.5) sur les observations OOF.

### 5.2 Résolution {θ : Score(θ) ≥ 0.95}

**Résultat :** ensemble vide dans l'espace courant.

Le gradient `∂Score/∂w_ssm < 0` pour tout w_ssm ∈ [0,1] à epochs=100.
Le θ* optimal trouvé : `(w_ssm=0, epochs=115, mimo_rank=4)` → score estimé 0.883.

**Implication :** le plafond de 0.95 nécessite d'étendre l'espace θ — nouveaux axes requis.

### 5.3 Nouveaux axes pour atteindre 0.95

| Axe | Levier attendu | Justification |
|---|---|---|
| **Scan 2D** (features × temps) | +0.04–0.06 | Capture interactions croisées automatiquement (FIB-4 appris, pas codé) |
| Stacking méta-modèle | +0.01–0.02 | OOF SSM + XGBoost comme features d'un 2e niveau |
| SMOTE survival | +0.01–0.02 | 47 events hépatiques — augmentation ciblée |
| MELD / NAS score | +0.005–0.01 | Features cliniques additionnelles validées |

**Décomposition nécessaire :**
```
Score = 0.7 × C_hep + 0.3 × C_dth ≥ 0.95
→ Si C_dth = 0.93 : C_hep ≥ 0.959  (gap actuel : +0.074)
→ Si C_dth = 0.95 : C_hep ≥ 0.950  (gap actuel : +0.065)
```

---

## 6. Hypothèse Scan 2D

La matrice patient `[T=22 × F=18]` est structurellement identique à une image `[H × W]`.

Un scan 1D (actuel) voit :
```
BMI:  t1 → t2 → ... → t22   (trajectoire temporelle d'une feature)
ALT:  t1 → t2 → ... → t22
...
```

Un scan 2D verrait :
```
[BMI_t1  ALT_t1  AST_t1  ...]
[BMI_t2  ALT_t2  AST_t2  ...]  → scan horizontal ET vertical simultanément
...
```

**Ce que le scan 2D apprendrait automatiquement :**
- FIB-4 = f(âge, AST, plt, ALT) — interaction multi-feature à un instant t
- Accélération = f(feature_t_early, feature_t_late) — interaction temporelle
- Corrélations croisées FibroScan × ALT — progression fibrose vs cytolyse

C'est W2-E1 du projet k-mamba appliqué à la médecine.

---

## 7. Corrélation SSM / XGBoost (signal complémentaire)

Sur le jeu test (423 patients) :

| Endpoint | Corrélation rang SSM↔XGBoost |
|---|---|
| Hépatique | **0.142** |
| Décès | **−0.014** |

Corrélation quasi-nulle = les deux modèles capturent des **signaux orthogonaux**.
Théoriquement optimal pour un ensemble — mais le SSM actuel n'est pas encore assez
précis pour que l'ensemble batte XGBoost seul (OOF SSM 0.842 < OOF XGBoost 0.882).

---

## 8. Limites et Travaux Futurs

| Limite | Impact | Solution envisagée |
|---|---|---|
| EPV = 1.3 (47 events) | SSM converge lentement | SMOTE survival + plus d'epochs |
| Scan 1D seulement | Interactions features non capturées | Scan 2D (k-mamba W2-E1) |
| 1003 patients pour k-fold SSM | OOF sous-estimé | Retrain sur full 1253 |
| GP fitté sur 5 points | Surface incertaine | Chaque soumission = 1 point de plus |

---

## À remplir après soumission

- [ ] Score Trustii submission_1
- [ ] Score Trustii submission_2
- [ ] Diff OOF vs score officiel (overfitting ?)
- [ ] Position dans le classement
- [ ] Mise à jour surface GP avec points réels
- [ ] Nouveau θ* après calibration

---

## Références clés

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)
- FIB-4 Index for Liver Fibrosis (Sterling et al., 2006)
- Concordance Index for Censored Survival (Harrell et al., 1982)
- Gaussian Process for Machine Learning (Rasmussen & Williams, 2006)
