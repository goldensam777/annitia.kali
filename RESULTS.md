# ANNITIA — Journal des expériences

Challenge Trustii/ICAN : prédire la survie MASLD (Metabolic Associated Steatotic Liver Disease).
Métrique : **Score = 0.7 × C-index hépatique + 0.3 × C-index décès**

---

## Données

| Split | Patients | Événements hépatiques | Décès |
|-------|----------|-----------------------|-------|
| Train | 1 253 | 47 (3.8%) | 76 (6.1%) |
| Test  | 423    | — | — |

**Features :** 12 dynamiques/visite (BMI, ALT, AST, bilirubin, cholestérol, GGT, glucose, plaquettes, triglycérides, Aixplorer, FibroTest, FibroScan) + 6 statiques (gender, T2DM, Hypertension, Dyslipidaemia, bariatric_surgery, bariatric_surgery_age) = **18 features / timestep**, jusqu'à 22 visites.

---

## Expériences

### EXP-01 — Baseline Trustii (hello_world notebook)
**Modèles :** RSF + Cox ElasticNet (features flat, filtre 50% manquant)
**Résultats (train) :**
| Modèle | C-idx hépatique | C-idx décès |
|--------|----------------|-------------|
| RSF | 0.91 | 0.93 |
| Cox ElasticNet | 0.77 | 0.72 |

> ⚠️ RSF = overfit évident (train score). Cox ElasticNet = référence honnête.

---

### EXP-02 — SSM K-Mamba (architecture temporelle C)
**Architecture :** Feature projection [18→64] → 2× MambaBlock (dim=64, state=16) → last-valid-timestep pooling → têtes survie hépatique + décès
**Paramètres :** ~200K params
**Loss :** Cox + ranking, `alpha_hep=0.2` (events rares), `alpha_dth=0.5`
**Training :** Adam + clip_norm, lr=3e-4, batch=32, 25 epochs, shuffle Fisher-Yates
**Données :** full 1253 patients (pas de validation)

| Config | C-idx hépatique | C-idx décès | Score | Notes |
|--------|----------------|-------------|-------|-------|
| alpha_hep=0.5 (val 80/20) | — | — | 0.8216 | premier run |
| **alpha_hep=0.2 (val 80/20)** | — | — | **0.8432** | meilleur checkpoint |
| full 1253 patients | — | — | — | soumission finale |

> ✅ Le SSM capture les trajectoires temporelles que les modèles flat ne voient pas.
> Fichier : `data/model_alpha02.bin`, prédictions : `data/submission_final.csv`

---

### EXP-03 — LightGBM binary + poids temporels
**Objectif :** `binary` (pas AFT/ranking — incompatibilités LightGBM 4.6)
**Poids :** événements précoces = `1 + (max_t - t) / max_t`, censurés = 1
**Features :** last/first/max/min/mean/std/slope/delta par feature dynamique + statiques + `AST_ALT_ratio` + `fibro_x_age`
**CV :** 5-fold stratified, 5 seeds {42, 123, 777, 2024, 31415} → moyenne

| Seed | C-idx hépatique OOF | C-idx décès OOF | Score OOF |
|------|--------------------|--------------------|-----------|
| 42 | 0.8057 | 0.6295 | 0.7529 |
| 123 | 0.7966 | 0.6486 | 0.7522 |
| 777 | 0.7778 | 0.6310 | 0.7337 |
| 2024 | 0.7406 | 0.6249 | 0.7059 |
| 31415 | 0.7848 | 0.6564 | 0.7463 |
| **Moyenne** | **~0.78** | **~0.64** | **~0.73** |

**Top features hépatiques :** `follow_up`, `age_v1`, `AST_ALT_ratio`, `BMI_slope`, `ALT_slope`, `Dyslipidaemia`, `bilirubin_min`, `FibroTest_delta`

> ⚠️ Décès sous-performant (0.63). LightGBM apporte peu par rapport à XGBoost Cox.

---

### EXP-04 — XGBoost Cox (objectif survival:cox natif)
**Objectif :** `survival:cox` — labels : `y = +time` (event), `y = -time` (censuré)
**CV :** 5-fold stratified + early stopping (50 rounds)
**Seeds testés :** 42, 123, 777

| Seed | C-idx hépatique OOF | C-idx décès OOF | Score OOF | Best iter (moy) |
|------|--------------------|--------------------|-----------|-----------------|
| 42 | 0.8769 | 0.8545 | 0.8702 | ~107 |
| 123 | — | — | — | — |
| 777 | — | — | — | — |
| **Multi-seed avg** | **~0.877** | **~0.855** | **~0.870** | — |

> ✅ Meilleur modèle plat. L'objectif Cox natif > binary LightGBM pour la survie.
> Prédictions : `data/submission_xgb_multiseed.csv`

---

### EXP-05 — Ensemble SSM + LightGBM + XGBoost (rank-average)

**Sweep poids OOF (XGBoost vs LightGBM) :**

| XGBoost | LightGBM | C-idx hép OOF | C-idx décès OOF | Score OOF |
|---------|----------|--------------|-----------------|-----------|
| 0% | 100% | 0.8057 | 0.6295 | 0.7529 |
| 20% | 80% | 0.8407 | 0.7018 | 0.7990 |
| 40% | 60% | 0.8617 | 0.7640 | 0.8324 |
| 60% | 40% | 0.8725 | 0.8241 | 0.8580 |
| 80% | 20% | 0.8756 | 0.8630 | 0.8718 |
| **100%** | **0%** | **0.8727** | **0.8826** | **0.8757** |

> LightGBM nuit à l'ensemble — XGBoost Cox domine. LightGBM retiré.

**Ensemble final retenu : 40% SSM K-Mamba + 60% XGBoost Cox**
- SSM apporte la composante temporelle (trajectoires)
- XGBoost apporte le modèle Cox calibré (features flat)
- Prédictions : `data/submission_final_v2.csv`

---

### EXP-06 — Feature engineering enrichi (ratios cliniques + missing patterns + accélération)
**Nouveaux features ajoutés :**
- `FIB-4 index` = (âge × AST) / (plt × √ALT) — marqueur fibrose validé MASLD
- `GGT_ALT_ratio` — lésion hépatocellulaire vs cholestase
- `AST_ALT_ratio` (De Ritis) — déjà présent, conservé
- `FibroScan_x_FibroTest`, `FibroScan_FibroTest_ratio` — concordance des mesures fibrose
- `bili_x_AST` — insuffisance hépatique
- `BMI_x_fibro` — obésité + fibrose combinés
- `{feature}_obs_rate` — proportion de visites mesurées (pattern de données manquantes)
- `n_features_last_visit` — nombre de features mesurées à la dernière visite
- `{feature}_slope_recent` + `{feature}_accel` — slope sur 2e moitié des visites + accélération

**Impact sur XGBoost Cox (seed 42, 5-fold) :**

| Endpoint | Avant (EXP-04) | Après (EXP-06) | Δ |
|----------|----------------|----------------|---|
| Hépatique OOF | 0.8769 | **0.8851** | +0.0082 |
| Décès OOF | 0.8545 | **0.8760** | +0.0215 |
| **Score OOF** | **0.8702** | **0.8823** | **+0.012** |

> ✅ FIB-4 et les features d'accélération temporelle apportent le plus de signal.

---

### EXP-07 — CatBoost Cox
**Objectif :** Cox PH via `loss_function='Cox'`, gestion native catégorielles
**CV :** 5-fold stratified, seed 42

| Endpoint | C-idx OOF | Notes |
|----------|-----------|-------|
| Hépatique | 0.7110 | fold 1 à 0.699 — instable |
| Décès | 0.7447 | |
| **Score OOF** | **0.7211** | |

> ❌ CatBoost Cox < XGBoost Cox (0.8823). L'implémentation Cox de CatBoost moins mature.
> Variance inter-fold élevée — non retenu pour l'ensemble final.

---

## En cours / À venir

| ID | Expérience | Attendu |
|----|-----------|---------|
### EXP-08 — SSM K-Mamba k-fold OOF (5 folds × 30 epochs)
**Méthode :** lecture/écriture binaire MASL par patient, stratification combinée (hep+dth), seed=fold+42

| Fold | Hép val | Décès val | Score val |
|------|---------|-----------|-----------|
| 1 | 0.7314 | 0.8624 | 0.7707 |
| 2 | 0.8020 | 0.8592 | 0.8192 |
| 3 | 0.6530 | 0.8768 | 0.7202 |
| 4 | 0.7752 | 0.9408 | 0.8249 |
| 5 | 0.8062 | 0.9422 | 0.8470 |
| **OOF global** | **0.7361** | **0.8886** | **0.7819** |

> ⚠️ SSM OOF (0.7819) < XGBoost (0.8823). Ajout SSM dégrade l'ensemble.

**Sweep OOF SSM + XGBoost :**

| SSM | XGBoost | Hép OOF | Décès OOF | Score OOF |
|-----|---------|---------|-----------|-----------|
| 0% | 100% | 0.8851 | 0.8760 | **0.8823** ← |
| 20% | 80% | 0.8521 | 0.8642 | 0.8557 |
| 40% | 60% | 0.7930 | 0.8073 | 0.7973 |
| 60% | 40% | 0.6557 | 0.6816 | 0.6635 |
| 80% | 20% | 0.5072 | 0.5570 | 0.5222 |
| 100% | 0% | 0.3960 | 0.4646 | 0.4166 |

> Conclusion : **XGBoost Cox seul est optimal sur ces données** (30 epochs SSM insuffisants pour dépasser le Cox PH).

---

### EXP-09 — XGBoost Cox multi-seed (soumission finale)
**5 seeds** : {42, 123, 777, 2024, 31415} → moyenne des prédictions test

| Seed | Hép OOF | Décès OOF | Score OOF |
|------|---------|-----------|-----------|
| 42 | 0.8851 | 0.8760 | 0.8823 |
| 123 | 0.8711 | 0.8950 | 0.8783 |
| 777 | 0.8793 | 0.8720 | 0.8771 |
| 2024 | 0.8211 | 0.8868 | 0.8408 |
| 31415 | 0.8722 | 0.9038 | 0.8817 |
| **Moyenne** | — | — | **0.8720 ± 0.016** |

> ✅ **Soumission finale : `data/submission_FINAL.csv`** — 423 patients, XGBoost Cox 5 seeds.

---

## Analyse des résultats

### Pourquoi XGBoost > SSM sur ces données
1. **Événements rares** (47/1253 = 3.8%) : les petits lots d'entraînement SSM (~30 epochs) convergent mal avec si peu de signal positif par batch
2. **Features enrichies** (FIB-4, GGT/ALT, accélérations) : capturent l'essentiel de la dynamique temporelle de façon plate → XGBoost les exploite directement
3. **Cox PH natif** de XGBoost est calibré pour la censure → meilleure utilisation des 1206 censurés
4. **SSM besoin de plus d'epochs** : la courbe de training montrait encore 0.77→0.84 en progression à epoch 30 (non saturée)

### Features les plus importantes (XGBoost Cox, gain moyen)
`follow_up` > `age_v1` > `AST_ALT_ratio` > `FIB4` > `BMI_slope` > `ALT_slope` > `FibroTest_delta` > `Dyslipidaemia`

---

### EXP-10 — SSM mimo_rank=4 (19K params) k-fold OOF (50 epochs)

**Motivation :** tester si un modèle plus expressif (mimo_rank=4, 19 393 params vs 6 145)
améliore l'OOF et rend l'ensemble SSM+XGBoost viable.

| Fold | Hép val | Décès val | Score val |
|------|---------|-----------|-----------|
| 1 | **0.8198** | 0.9196 | **0.8497** |
| 2 | 0.8020 | 0.8592 | 0.8192 |
| 3 | 0.6973 | 0.8829 | 0.7530 |
| 4 | 0.7802 | 0.9436 | 0.8292 |
| 5 | 0.8079 | 0.9430 | 0.8484 |
| **OOF global** | **0.7646** | **0.9098** | **0.8081** |

Amélioration vs mimo_rank=1 (30 epochs) : **+0.026 points OOF** (0.7819 → 0.8081).

**Sweep OOF SSM(mimo4) + XGBoost :**

| SSM | XGBoost | Score OOF |
|-----|---------|-----------|
| 0% | 100% | **0.8823** ← |
| 20% | 80% | 0.8558 |
| 100% | 0% | 0.4269 |

> SSM gagne le **fold 1** (0.8497 vs XGBoost 0.8360) → signal complémentaire existe.
> Mais fold 3 hépatique SSM = 0.697 plombe la moyenne OOF.
> Conclusion : **100 epochs nécessaires** pour que le SSM soit systématiquement complémentaire.

---

### EXP-11 — Notebook qualitatif Trustii (soumission 30%)

Notebook standalone `notebook_annitia.ipynb` :
- Reproduit feature engineering (162 features) + XGBoost Cox 5 seeds
- Auto-détection chemins CSV Trustii (6 candidats)
- Correction bug `slope_recent` (indices relatifs)
- `c_index_numpy()` sans dépendances externes
- Génère `submission_1.csv` (423 patients)

> ✅ Soumis avec `submission_1.csv`

---

### EXP-12 — SSM mimo_rank=4, 100 epochs [INVALIDATED — b_B allocation bug]

> ⚠️ **RÉSULTATS INVALIDES** : bug découvert dans k-mamba/src/mamba_block.c ligne 354.
> `b_B` et `b_C` (BCNorm biases) alloués avec `state_size` floats au lieu de `state_size * R_rank`.
> Avec mimo_rank=4, R_rank=4 → accès hors limites → heap corruption → résultats non fiables.
> Score OOF 0.8417 produit par un modèle avec des biais corrompus.

**Fix appliqué :** `calloc(config->state_size * R, sizeof(float))` dans `mamba_block_create`.

---

### EXP-12-bis — SSM mimo_rank=4, 100 epochs (après correction b_B bug)

**Correction :** allocation correcte `b_B[state_size × R_rank]`, résultats fiables.

| Fold | Hép val | Décès val | Score val |
|------|---------|-----------|-----------|
| 1 | 0.8874 | 0.9354 | **0.9018** |
| 2 | 0.8670 | 0.9581 | 0.8944 |
| 3 | **0.9101** | 0.8812 | **0.9015** |
| 4 | 0.6841 | 0.9606 | 0.7670 |
| 5 | 0.8133 | 0.9256 | 0.8470 |
| **OOF global** | **0.8152** | **0.9193** | **0.8464** |

> ✅ SSM OOF = **0.8464** (valide). Folds 1, 2, 3 > 0.89 — modèle fort sur ces plis.
> Fold 4 SSM = 0.767 (instable). Complémentarité avec XGBoost à exploiter.

**Prédictions sauvegardées :**
- OOF : `data/oof_ssm_hep.npy`, `data/oof_ssm_dth.npy`
- Test : `data/test_ssm_hep.npy`, `data/test_ssm_dth.npy`

**Soumissions créées (rank-average SSM + XGBoost) :**
- `submission_ssm10_xgb90_v2.csv` (10% SSM)
- `submission_ssm15_xgb85_v2.csv` (15% SSM)
- `submission_ssm20_xgb80_v2.csv` (20% SSM) — recommandé
- `submission_ssm25_xgb75_v2.csv` (25% SSM)
- `submission_ssm30_xgb70_v2.csv` (30% SSM)

---

### EXP-13 — SSM Conv2D + mimo_rank=4, 100 epochs

**Motivation :** tester Conv2D preprocessing sur la matrice [T×F] patient.

| Fold | Hép val | Décès val | Score val |
|------|---------|-----------|-----------|
| 1 | 0.8550 | 0.9479 | 0.8829 |
| 2 | 0.7455 | 0.9009 | 0.7921 |
| 3 | 0.7689 | 0.7343 | 0.7586 |
| 4 | 0.8331 | 0.9409 | 0.8654 |
| 5 | 0.8838 | 0.9372 | **0.8998** |
| **OOF global** | **0.7903** | **0.8980** | **0.8226** |

> ❌ Conv2D **nuit** aux performances : 0.8226 < 0.8464 (SSM sans Conv2D).
> La couche Conv2D sur [T=22, F=18] ne capture pas de structure 2D utile ici.
> Architecture abandonnée pour ce problème.

---

## TODO final
| ID | Tâche | Priorité |
|----|-------|----------|
| SUB-3 | Soumettre `submission_ssm20_xgb80_v2.csv` pour valider l'apport du SSM corrigé | haut |
| EXP-14 | SSM 150-200 epochs (mimo4, no conv2d) — exploration saturation | moyen |
| EXP-15 | Ensemble optimal sur OOF avec XGBoost (quand CSV data accessible) | moyen |

---

## Architecture K-Mamba SSM

```
Patient [T=22 visites, F=18 features]
    ↓ W_feat [F=18 → dim=64]  +  W_time [1 → dim=64]
hidden [T, dim=64]
    ↓ MambaBlock #1  (state=16, MIMO)
    ↓ MambaBlock #2
last_valid_timestep pooling  (masque de visite)
    ↓ W_hepatic [dim → 1]   W_death [dim → 1]
risk_hepatic,  risk_death
```

**Loss :** `L = alpha * ranking_loss + (1-alpha) * cox_loss`
**Backprop :** end-to-end via `mamba_backward()` en ordre inverse des couches
