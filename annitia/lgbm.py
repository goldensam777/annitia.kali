"""
lgbm.py — Feature engineering temporel + LightGBM survival pour ANNITIA.

Extrait depuis le CSV wide :
  - Dernière valeur observée (last)
  - Tendance linéaire (slope sur toutes les visites)
  - Nombre de visites (n_visits)
  - Valeur max, min, std par feature dynamique
  - Features statiques

Puis entraîne deux modèles LightGBM (ranking survival) :
  - hepatic : événement hépatique
  - death   : décès
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Features dynamiques dans l'ordre
DYN_PREFIXES = [
    "BMI_v", "alt_v", "ast_v", "bilirubin_v", "chol_v", "ggt_v",
    "gluc_fast_v", "plt_v", "triglyc_v",
    "aixp_aix_result_BM_3_v", "fibrotest_BM_2_v", "fibs_stiffness_med_BM_1_v",
]
DYN_NAMES = [
    "BMI", "ALT", "AST", "bilirubin", "chol", "GGT",
    "gluc_fast", "plt", "triglyc", "Aixplorer", "FibroTest", "FibroScan",
]
STAT_COLS = [
    "gender", "T2DM", "Hypertension", "Dyslipidaemia",
    "bariatric_surgery", "bariatric_surgery_age",
]
N_VISITS = 22
AGE_PREFIX = "Age_v"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _slope(values: np.ndarray) -> float:
    """Pente OLS sur les valeurs non-NaN."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = np.where(mask)[0].astype(float)
    y = values[mask]
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le CSV wide en un dataframe de features engineerées.
    Une ligne par patient.
    """
    rows = []
    for _, row in df.iterrows():
        feat = {}

        # Ages par visite
        ages = np.array([
            float(row[f"{AGE_PREFIX}{v}"]) if f"{AGE_PREFIX}{v}" in row.index
            and not pd.isna(row[f"{AGE_PREFIX}{v}"]) else np.nan
            for v in range(1, N_VISITS + 1)
        ])
        age_v1 = ages[0] if not np.isnan(ages[0]) else 0.0
        valid_mask = ~np.isnan(ages)
        n_visits = int(valid_mask.sum())
        last_age = float(ages[valid_mask][-1]) if n_visits > 0 else age_v1
        follow_up = last_age - age_v1

        feat["n_visits"]  = n_visits
        feat["age_v1"]    = age_v1
        feat["follow_up"] = follow_up

        # Features dynamiques
        for prefix, name in zip(DYN_PREFIXES, DYN_NAMES):
            vals = np.array([
                float(row[f"{prefix}{v}"]) if f"{prefix}{v}" in row.index
                and not pd.isna(row[f"{prefix}{v}"]) else np.nan
                for v in range(1, N_VISITS + 1)
            ])
            valid = vals[~np.isnan(vals)]

            feat[f"{name}_last"]  = float(valid[-1])  if len(valid) > 0 else np.nan
            feat[f"{name}_first"] = float(valid[0])   if len(valid) > 0 else np.nan
            feat[f"{name}_max"]   = float(valid.max()) if len(valid) > 0 else np.nan
            feat[f"{name}_min"]   = float(valid.min()) if len(valid) > 0 else np.nan
            feat[f"{name}_mean"]  = float(valid.mean()) if len(valid) > 0 else np.nan
            feat[f"{name}_std"]   = float(valid.std())  if len(valid) > 1 else 0.0
            feat[f"{name}_slope"] = _slope(vals)
            feat[f"{name}_n_obs"] = int(len(valid))
            # Δ last - first (progression absolue)
            if len(valid) >= 2:
                feat[f"{name}_delta"] = float(valid[-1] - valid[0])
            else:
                feat[f"{name}_delta"] = np.nan

        # Features statiques
        for col in STAT_COLS:
            feat[col] = float(row[col]) if col in row.index and not pd.isna(row[col]) else np.nan

        # Ratios cliniques utiles pour MASLD
        alt_last = feat.get("ALT_last", np.nan)
        ast_last = feat.get("AST_last", np.nan)
        if not (np.isnan(alt_last) or np.isnan(ast_last) or alt_last == 0):
            feat["AST_ALT_ratio"] = ast_last / alt_last
        else:
            feat["AST_ALT_ratio"] = np.nan

        fibro_last = feat.get("FibroScan_last", np.nan)
        feat["fibro_x_age"] = fibro_last * age_v1 if not np.isnan(fibro_last) else np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entraînement LightGBM survival
# ---------------------------------------------------------------------------

def train_lgbm(
    train_csv:  str,
    test_csv:   str,
    out_dir:    str = "data",
    n_folds:    int = 5,
    seed:       int = 42,
    verbose:    bool = True,
):
    """
    Entraîne deux modèles LightGBM (hépatique + décès) avec CV k-fold.
    Utilise l'objectif AFT (Accelerated Failure Time) pour la survie.
    Retourne les prédictions OOF (train) et test.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if verbose:
        print("Chargement CSV...")
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    trustii_ids = df_test["trustii_id"].values if "trustii_id" in df_test.columns else None

    if verbose:
        print("Feature engineering...")
    X_train = engineer_features(df_train)
    X_test  = engineer_features(df_test)

    # Targets
    event_hep  = df_train["evenements_hepatiques_majeurs"].fillna(0).astype(int).values
    time_hep   = df_train["evenements_hepatiques_age_occur"].where(
                     df_train["evenements_hepatiques_majeurs"] == 1,
                     df_train[[f"Age_v{v}" for v in range(1, N_VISITS+1)]].max(axis=1)
                 ).fillna(0).values
    event_dth  = df_train["death"].fillna(0).astype(int).values
    time_dth   = df_train["death_age_occur"].where(
                     df_train["death"] == 1,
                     df_train[[f"Age_v{v}" for v in range(1, N_VISITS+1)]].max(axis=1)
                 ).fillna(0).values

    # Paramètres LightGBM — binary + poids temporels (efficace pour survie)
    lgb_params = {
        "objective":         "binary",
        "metric":            "auc",
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        15,
        "min_child_samples": 10,
        "reg_alpha":         0.5,
        "reg_lambda":        1.0,
        "subsample":         0.7,
        "colsample_bytree":  0.7,
        "min_gain_to_split": 0.01,
        "random_state":      seed,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    results = {}
    from annitia.metrics import c_index as _ci

    for endpoint, events, times in [
        ("hepatic", event_hep, time_hep),
        ("death",   event_dth, time_dth),
    ]:
        if verbose:
            print(f"\n--- {endpoint} ({events.sum()} événements / {len(events)}) ---")

        oof_preds  = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, events)):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            t_tr, t_va = times[tr_idx], times[va_idx]
            e_tr, e_va = events[tr_idx], events[va_idx]

            # Poids : événements précoces = poids élevé ; censurés = poids 1
            max_t = t_tr.max() if t_tr.max() > 0 else 1.0
            w_tr  = np.ones(len(e_tr))
            event_mask = e_tr == 1
            w_tr[event_mask] = 1.0 + (max_t - t_tr[event_mask]) / max_t

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_tr, e_tr,
                sample_weight=w_tr,
                callbacks=[lgb.log_evaluation(period=-1)],
            )

            oof_preds[va_idx]  = model.predict_proba(X_va)[:, 1]
            test_preds        += model.predict_proba(X_test)[:, 1] / n_folds

            if verbose:
                ci = _ci(
                    oof_preds[va_idx].astype(np.float32),
                    t_va.astype(np.float32),
                    e_va.astype(np.uint8),
                )
                print(f"  fold {fold+1}/{n_folds} — C-index val: {ci:.4f}")

        ci_oof = _ci(
            oof_preds.astype(np.float32),
            times.astype(np.float32),
            events.astype(np.uint8),
        )
        if verbose:
            print(f"  OOF C-index: {ci_oof:.4f}")

        results[endpoint] = {
            "oof":    oof_preds,
            "test":   test_preds,
            "ci_oof": ci_oof,
        }

    sub = pd.DataFrame({
        "trustii_id":         trustii_ids if trustii_ids is not None else np.arange(len(X_test)),
        "risk_hepatic_event": results["hepatic"]["test"],
        "risk_death":         results["death"]["test"],
    })
    sub_path = out_dir / "submission_lgbm.csv"
    sub.to_csv(sub_path, index=False)
    if verbose:
        score = 0.7 * results["hepatic"]["ci_oof"] + 0.3 * results["death"]["ci_oof"]
        print(f"\nScore OOF LightGBM : {score:.4f}")
        print(f"Soumission : {sub_path}")

    return results
