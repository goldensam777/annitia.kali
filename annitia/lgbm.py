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

        # --- Ratios cliniques MASLD ---
        alt_last  = feat.get("ALT_last",      np.nan)
        ast_last  = feat.get("AST_last",      np.nan)
        ggt_last  = feat.get("GGT_last",      np.nan)
        bili_last = feat.get("bilirubin_last", np.nan)
        plt_last  = feat.get("plt_last",      np.nan)
        fibro_last = feat.get("FibroScan_last", np.nan)
        ftest_last = feat.get("FibroTest_last", np.nan)
        bmi_last   = feat.get("BMI_last",      np.nan)

        # AST/ALT (De Ritis) > 1 → fibrose avancée
        feat["AST_ALT_ratio"] = (ast_last / alt_last
            if not any(np.isnan([ast_last, alt_last])) and alt_last != 0 else np.nan)

        # GGT/ALT → lésion hépatocellulaire vs cholestase
        feat["GGT_ALT_ratio"] = (ggt_last / alt_last
            if not any(np.isnan([ggt_last, alt_last])) and alt_last != 0 else np.nan)

        # FIB-4 index : (âge × AST) / (plt × √ALT)  — marqueur fibrose validé MASLD
        if not any(np.isnan([age_v1, ast_last, plt_last, alt_last])) and plt_last > 0 and alt_last > 0:
            feat["FIB4"] = (age_v1 * ast_last) / (plt_last * np.sqrt(alt_last))
        else:
            feat["FIB4"] = np.nan

        # FibroScan × FibroTest (deux mesures de fibrose — concordance)
        feat["FibroScan_x_FibroTest"] = (fibro_last * ftest_last
            if not any(np.isnan([fibro_last, ftest_last])) else np.nan)

        # FibroScan / FibroTest — ratio discordance
        feat["FibroScan_FibroTest_ratio"] = (fibro_last / ftest_last
            if not any(np.isnan([fibro_last, ftest_last])) and ftest_last != 0 else np.nan)

        # Bilirubin × AST — marqueur d'insuffisance hépatique
        feat["bili_x_AST"] = (bili_last * ast_last
            if not any(np.isnan([bili_last, ast_last])) else np.nan)

        # BMI × FibroScan — obésité + fibrose combinés
        feat["fibro_x_age"] = fibro_last * age_v1 if not np.isnan(fibro_last) else np.nan
        feat["BMI_x_fibro"]  = (bmi_last * fibro_last
            if not any(np.isnan([bmi_last, fibro_last])) else np.nan)

        # --- Patterns de données manquantes ---
        # Proportion de visites avec mesure (≠ proportion NaN)
        for prefix, name in zip(DYN_PREFIXES, DYN_NAMES):
            n_obs = feat.get(f"{name}_n_obs", 0)
            feat[f"{name}_obs_rate"] = n_obs / N_VISITS

        # Nombre de features mesurées à la dernière visite
        last_v = n_visits  # indice de la dernière visite
        n_measured_last = sum(
            1 for prefix, name in zip(DYN_PREFIXES, DYN_NAMES)
            if f"{prefix}{last_v}" in df.columns
            and not pd.isna(row.get(f"{prefix}{last_v}", np.nan))
        ) if n_visits > 0 else 0
        feat["n_features_last_visit"] = n_measured_last

        # --- Accélération temporelle (slope des slopes) ---
        # Indique si la dégradation s'accélère
        for prefix, name in zip(DYN_PREFIXES, DYN_NAMES):
            vals = np.array([
                float(row[f"{prefix}{v}"]) if f"{prefix}{v}" in row.index
                and not pd.isna(row[f"{prefix}{v}"]) else np.nan
                for v in range(1, N_VISITS + 1)
            ])
            # Slope sur la 2e moitié des visites vs 1ère moitié
            valid_idx = np.where(~np.isnan(vals))[0]
            if len(valid_idx) >= 4:
                mid = len(valid_idx) // 2
                first_half = vals[valid_idx[:mid]]
                second_half = vals[valid_idx[mid:]]
                feat[f"{name}_slope_recent"] = _slope(
                    np.concatenate([np.full(mid, np.nan), second_half
                                    if len(second_half) > 1 else np.array([np.nan])])
                )
                feat[f"{name}_accel"] = (
                    float(second_half.mean() - first_half.mean())
                    if len(first_half) > 0 and len(second_half) > 0 else np.nan
                )
            else:
                feat[f"{name}_slope_recent"] = np.nan
                feat[f"{name}_accel"]         = np.nan

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
