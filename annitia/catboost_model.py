"""
catboost_model.py — CatBoost Cox survival pour ANNITIA.

Avantages vs XGBoost sur ces données :
  - Gestion native des features catégorielles (gender, T2DM, Hypertension, Dyslipidaemia, bariatric_surgery)
  - Moins sensible aux outliers (ordered boosting)
  - Pas besoin d'encodage one-hot

Objectif : Cox PH via `LossFunctionDescription='Cox'`
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .lgbm import engineer_features, N_VISITS, STAT_COLS


# Features catégorielles — CatBoost les gère nativement
CAT_FEATURES = ["gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]


def train_catboost(
    train_csv: str,
    test_csv:  str,
    out_dir:   str = "data",
    n_folds:   int = 5,
    seed:      int = 42,
    verbose:   bool = True,
):
    """
    Entraîne deux modèles CatBoost Cox (hépatique + décès) avec CV k-fold.
    Retourne les prédictions OOF (train) et test.
    """
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import StratifiedKFold
    from annitia.metrics import c_index as _ci

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

    # Indices des colonnes catégorielles dans le DataFrame
    all_cols  = X_train.columns.tolist()
    cat_idxs  = [all_cols.index(c) for c in CAT_FEATURES if c in all_cols]

    # Remplir NaN dans les catégorielles par -1 (CatBoost les accepte)
    for c in CAT_FEATURES:
        if c in X_train.columns:
            X_train[c] = X_train[c].fillna(-1).astype(int).astype(str)
            X_test[c]  = X_test[c].fillna(-1).astype(int).astype(str)

    # Targets
    event_hep = df_train["evenements_hepatiques_majeurs"].fillna(0).astype(int).values
    time_hep  = df_train["evenements_hepatiques_age_occur"].where(
                    df_train["evenements_hepatiques_majeurs"] == 1,
                    df_train[[f"Age_v{v}" for v in range(1, N_VISITS+1)]].max(axis=1)
                ).fillna(1e-3).values
    event_dth = df_train["death"].fillna(0).astype(int).values
    time_dth  = df_train["death_age_occur"].where(
                    df_train["death"] == 1,
                    df_train[[f"Age_v{v}" for v in range(1, N_VISITS+1)]].max(axis=1)
                ).fillna(1e-3).values

    # Paramètres CatBoost Cox
    cb_params = dict(
        loss_function    = "Cox",
        eval_metric      = "Cox",
        iterations       = 1000,
        learning_rate    = 0.05,
        depth            = 4,
        l2_leaf_reg      = 3.0,
        subsample        = 0.8,
        colsample_bylevel= 0.8,
        min_data_in_leaf = 5,
        random_seed      = seed,
        thread_count     = -1,
        verbose          = False,
        early_stopping_rounds = 50,
    )

    results = {}

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
            X_tr = X_train.iloc[tr_idx].values
            X_va = X_train.iloc[va_idx].values
            t_tr, t_va = times[tr_idx], times[va_idx]
            e_tr, e_va = events[tr_idx], events[va_idx]

            # CatBoost Cox : label = +time pour events, -time pour censurés
            y_tr = np.where(e_tr == 1,  t_tr, -t_tr).astype(np.float32)
            y_va = np.where(e_va == 1,  t_va, -t_va).astype(np.float32)

            pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_idxs,
                           feature_names=all_cols)
            pool_va = Pool(X_va, label=y_va, cat_features=cat_idxs,
                           feature_names=all_cols)
            pool_te = Pool(X_test.values,    cat_features=cat_idxs,
                           feature_names=all_cols)

            model = CatBoostRegressor(**cb_params)
            model.fit(pool_tr, eval_set=pool_va)

            oof_preds[va_idx]  = model.predict(pool_va)
            test_preds        += model.predict(pool_te) / n_folds

            if verbose:
                ci = _ci(
                    oof_preds[va_idx].astype(np.float32),
                    t_va.astype(np.float32),
                    e_va.astype(np.uint8),
                )
                print(f"  fold {fold+1}/{n_folds} — C-index val: {ci:.4f}  "
                      f"(best iter: {model.best_iteration_})")

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
    sub_path = out_dir / "submission_catboost.csv"
    sub.to_csv(sub_path, index=False)

    if verbose:
        score = 0.7 * results["hepatic"]["ci_oof"] + 0.3 * results["death"]["ci_oof"]
        print(f"\nScore OOF CatBoost : {score:.4f}")
        print(f"Soumission : {sub_path}")

    return results
