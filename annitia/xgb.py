"""
xgb.py — XGBoost Cox survival pour ANNITIA.

Utilise l'objectif `survival:cox` natif de XGBoost :
  y > 0  → événement au temps y
  y < 0  → censuré au temps |y|

Réutilise engineer_features() de lgbm.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .lgbm import engineer_features, N_VISITS


def train_xgb(
    train_csv: str,
    test_csv:  str,
    out_dir:   str = "data",
    n_folds:   int = 5,
    seed:      int = 42,
    verbose:   bool = True,
):
    """
    Entraîne deux modèles XGBoost Cox (hépatique + décès) avec CV k-fold.
    Retourne les prédictions OOF (train) et test.
    """
    import xgboost as xgb
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
    X_train = engineer_features(df_train).values.astype(np.float32)
    X_test  = engineer_features(df_test).values.astype(np.float32)
    feat_names = engineer_features(df_train).columns.tolist()

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

    # Paramètres XGBoost Cox
    xgb_params = {
        "objective":        "survival:cox",
        "eval_metric":      "cox-nloglik",
        "tree_method":      "hist",
        "n_estimators":     500,
        "learning_rate":    0.05,
        "max_depth":        4,
        "min_child_weight": 5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "seed":             seed,
        "nthread":          -1,
        "verbosity":        0,
    }

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
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            t_tr, t_va = times[tr_idx], times[va_idx]
            e_tr, e_va = events[tr_idx], events[va_idx]

            # Cox label : positif = event, négatif = censuré
            y_tr = np.where(e_tr == 1,  t_tr, -t_tr).astype(np.float32)
            y_va = np.where(e_va == 1,  t_va, -t_va).astype(np.float32)

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
            dval   = xgb.DMatrix(X_va, label=y_va, feature_names=feat_names)
            dtest  = xgb.DMatrix(X_test,            feature_names=feat_names)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=xgb_params["n_estimators"],
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            oof_preds[va_idx]  = model.predict(dval)
            test_preds        += model.predict(dtest) / n_folds

            if verbose:
                ci = _ci(
                    oof_preds[va_idx].astype(np.float32),
                    t_va.astype(np.float32),
                    e_va.astype(np.uint8),
                )
                print(f"  fold {fold+1}/{n_folds} — C-index val: {ci:.4f}  "
                      f"(best iter: {model.best_iteration})")

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
    sub_path = out_dir / "submission_xgb.csv"
    sub.to_csv(sub_path, index=False)

    if verbose:
        score = 0.7 * results["hepatic"]["ci_oof"] + 0.3 * results["death"]["ci_oof"]
        print(f"\nScore OOF XGBoost : {score:.4f}")
        print(f"Soumission : {sub_path}")

    return results
