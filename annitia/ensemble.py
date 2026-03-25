"""
ensemble.py — Rank-average ensemble SSM K-Mamba + LightGBM + XGBoost pour ANNITIA.

Stratégie :
  - Normaliser les scores de chaque modèle en rangs [0,1]
  - Moyenne pondérée des rangs
  - Génère submission_ensemble.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _to_rank(scores: np.ndarray) -> np.ndarray:
    """Convertit des scores en rangs normalisés [0,1]."""
    n = len(scores)
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


def _get_col(df: pd.DataFrame, endpoint: str, suffix: str) -> str:
    """Trouve la colonne risk_{endpoint}_{suffix} ou risk_{endpoint}."""
    col = f"risk_{endpoint}_{suffix}"
    if col in df.columns:
        return col
    col = f"risk_{endpoint}"
    if col in df.columns:
        return col
    raise ValueError(f"Colonne introuvable pour endpoint='{endpoint}', suffix='{suffix}'. "
                     f"Colonnes disponibles: {df.columns.tolist()}")


def ensemble_predictions(
    ssm_csv:   str,
    lgbm_csv:  str,
    xgb_csv:   str  = None,
    out_csv:   str  = "data/submission_ensemble.csv",
    w_ssm:     float = 0.5,
    w_lgbm:    float = 0.3,
    w_xgb:     float = 0.2,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Combine SSM + LightGBM + XGBoost (optionnel) par rank-average pondéré.
    Les poids sont normalisés automatiquement (somme = 1).
    """
    frames   = [pd.read_csv(ssm_csv), pd.read_csv(lgbm_csv)]
    suffixes = ["ssm", "lgbm"]
    weights  = [w_ssm, w_lgbm]

    if xgb_csv is not None:
        frames.append(pd.read_csv(xgb_csv))
        suffixes.append("xgb")
        weights.append(w_xgb)

    # Fusion sur trustii_id
    df = frames[0]
    for i, (frame, suf) in enumerate(zip(frames[1:], suffixes[1:]), start=1):
        df = df.merge(frame, on="trustii_id",
                      suffixes=(f"_{suffixes[i-1]}", f"_{suf}"))

    # Renommer les colonnes du premier modèle si elles n'ont pas de suffixe
    for endpoint in ("hepatic_event", "death"):
        bare = f"risk_{endpoint}"
        if bare in df.columns and f"{bare}_{suffixes[0]}" not in df.columns:
            df = df.rename(columns={bare: f"{bare}_{suffixes[0]}"})

    # Normaliser les poids
    total = sum(weights)
    weights = [w / total for w in weights]

    for endpoint in ("hepatic_event", "death"):
        blended = np.zeros(len(df))
        for suf, w in zip(suffixes, weights):
            col = _get_col(df, endpoint, suf)
            blended += w * _to_rank(df[col].values)
        df[f"risk_{endpoint}"] = blended

    sub = df[["trustii_id", "risk_hepatic_event", "risk_death"]].copy()
    sub.to_csv(out_csv, index=False)

    if verbose:
        model_desc = " + ".join(
            f"{s.upper()} {w*100:.0f}%" for s, w in zip(suffixes, weights)
        )
        print(f"Ensemble : {model_desc}")
        print(f"  {len(sub)} patients → {out_csv}")

    return sub


def sweep_alpha(
    oof_preds:  dict,   # {"ssm_hep": arr, "ssm_dth": arr, "lgbm_hep": arr, ...}
    times_hep:  np.ndarray,
    events_hep: np.ndarray,
    times_dth:  np.ndarray,
    events_dth: np.ndarray,
    models:     list = None,  # ["ssm", "lgbm", "xgb"]
    steps:      int  = 5,
) -> dict:
    """
    Cherche les poids optimaux par grid search sur les prédictions OOF.
    Retourne le dict de poids optimal.
    """
    from annitia.metrics import c_index as _ci
    import itertools

    if models is None:
        models = [k.replace("_hep", "") for k in oof_preds if k.endswith("_hep")]

    best_score  = -1.0
    best_weights = {m: 1.0 / len(models) for m in models}

    # Grille de poids discrète
    grid = [round(i / steps, 2) for i in range(steps + 1)]
    combos = [c for c in itertools.product(grid, repeat=len(models)) if abs(sum(c) - 1.0) < 1e-6]

    print(f"\nSweep {len(combos)} combinaisons de poids ({models}) :")

    for combo in combos:
        w = dict(zip(models, combo))
        ens_h = sum(w[m] * _to_rank(oof_preds[f"{m}_hep"]) for m in models)
        ens_d = sum(w[m] * _to_rank(oof_preds[f"{m}_dth"]) for m in models)

        ci_h  = _ci(ens_h.astype(np.float32), times_hep.astype(np.float32), events_hep.astype(np.uint8))
        ci_d  = _ci(ens_d.astype(np.float32), times_dth.astype(np.float32), events_dth.astype(np.uint8))
        score = 0.7 * ci_h + 0.3 * ci_d

        if score > best_score:
            best_score   = score
            best_weights = w
            print(f"  {w} → hep={ci_h:.4f} dth={ci_d:.4f} score={score:.4f} ←")

    print(f"\nMeilleurs poids : {best_weights} → score OOF = {best_score:.4f}")
    return best_weights
