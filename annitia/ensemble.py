"""
ensemble.py — Rank-average ensemble SSM K-Mamba + LightGBM pour ANNITIA.

Stratégie :
  - Normaliser les scores SSM et LightGBM en rangs [0,1]
  - Moyenne pondérée des rangs (alpha = poids SSM)
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


def ensemble_predictions(
    ssm_csv:   str,
    lgbm_csv:  str,
    out_csv:   str = "data/submission_ensemble.csv",
    alpha_ssm: float = 0.6,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Combine les prédictions SSM et LightGBM par rank-average pondéré.

    alpha_ssm = poids donné au modèle SSM (1-alpha_ssm = LightGBM).
    Retourne le DataFrame de soumission.
    """
    df_ssm  = pd.read_csv(ssm_csv)
    df_lgbm = pd.read_csv(lgbm_csv)

    # Aligner sur trustii_id
    df = df_ssm.merge(df_lgbm, on="trustii_id", suffixes=("_ssm", "_lgbm"))

    for endpoint in ("hepatic_event", "death"):
        col_ssm  = f"risk_{endpoint}_ssm"  if f"risk_{endpoint}_ssm"  in df else f"risk_{endpoint}"
        col_lgbm = f"risk_{endpoint}_lgbm" if f"risk_{endpoint}_lgbm" in df else f"risk_{endpoint}"

        # Fallback: chercher les colonnes exactes
        if col_ssm not in df.columns:
            candidates = [c for c in df.columns if endpoint in c and "ssm" in c]
            col_ssm = candidates[0] if candidates else None
        if col_lgbm not in df.columns:
            candidates = [c for c in df.columns if endpoint in c and "lgbm" in c]
            col_lgbm = candidates[0] if candidates else None

        if col_ssm is None or col_lgbm is None:
            raise ValueError(f"Colonnes introuvables pour endpoint '{endpoint}'. "
                             f"Colonnes : {df.columns.tolist()}")

        r_ssm  = _to_rank(df[col_ssm].values)
        r_lgbm = _to_rank(df[col_lgbm].values)
        df[f"risk_{endpoint}"] = alpha_ssm * r_ssm + (1 - alpha_ssm) * r_lgbm

    sub = df[["trustii_id", "risk_hepatic_event", "risk_death"]].copy()
    sub.to_csv(out_csv, index=False)

    if verbose:
        print(f"Ensemble ({alpha_ssm:.0%} SSM + {1-alpha_ssm:.0%} LightGBM)")
        print(f"  {len(sub)} patients → {out_csv}")

    return sub


def sweep_alpha(
    ssm_csv:       str,
    lgbm_csv:      str,
    oof_ssm_hep:   np.ndarray,
    oof_ssm_dth:   np.ndarray,
    oof_lgbm_hep:  np.ndarray,
    oof_lgbm_dth:  np.ndarray,
    times_hep:     np.ndarray,
    events_hep:    np.ndarray,
    times_dth:     np.ndarray,
    events_dth:    np.ndarray,
    alphas:        list = None,
) -> float:
    """
    Cherche le meilleur alpha_ssm sur les prédictions OOF.
    Retourne le meilleur alpha.
    """
    from annitia.metrics import c_index as _ci

    if alphas is None:
        alphas = [round(a * 0.1, 1) for a in range(0, 11)]

    best_score = -1.0
    best_alpha = 0.5

    print("\nSweep alpha_ssm (OOF) :")
    print("  alpha | C-idx Hep | C-idx Dth | Score")
    print("  ------|-----------|-----------|------")

    for alpha in alphas:
        r_ssm_h  = _to_rank(oof_ssm_hep)
        r_lgbm_h = _to_rank(oof_lgbm_hep)
        r_ssm_d  = _to_rank(oof_ssm_dth)
        r_lgbm_d = _to_rank(oof_lgbm_dth)

        ens_h = alpha * r_ssm_h + (1 - alpha) * r_lgbm_h
        ens_d = alpha * r_ssm_d + (1 - alpha) * r_lgbm_d

        ci_h = _ci(ens_h.astype(np.float32), times_hep.astype(np.float32), events_hep.astype(np.uint8))
        ci_d = _ci(ens_d.astype(np.float32), times_dth.astype(np.float32), events_dth.astype(np.uint8))
        score = 0.7 * ci_h + 0.3 * ci_d

        flag = " ←" if score > best_score else ""
        print(f"  {alpha:.1f}   | {ci_h:.4f}    | {ci_d:.4f}    | {score:.4f}{flag}")

        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"\n  Meilleur alpha_ssm = {best_alpha:.1f} → score = {best_score:.4f}")
    return best_alpha
