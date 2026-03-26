"""
ssm_kfold.py — SSM K-Mamba k-fold OOF pour ANNITIA.

Stratégie :
  1. Charge le dataset binaire complet
  2. Split stratifié en k folds (par event_hepatic + event_death)
  3. Pour chaque fold : écrit train.bin / val.bin temporaires → entraîne SSM → prédit OOF
  4. Retourne prédictions OOF (1253 patients) + prédictions test (423 patients)

Format binaire MASL (reproduit depuis masld_data.c) :
  Header : magic(u32) n_patients(u32) seq_len(u32) n_features(u32) version(u32) pad(u32)
  Per patient : features[T*F](f32) mask[T*F](f32) time_gaps[T](f32)
                n_visits(i32) time_hepatic(f32) event_hepatic(u8) pad(u8)
                time_death(f32) event_death(u8) pad[3](u8)
"""

import struct
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

MASLD_MAGIC = 0x4D41534C


# ---------------------------------------------------------------------------
# Lecture / écriture binaire MASL
# ---------------------------------------------------------------------------

def _read_bin(path: str) -> dict:
    """Charge un fichier .bin MASL en numpy arrays.
    Format entrelacé par patient : features/mask/time_gaps/labels pour chaque patient.
    """
    with open(path, "rb") as f:
        magic, n, T, F, version, _pad = struct.unpack("<6I", f.read(24))
        assert magic == MASLD_MAGIC, f"Magic invalide dans {path}"
        TF = T * F

        features      = np.empty((n, T, F), dtype=np.float32)
        mask          = np.empty((n, T, F), dtype=np.float32)
        time_gaps     = np.empty((n, T),    dtype=np.float32)
        n_visits      = np.empty(n, dtype=np.int32)
        time_hepatic  = np.empty(n, dtype=np.float32)
        event_hepatic = np.empty(n, dtype=np.uint8)
        time_death    = np.empty(n, dtype=np.float32)
        event_death   = np.empty(n, dtype=np.uint8)

        for i in range(n):
            features[i]  = np.frombuffer(f.read(TF * 4),  dtype=np.float32).reshape(T, F)
            mask[i]      = np.frombuffer(f.read(TF * 4),  dtype=np.float32).reshape(T, F)
            time_gaps[i] = np.frombuffer(f.read(T  * 4),  dtype=np.float32)
            nv, th       = struct.unpack("<if", f.read(8))
            eh, _p1      = struct.unpack("<BB", f.read(2))
            td,          = struct.unpack("<f",  f.read(4))
            ed, _p2, _p3, _p4 = struct.unpack("<BBBB", f.read(4))
            n_visits[i]      = nv
            time_hepatic[i]  = th
            event_hepatic[i] = eh
            time_death[i]    = td
            event_death[i]   = ed

    return dict(
        n=n, T=T, F=F,
        features=features, mask=mask, time_gaps=time_gaps,
        n_visits=n_visits, time_hepatic=time_hepatic,
        event_hepatic=event_hepatic, time_death=time_death, event_death=event_death,
    )


def _write_bin(path: str, data: dict, indices: np.ndarray):
    """Écrit un sous-ensemble de patients dans un fichier .bin MASL."""
    n   = len(indices)
    T   = data["T"]
    F   = data["F"]
    TF  = T * F

    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MASLD_MAGIC, n, T, F, 1, 0))
        for i in indices:
            f.write(data["features"][i].astype(np.float32).tobytes())
            f.write(data["mask"][i].astype(np.float32).tobytes())
            f.write(data["time_gaps"][i].astype(np.float32).tobytes())
            f.write(struct.pack("<i", int(data["n_visits"][i])))
            f.write(struct.pack("<f", float(data["time_hepatic"][i])))
            f.write(struct.pack("<BB", int(data["event_hepatic"][i]), 0))
            f.write(struct.pack("<f", float(data["time_death"][i])))
            f.write(struct.pack("<BBBB", int(data["event_death"][i]), 0, 0, 0))


# ---------------------------------------------------------------------------
# K-fold SSM OOF
# ---------------------------------------------------------------------------

def train_ssm_kfold(
    full_bin:   str,
    test_bin:   str,
    out_dir:    str  = "data",
    n_folds:    int  = 5,
    epochs:     int  = 30,
    seed:       int  = 42,
    verbose:    bool = True,
    use_conv2d: int  = 0,
    conv2d_K:   int  = 3,
    mimo_rank:  int  = 1,
) -> dict:
    """
    Entraîne le SSM K-Mamba en k-fold et retourne les prédictions OOF + test.
    """
    from annitia.model    import AnnitiaModel
    from annitia.dataset  import MasldDataset
    from annitia.train    import train as ssm_train
    from annitia.metrics  import c_index as _ci

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if verbose:
        print("Chargement données...")

    data_full = _read_bin(full_bin)
    n_total   = data_full["n"]
    T         = data_full["T"]
    F         = data_full["F"]

    # Stratification combinée hépatique + décès
    strat = (data_full["event_hepatic"].astype(int) * 2
             + data_full["event_death"].astype(int))

    oof_hep = np.zeros(n_total, dtype=np.float32)
    oof_dth = np.zeros(n_total, dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    with tempfile.TemporaryDirectory() as tmpdir:

        # Prédictions test : moyennées sur les k modèles
        test_hep_acc = np.zeros(0, dtype=np.float32)
        test_dth_acc = np.zeros(0, dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_total), strat)):
            if verbose:
                print(f"\n--- Fold {fold+1}/{n_folds}  "
                      f"({len(tr_idx)} train / {len(va_idx)} val) ---")

            # Écrire fichiers binaires temporaires
            tr_bin = str(Path(tmpdir) / f"fold{fold}_train.bin")
            va_bin = str(Path(tmpdir) / f"fold{fold}_val.bin")
            _write_bin(tr_bin, data_full, tr_idx)
            _write_bin(va_bin, data_full, va_idx)

            ds_train = MasldDataset(tr_bin)
            ds_val   = MasldDataset(va_bin)

            # Config SSM (mêmes hyperparamètres que le modèle principal)
            model = AnnitiaModel(
                dim        = 64,
                state      = 16,
                layers     = 2,
                n_features = F,
                seq_len    = T,
                mimo_rank  = mimo_rank,
                use_conv2d = use_conv2d,
                conv2d_K   = conv2d_K,
            )
            model.init(seed=seed + fold)
            model.enable_training(lr=3e-4, wd=1e-4, mu=0.9,
                                  beta2=0.999, eps=1e-8, clip_norm=1.0)

            history = ssm_train(
                model, ds_train, ds_val,
                epochs     = epochs,
                batch_size = 32,
                seed       = seed + fold,
                checkpoint = str(out_dir / f"model_fold{fold}.bin"),
                verbose    = verbose,
            )

            # Charger le meilleur checkpoint du fold
            best = AnnitiaModel.load(str(out_dir / f"model_fold{fold}.bin"))

            # Prédictions OOF (val fold)
            batch_va = ds_val.get_all()
            rh, rd   = best.forward(batch_va)
            oof_hep[va_idx] = rh.astype(np.float32)
            oof_dth[va_idx] = rd.astype(np.float32)

            ci_h = _ci(rh.astype(np.float32),
                       data_full["time_hepatic"][va_idx],
                       data_full["event_hepatic"][va_idx])
            ci_d = _ci(rd.astype(np.float32),
                       data_full["time_death"][va_idx],
                       data_full["event_death"][va_idx])
            score = 0.7 * ci_h + 0.3 * ci_d
            if verbose:
                print(f"  OOF fold {fold+1}: hep={ci_h:.4f} dth={ci_d:.4f} score={score:.4f}")

            # Prédictions test
            ds_test   = MasldDataset(test_bin)
            batch_te  = ds_test.get_all()
            rh_te, rd_te = best.forward(batch_te)
            if len(test_hep_acc) == 0:
                test_hep_acc = rh_te.astype(np.float32) / n_folds
                test_dth_acc = rd_te.astype(np.float32) / n_folds
            else:
                test_hep_acc += rh_te.astype(np.float32) / n_folds
                test_dth_acc += rd_te.astype(np.float32) / n_folds

    # C-index OOF global
    ci_h_oof = _ci(oof_hep, data_full["time_hepatic"], data_full["event_hepatic"])
    ci_d_oof = _ci(oof_dth, data_full["time_death"],   data_full["event_death"])
    score_oof = 0.7 * ci_h_oof + 0.3 * ci_d_oof

    if verbose:
        print(f"\n=== SSM OOF global ===")
        print(f"  hépatique : {ci_h_oof:.4f}")
        print(f"  décès     : {ci_d_oof:.4f}")
        print(f"  score     : {score_oof:.4f}")

    # Sauvegarder OOF
    np.save(str(out_dir / "oof_ssm_hep.npy"), oof_hep)
    np.save(str(out_dir / "oof_ssm_dth.npy"), oof_dth)
    np.save(str(out_dir / "test_ssm_hep.npy"), test_hep_acc)
    np.save(str(out_dir / "test_ssm_dth.npy"), test_dth_acc)

    return dict(
        oof_hep=oof_hep, oof_dth=oof_dth,
        test_hep=test_hep_acc, test_dth=test_dth_acc,
        ci_hep=ci_h_oof, ci_dth=ci_d_oof, score=score_oof,
    )
