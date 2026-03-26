#ifndef ANNITIA_H
#define ANNITIA_H

#include <stddef.h>
#include <stdint.h>
#include "kmamba.h"   /* MambaBlock, MBConfig, MBOptimConfig */

/* ------------------------------------------------------------------ */
/* Dimensions MASLD                                                     */
/* ------------------------------------------------------------------ */
#define ANNITIA_MAX_TIMESTEPS  22
/* 12 features dynamiques par visite :
 * BMI, ALT, AST, bilirubin, chol, GGT, gluc_fast, plt, triglyc,
 * Aixplorer, FibroTest, FibroScan
 * + 6 features statiques répétées :
 * gender, T2DM, Hypertension, Dyslipidaemia, bariatric_surgery, bariatric_surgery_age */
#define ANNITIA_DYN_FEATURES   12
#define ANNITIA_STAT_FEATURES   6
#define ANNITIA_N_FEATURES     (ANNITIA_DYN_FEATURES + ANNITIA_STAT_FEATURES)  /* 18 */

/* Magic number for MASL binary format ('MASL') */
#define MASLD_MAGIC  0x4D41534Cu

/* ------------------------------------------------------------------ */
/* Configuration du modèle                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    size_t n_features;   /* nombre de features continues par timestep  */
    size_t dim;          /* dimension interne SSM                       */
    size_t state_size;   /* taille de l'état SSM (N)                    */
    size_t seq_len;      /* nombre max de timesteps (T ≤ 22)            */
    size_t n_layers;     /* nombre de MambaBlocks empilés               */
    size_t mimo_rank;    /* MIMO rank R (0/1 = SISO)                    */
    float  dt_scale;
    float  dt_min;
    float  dt_max;
    /* Conv2D preprocessing on raw [T, F] feature matrix */
    int    use_conv2d;   /* 1 = enable Conv2D preprocessing layer       */
    size_t conv2d_K;     /* kernel size along each axis (default 3)     */
} AnnitiaConfig;

/* ------------------------------------------------------------------ */
/* Modèle principal                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    AnnitiaConfig cfg;

    /* Couche d'entrée : projection features → dim */
    float *W_feat;       /* [n_features, dim] — remplace embedding      */
    float *b_feat;       /* [dim]             — biais optionnel          */

    /* Encodage temporel : time_gap → dim (projection scalaire) */
    float *W_time;       /* [1, dim]                                    */

    /* Cœur SSM : MambaBlocks de k-mamba (inchangés) */
    MambaBlock **layers; /* [n_layers]                                  */

    /* Têtes de survie (dual output) */
    float *W_hepatic;    /* [dim, 1] — risque événement hépatique       */
    float *W_death;      /* [dim, 1] — risque décès toutes causes       */

    /* Conv2D preprocessing layer (optional, cfg.use_conv2d=1)
     * Operates on raw [T, F] feature matrix before W_feat projection.
     * convnd: ndims=2, dims=[T,F], D=1, K=conv2d_K
     * kernel: [2 * conv2d_K] (one K-vector per axis, D=1 so scalar) */
    float *conv2d_kernel;   /* [2 * conv2d_K * 1] */
    float *conv2d_bias;     /* [1] */
    float *m_conv2d_kernel, *v_conv2d_kernel;
    float *m_conv2d_bias,   *v_conv2d_bias;

    /* État d'entraînement */
    int for_training;
    MBOptimConfig opt_cfg;
    float lr;
    float weight_decay;

    /* Moments Adam pour W_feat, b_feat, W_time, W_hepatic, W_death */
    float *m_W_feat,    *v_W_feat;
    float *m_b_feat,    *v_b_feat;
    float *m_W_time,    *v_W_time;
    float *m_W_hepatic, *v_W_hepatic;
    float *m_W_death,   *v_W_death;
    size_t step;

    /* Métriques du dernier step */
    float last_loss;
    float last_grad_norm;
} AnnitiaModel;

/* ------------------------------------------------------------------ */
/* Batch de survie                                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    /* Données d'entrée */
    float   *features;      /* [B, T, F] row-major — valeurs normalisées */
    float   *mask;          /* [B, T, F] — 1=valide, 0=manquant          */
    float   *time_gaps;     /* [B, T]   — jours depuis visite précédente */
    int     *n_visits;      /* [B]      — nombre réel de visites par patient */

    /* Targets de survie */
    float   *time_hepatic;  /* [B] — temps jusqu'à événement hépatique   */
    uint8_t *event_hepatic; /* [B] — 1=événement observé, 0=censuré      */
    float   *time_death;    /* [B] — temps jusqu'au décès                */
    uint8_t *event_death;   /* [B] — 1=décès observé, 0=censuré          */

    size_t   batch_size;    /* B                                          */
    size_t   seq_len;       /* T (padded)                                 */
    size_t   n_features;    /* F                                          */
} SurvivalBatch;

/* ------------------------------------------------------------------ */
/* API du modèle                                                        */
/* ------------------------------------------------------------------ */

/* Création / destruction */
AnnitiaModel *annitia_create(const AnnitiaConfig *cfg);
void          annitia_free(AnnitiaModel *m);

/* Initialisation des poids (Xavier uniform, seed reproductible) */
void annitia_init(AnnitiaModel *m, unsigned long seed);

/* Activation du mode entraînement */
void annitia_enable_training(AnnitiaModel *m,
                              const MBOptimConfig *opt_cfg,
                              float lr,
                              float weight_decay);

/* Forward : calcule les risk scores pour un batch
 * risk_hepatic[B], risk_death[B] : sorties allouées par l'appelant */
void annitia_forward(AnnitiaModel *m,
                     const SurvivalBatch *batch,
                     float *risk_hepatic,
                     float *risk_death);

/* Train step : forward + loss + backward + update
 * Retourne la loss scalaire combinée */
float annitia_train_step(AnnitiaModel *m, const SurvivalBatch *batch);

/* Sauvegarde / chargement checkpoint binaire */
int annitia_save(const AnnitiaModel *m, const char *path);
AnnitiaModel *annitia_load(const char *path, int for_training,
                            const MBOptimConfig *opt_cfg,
                            float lr, float weight_decay);

/* ------------------------------------------------------------------ */
/* API des données                                                      */
/* ------------------------------------------------------------------ */

/* Chargement depuis fichier binaire produit par preprocess.py */
typedef struct MasldDataset MasldDataset;

MasldDataset *masld_load(const char *path);
void          masld_free(MasldDataset *ds);
size_t        masld_n_patients(const MasldDataset *ds);

/* Construit un batch (alloué par l'appelant via masld_batch_alloc) */
SurvivalBatch *masld_batch_alloc(size_t batch_size, size_t seq_len, size_t n_features);
void           masld_batch_free(SurvivalBatch *b);

/* Remplit le batch avec les patients [start, start+B) */
void masld_get_batch(const MasldDataset *ds,
                     SurvivalBatch *batch,
                     size_t start,
                     size_t batch_size);

/* Remplit le batch à partir d'une liste d'indices (pour shuffled batching) */
void masld_get_batch_idx(const MasldDataset *ds,
                          SurvivalBatch *batch,
                          const size_t *idx,
                          size_t batch_size);

/* ------------------------------------------------------------------ */
/* API de la loss de survie                                             */
/* ------------------------------------------------------------------ */

/* Cox partial likelihood (pour un endpoint)
 * risks[B], times[B], events[B] → scalar loss + gradients grads[B] */
float cox_loss(const float *risks, const float *times,
               const uint8_t *events, float *grads, size_t n);

/* Pairwise ranking loss (proxy C-index différentiable)
 * risks[B], times[B], events[B] → scalar loss + gradients grads[B] */
float ranking_loss(const float *risks, const float *times,
                   const uint8_t *events, float *grads, size_t n);

/* Loss combinée : alpha * cox + (1-alpha) * ranking */
float survival_loss_combined(const float *risks, const float *times,
                              const uint8_t *events, float *grads,
                              size_t n, float alpha);

/* C-index (évaluation, non différentiable) */
float c_index(const float *risks, const float *times,
              const uint8_t *events, size_t n);

/* ------------------------------------------------------------------ */
/* Utilitaires masque (mask_utils.c)                                   */
/* ------------------------------------------------------------------ */

void   mask_apply(float *features, const float *mask, size_t T, size_t F);
size_t mask_last_valid(const float *mask, size_t T, size_t F);
void   mask_timestep(const float *mask, float *out, size_t T, size_t F);
void   mask_mean_pool(const float *hidden, const float *tmask,
                      float *out, size_t T, size_t D);
void   mask_last_pool(const float *hidden, const float *tmask,
                      float *out, size_t T, size_t D);

#endif /* ANNITIA_H */
