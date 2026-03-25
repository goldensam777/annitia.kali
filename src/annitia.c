#include "annitia.h"
#include "openblas_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Utilitaires internes                                                 */
/* ------------------------------------------------------------------ */

static float randf(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* Xavier uniform : U[-limit, limit] avec limit = sqrt(6/(fan_in+fan_out)) */
static void xavier_uniform(float *w, size_t n, size_t fan_in, size_t fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < n; i++)
        w[i] = (2.0f * randf() - 1.0f) * limit;
}

/* Adam update in-place */
static void adam_update(float *param, float *grad, float *m, float *v,
                         size_t n, float lr, float wd,
                         float beta1, float beta2, float eps, size_t step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (size_t i = 0; i < n; i++) {
        float g = grad[i] + wd * param[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ------------------------------------------------------------------ */
/* Création / destruction                                               */
/* ------------------------------------------------------------------ */

AnnitiaModel *annitia_create(const AnnitiaConfig *cfg) {
    AnnitiaModel *m = calloc(1, sizeof(AnnitiaModel));
    m->cfg = *cfg;

    size_t F   = cfg->n_features;
    size_t D   = cfg->dim;
    size_t L   = cfg->n_layers;

    /* Feature projection */
    m->W_feat    = malloc(F * D * sizeof(float));
    m->b_feat    = calloc(D, sizeof(float));
    m->W_time    = malloc(1 * D * sizeof(float));

    /* Survival heads */
    m->W_hepatic = malloc(D * sizeof(float));
    m->W_death   = malloc(D * sizeof(float));

    /* MambaBlocks (réutilisés de k-mamba) */
    MBConfig block_cfg = {
        .dim        = D,
        .state_size = cfg->state_size,
        .seq_len    = cfg->seq_len,
        .mimo_rank  = cfg->mimo_rank,
        .dt_scale   = cfg->dt_scale  > 0 ? cfg->dt_scale  : 1.0f,
        .dt_min     = cfg->dt_min    > 0 ? cfg->dt_min    : 0.001f,
        .dt_max     = cfg->dt_max    > 0 ? cfg->dt_max    : 0.1f,
        .use_convnd = 0,
    };
    m->layers = malloc(L * sizeof(MambaBlock *));
    for (size_t i = 0; i < L; i++)
        m->layers[i] = mamba_block_create(&block_cfg);

    /* Moments Adam */
    m->m_W_feat    = calloc(F * D, sizeof(float));
    m->v_W_feat    = calloc(F * D, sizeof(float));
    m->m_b_feat    = calloc(D, sizeof(float));
    m->v_b_feat    = calloc(D, sizeof(float));
    m->m_W_time    = calloc(D, sizeof(float));
    m->v_W_time    = calloc(D, sizeof(float));
    m->m_W_hepatic = calloc(D, sizeof(float));
    m->v_W_hepatic = calloc(D, sizeof(float));
    m->m_W_death   = calloc(D, sizeof(float));
    m->v_W_death   = calloc(D, sizeof(float));

    return m;
}

void annitia_free(AnnitiaModel *m) {
    if (!m) return;
    free(m->W_feat);    free(m->b_feat);  free(m->W_time);
    free(m->W_hepatic); free(m->W_death);
    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_block_free(m->layers[i]);
    free(m->layers);
    free(m->m_W_feat);    free(m->v_W_feat);
    free(m->m_b_feat);    free(m->v_b_feat);
    free(m->m_W_time);    free(m->v_W_time);
    free(m->m_W_hepatic); free(m->v_W_hepatic);
    free(m->m_W_death);   free(m->v_W_death);
    free(m);
}

/* ------------------------------------------------------------------ */
/* Initialisation des poids                                             */
/* ------------------------------------------------------------------ */

void annitia_init(AnnitiaModel *m, unsigned long seed) {
    srand((unsigned int)seed);
    size_t F = m->cfg.n_features;
    size_t D = m->cfg.dim;

    xavier_uniform(m->W_feat,    F * D, F, D);
    xavier_uniform(m->W_time,    D,     1, D);
    xavier_uniform(m->W_hepatic, D,     D, 1);
    xavier_uniform(m->W_death,   D,     D, 1);
    /* b_feat déjà à zéro (calloc) */

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        srand((unsigned int)(seed + i + 1));
        mamba_block_init(m->layers[i]);
    }
}

/* ------------------------------------------------------------------ */
/* Activation mode entraînement                                        */
/* ------------------------------------------------------------------ */

void annitia_enable_training(AnnitiaModel *m,
                              const MBOptimConfig *opt_cfg,
                              float lr, float weight_decay) {
    m->for_training  = 1;
    m->opt_cfg       = *opt_cfg;
    m->lr            = lr;
    m->weight_decay  = weight_decay;
    m->step          = 0;

    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_attach_optimizer(m->layers[i], OPTIMIZER_ADAM_CLIP, opt_cfg);
}

/* ------------------------------------------------------------------ */
/* Forward pass pour un patient unique                                  */
/*                                                                      */
/* features_p[T, F], mask_p[T, F], time_gaps_p[T]                     */
/* → hidden_out[T, D] (buffer alloué par l'appelant)                   */
/* → patient_vec[D]   (pooling final)                                  */
/* ------------------------------------------------------------------ */
static void forward_patient(AnnitiaModel *m,
                             const float *features_p,
                             const float *mask_p,
                             const float *time_gaps_p,
                             float *hidden_out,
                             float *patient_vec,
                             float *buf_tmp) {
    size_t T = m->cfg.seq_len;
    size_t F = m->cfg.n_features;
    size_t D = m->cfg.dim;

    /* Étape 1 : Feature projection → hidden_in[T, D]
     * hidden_in[t] = W_feat^T @ features_p[t] + b_feat
     * + W_time * time_gaps_p[t]                        */
    float *hidden_in = buf_tmp;  /* [T, D] */

    for (size_t t = 0; t < T; t++) {
        const float *x = features_p + t * F;
        float       *h = hidden_in  + t * D;

        /* h = W_feat^T @ x  (gemv : M=D, K=F) */
        gemv_rowmajor(m->W_feat, x, h, (int)D, (int)F);

        /* h += b_feat + W_time * time_gap */
        float tg = time_gaps_p[t];
        for (size_t d = 0; d < D; d++)
            h[d] += m->b_feat[d] + m->W_time[d] * tg;

        /* Appliquer le masque temporel : si timestep invalide, zéro */
        float any = 0.0f;
        for (size_t f = 0; f < F; f++) any += mask_p[t * F + f];
        if (any < 0.5f)
            memset(h, 0, D * sizeof(float));
    }

    /* Étape 2 : MambaBlocks (cœur SSM, inchangé de k-mamba)
     * batch_size=1, input/output alternés entre hidden_in et hidden_out */
    float *cur = hidden_in;
    float *nxt = hidden_out;

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_forward(m->layers[i], nxt, cur, 1);
        /* Swap */
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    /* Après n_layers swaps, le résultat final est dans cur.
     * Si n_layers est pair, cur = hidden_in (buf_tmp) ; sinon hidden_out.
     * On copie dans hidden_out pour cohérence. */
    if (cur != hidden_out)
        memcpy(hidden_out, cur, T * D * sizeof(float));

    /* Étape 3 : Pooling — last valid timestep */
    float tmask[ANNITIA_MAX_TIMESTEPS];
    mask_timestep(mask_p, tmask, T, F);
    mask_last_pool(hidden_out, tmask, patient_vec, T, D);
}

/* ------------------------------------------------------------------ */
/* Forward batch                                                        */
/* ------------------------------------------------------------------ */

void annitia_forward(AnnitiaModel *m,
                     const SurvivalBatch *batch,
                     float *risk_hepatic,
                     float *risk_death) {
    size_t B  = batch->batch_size;
    size_t T  = batch->seq_len;
    size_t F  = batch->n_features;
    size_t D  = m->cfg.dim;

    float *hidden_out  = malloc(T * D * sizeof(float));
    float *buf_tmp     = malloc(T * D * sizeof(float));
    float *patient_vec = malloc(D * sizeof(float));

    for (size_t b = 0; b < B; b++) {
        const float *feat  = batch->features  + b * T * F;
        const float *mask  = batch->mask      + b * T * F;
        const float *tgaps = batch->time_gaps + b * T;

        forward_patient(m, feat, mask, tgaps,
                         hidden_out, patient_vec, buf_tmp);

        /* Dot product avec les têtes de survie */
        float rh = 0.0f, rd = 0.0f;
        for (size_t d = 0; d < D; d++) {
            rh += patient_vec[d] * m->W_hepatic[d];
            rd += patient_vec[d] * m->W_death[d];
        }
        risk_hepatic[b] = rh;
        risk_death[b]   = rd;
    }

    free(hidden_out);
    free(buf_tmp);
    free(patient_vec);
}

/* ------------------------------------------------------------------ */
/* Train step                                                           */
/* ------------------------------------------------------------------ */

float annitia_train_step(AnnitiaModel *m, const SurvivalBatch *batch) {
    size_t B  = batch->batch_size;
    size_t T  = batch->seq_len;
    size_t F  = batch->n_features;
    size_t D  = m->cfg.dim;
    size_t NL = m->cfg.n_layers;

    /* ---- Phase 1 : Forward avec stockage des activations ---- */

    /* acts[b][(NL+1)*T*D] : acts[b][i*T*D] = input to layer i (i=0..NL)
     * acts[b][NL*T*D]     = final SSM output                          */
    float **acts = malloc(B * sizeof(float *));
    for (size_t b = 0; b < B; b++)
        acts[b] = malloc((NL + 1) * T * D * sizeof(float));

    float *patient_vec = malloc(D * sizeof(float));
    float *risk_hepatic = malloc(B * sizeof(float));
    float *risk_death   = malloc(B * sizeof(float));
    float *tmask_buf    = malloc(T * sizeof(float));

    for (size_t b = 0; b < B; b++) {
        const float *feat  = batch->features  + b * T * F;
        const float *mask  = batch->mask      + b * T * F;
        const float *tgaps = batch->time_gaps + b * T;
        float *a0 = acts[b];  /* acts[0] = feature proj output */

        /* Feature projection */
        for (size_t t = 0; t < T; t++) {
            const float *x = feat  + t * F;
            float       *h = a0   + t * D;
            gemv_rowmajor(m->W_feat, x, h, (int)D, (int)F);
            float tg  = tgaps[t];
            float any = 0.0f;
            for (size_t f = 0; f < F; f++) any += mask[t * F + f];
            if (any < 0.5f) { memset(h, 0, D * sizeof(float)); continue; }
            for (size_t d = 0; d < D; d++)
                h[d] += m->b_feat[d] + m->W_time[d] * tg;
        }

        /* SSM forward through NL layers */
        for (size_t i = 0; i < NL; i++)
            mamba_block_forward(m->layers[i],
                                acts[b] + (i + 1) * T * D,
                                acts[b] + i       * T * D, 1);

        /* Pool and compute risks */
        mask_timestep(mask, tmask_buf, T, F);
        mask_last_pool(acts[b] + NL * T * D, tmask_buf, patient_vec, T, D);

        float rh = 0.0f, rd = 0.0f;
        for (size_t d = 0; d < D; d++) {
            rh += patient_vec[d] * m->W_hepatic[d];
            rd += patient_vec[d] * m->W_death[d];
        }
        risk_hepatic[b] = rh;
        risk_death[b]   = rd;
    }

    /* ---- Phase 2 : Loss ---- */
    float *g_hepatic = calloc(B, sizeof(float));
    float *g_death   = calloc(B, sizeof(float));

    /* alpha_hep=0.2 : 80% ranking + 20% Cox — adapté aux ~37 événements en train
     * alpha_dth=0.5 : équilibré — 59 événements, moins rare                    */
    float l_hep = survival_loss_combined(risk_hepatic,
                                          batch->time_hepatic,
                                          batch->event_hepatic,
                                          g_hepatic, B, 0.2f);
    float l_dth = survival_loss_combined(risk_death,
                                          batch->time_death,
                                          batch->event_death,
                                          g_death, B, 0.5f);

    float loss = 0.7f * l_hep + 0.3f * l_dth;
    m->last_loss = loss;

    /* ---- Phase 3 : Backward ---- */
    float *gW_hepatic = calloc(D, sizeof(float));
    float *gW_death   = calloc(D, sizeof(float));
    float *gW_feat    = calloc(F * D, sizeof(float));
    float *gb_feat    = calloc(D, sizeof(float));
    float *gW_time    = calloc(D, sizeof(float));
    float *d_pool     = calloc(T * D, sizeof(float));  /* grad through pooling */
    float *dcur       = malloc(T * D * sizeof(float));
    float *dnext      = malloc(T * D * sizeof(float));

    /* Zero SSM gradients before accumulating over batch */
    for (size_t i = 0; i < NL; i++)
        mamba_zero_grads(m->layers[i]);

    float *d_pv = malloc(D * sizeof(float));

    for (size_t b = 0; b < B; b++) {
        const float *feat  = batch->features  + b * T * F;
        const float *mask  = batch->mask      + b * T * F;
        const float *tgaps = batch->time_gaps + b * T;

        /* Pool from last SSM layer output */
        mask_timestep(mask, tmask_buf, T, F);
        mask_last_pool(acts[b] + NL * T * D, tmask_buf, patient_vec, T, D);

        float gh = 0.7f * g_hepatic[b];
        float gd = 0.3f * g_death[b];

        /* Gradient w.r.t. heads */
        for (size_t d = 0; d < D; d++) {
            gW_hepatic[d] += gh * patient_vec[d];
            gW_death[d]   += gd * patient_vec[d];
        }

        /* Gradient w.r.t. patient_vec = gh*W_hepatic + gd*W_death */
        for (size_t d = 0; d < D; d++)
            d_pv[d] = gh * m->W_hepatic[d] + gd * m->W_death[d];

        /* Unpool : copy d_pv to d_pool at last valid timestep, zero elsewhere */
        memset(d_pool, 0, T * D * sizeof(float));
        size_t last_t = 0;
        for (size_t t = 0; t < T; t++)
            if (tmask_buf[t] > 0.5f) last_t = t;
        memcpy(d_pool + last_t * D, d_pv, D * sizeof(float));

        /* Backward through SSM layers (reverse order) */
        memcpy(dcur, d_pool, T * D * sizeof(float));
        for (size_t li = NL; li-- > 0;) {
            memset(dnext, 0, T * D * sizeof(float));
            mamba_backward(m->layers[li], dcur,
                           acts[b] + li * T * D, dnext, 0);
            float *tmp = dcur; dcur = dnext; dnext = tmp;
        }
        /* dcur = d_hidden_in[T,D] = gradient w.r.t. feature projection output */

        /* Backward through feature projection */
        for (size_t t = 0; t < T; t++) {
            const float *x  = feat + t * F;
            const float *dh = dcur + t * D;
            float any = 0.0f;
            for (size_t f = 0; f < F; f++) any += mask[t * F + f];
            if (any < 0.5f) continue;  /* masked timestep — no gradient */

            /* gW_feat += dh ⊗ x  (outer product accumulate) */
            for (size_t d = 0; d < D; d++) {
                for (size_t f = 0; f < F; f++)
                    gW_feat[d * F + f] += dh[d] * x[f];
                gb_feat[d]  += dh[d];
                gW_time[d]  += dh[d] * tgaps[t];
            }
        }
    }

    m->step++;

    /* Adam update for all projection params */
    adam_update(m->W_hepatic, gW_hepatic,
                m->m_W_hepatic, m->v_W_hepatic,
                D, m->lr, m->weight_decay,
                0.9f, 0.999f, 1e-8f, m->step);
    adam_update(m->W_death, gW_death,
                m->m_W_death, m->v_W_death,
                D, m->lr, m->weight_decay,
                0.9f, 0.999f, 1e-8f, m->step);
    adam_update(m->W_feat, gW_feat,
                m->m_W_feat, m->v_W_feat,
                F * D, m->lr, m->weight_decay,
                0.9f, 0.999f, 1e-8f, m->step);
    adam_update(m->b_feat, gb_feat,
                m->m_b_feat, m->v_b_feat,
                D, m->lr, m->weight_decay,
                0.9f, 0.999f, 1e-8f, m->step);
    adam_update(m->W_time, gW_time,
                m->m_W_time, m->v_W_time,
                D, m->lr, m->weight_decay,
                0.9f, 0.999f, 1e-8f, m->step);

    /* Update SSM layers */
    for (size_t i = 0; i < NL; i++)
        mamba_optimizer_step(m->layers[i], &m->opt_cfg);

    /* Cleanup */
    for (size_t b = 0; b < B; b++) free(acts[b]);
    free(acts);
    free(patient_vec);
    free(risk_hepatic); free(risk_death);
    free(g_hepatic);    free(g_death);
    free(gW_hepatic);   free(gW_death);
    free(gW_feat);      free(gb_feat);  free(gW_time);
    free(d_pool);       free(dcur);     free(dnext);
    free(d_pv);         free(tmask_buf);

    return loss;
}

/* ------------------------------------------------------------------ */
/* Sauvegarde / chargement                                              */
/* ------------------------------------------------------------------ */

#define ANNITIA_MAGIC 0x414E4E49  /* "ANNI" */

int annitia_save(const AnnitiaModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "annitia_save: cannot open %s\n", path); return -1; }

    uint32_t magic = ANNITIA_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&m->cfg, sizeof(AnnitiaConfig), 1, f);

    size_t F = m->cfg.n_features;
    size_t D = m->cfg.dim;
    fwrite(m->W_feat,    sizeof(float), F * D, f);
    fwrite(m->b_feat,    sizeof(float), D,     f);
    fwrite(m->W_time,    sizeof(float), D,     f);
    fwrite(m->W_hepatic, sizeof(float), D,     f);
    fwrite(m->W_death,   sizeof(float), D,     f);

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        const MambaBlock *b = m->layers[i];
        size_t theta_size = b->config.state_size / 2;
        if (theta_size == 0) theta_size = 1;
        fwrite(b->W_in.data,        sizeof(float), b->W_in.rows   * b->W_in.cols,        f);
        fwrite(b->W_out.data,       sizeof(float), b->W_out.rows  * b->W_out.cols,       f);
        fwrite(b->A_log.data,       sizeof(float), b->A_log.rows  * b->A_log.cols,       f);
        fwrite(b->W_B.data,         sizeof(float), b->W_B.rows    * b->W_B.cols,         f);
        fwrite(b->W_C.data,         sizeof(float), b->W_C.rows    * b->W_C.cols,         f);
        fwrite(b->b_B,              sizeof(float), b->W_B.rows,                          f);
        fwrite(b->b_C,              sizeof(float), b->W_C.rows,                          f);
        fwrite(b->delta_proj.data,  sizeof(float), b->delta_proj.rows * b->delta_proj.cols, f);
        fwrite(b->lambda_proj.data, sizeof(float), b->lambda_proj.rows * b->lambda_proj.cols, f);
        fwrite(b->theta,            sizeof(float), theta_size,                           f);
    }

    fclose(f);
    return 0;
}

static int read_floats_f(FILE *f, float *buf, size_t n) {
    return fread(buf, sizeof(float), n, f) != n;
}

AnnitiaModel *annitia_load(const char *path, int for_training,
                            const MBOptimConfig *opt_cfg,
                            float lr, float weight_decay) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "annitia_load: cannot open %s\n", path); return NULL; }

    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, f);
    if (magic != ANNITIA_MAGIC) {
        fprintf(stderr, "annitia_load: invalid magic in %s\n", path);
        fclose(f); return NULL;
    }

    AnnitiaConfig cfg;
    fread(&cfg, sizeof(AnnitiaConfig), 1, f);

    AnnitiaModel *m = annitia_create(&cfg);

    size_t F = cfg.n_features;
    size_t D = cfg.dim;
    fread(m->W_feat,    sizeof(float), F * D, f);
    fread(m->b_feat,    sizeof(float), D,     f);
    fread(m->W_time,    sizeof(float), D,     f);
    fread(m->W_hepatic, sizeof(float), D,     f);
    fread(m->W_death,   sizeof(float), D,     f);

    for (size_t i = 0; i < cfg.n_layers; i++) {
        MambaBlock *b = m->layers[i];
        size_t theta_size = b->config.state_size / 2;
        if (theta_size == 0) theta_size = 1;
        if (read_floats_f(f, b->W_in.data,        b->W_in.rows   * b->W_in.cols)        ||
            read_floats_f(f, b->W_out.data,       b->W_out.rows  * b->W_out.cols)       ||
            read_floats_f(f, b->A_log.data,       b->A_log.rows  * b->A_log.cols)       ||
            read_floats_f(f, b->W_B.data,         b->W_B.rows    * b->W_B.cols)         ||
            read_floats_f(f, b->W_C.data,         b->W_C.rows    * b->W_C.cols)         ||
            read_floats_f(f, b->b_B,              b->W_B.rows)                          ||
            read_floats_f(f, b->b_C,              b->W_C.rows)                          ||
            read_floats_f(f, b->delta_proj.data,  b->delta_proj.rows * b->delta_proj.cols) ||
            read_floats_f(f, b->lambda_proj.data, b->lambda_proj.rows * b->lambda_proj.cols) ||
            read_floats_f(f, b->theta,            theta_size)) {
            annitia_free(m); fclose(f); return NULL;
        }
    }

    fclose(f);

    if (for_training && opt_cfg)
        annitia_enable_training(m, opt_cfg, lr, weight_decay);

    return m;
}
