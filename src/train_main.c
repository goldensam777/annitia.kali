#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usage :
 *   ./annitia_train --train data/train.bin --val data/val.bin \
 *                  --out checkpoint.bin [options]
 *
 * Options :
 *   --dim       D     (défaut: 64)
 *   --state     N     (défaut: 16)
 *   --layers    L     (défaut: 2)
 *   --mimo      R     (défaut: 1)
 *   --epochs    E     (défaut: 50)
 *   --batch     B     (défaut: 32)
 *   --lr        lr    (défaut: 1e-3)
 *   --wd        wd    (défaut: 1e-4)
 *   --seed      s     (défaut: 42)
 */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --train <file> --val <file> --out <file> [options]\n"
        "  --dim D --state N --layers L --mimo R\n"
        "  --epochs E --batch B --lr X --wd X --seed S\n",
        prog);
    exit(1);
}

/* Fisher-Yates shuffle on size_t array */
static void shuffle_idx(size_t *idx, size_t n, unsigned *seed) {
    for (size_t i = n - 1; i > 0; i--) {
        unsigned r = (unsigned)rand_r(seed);
        size_t j = (size_t)(r % (unsigned)(i + 1));
        size_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *val_path   = NULL;
    const char *out_path   = "checkpoint.bin";
    size_t dim     = 64;
    size_t state   = 16;
    size_t layers  = 2;
    size_t mimo    = 1;
    size_t epochs  = 50;
    size_t batch   = 32;
    float  lr      = 1e-3f;
    float  wd      = 1e-4f;
    unsigned long seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--train"))  train_path = argv[++i];
        else if (!strcmp(argv[i], "--val"))    val_path   = argv[++i];
        else if (!strcmp(argv[i], "--out"))    out_path   = argv[++i];
        else if (!strcmp(argv[i], "--dim"))    dim        = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--state"))  state      = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers")) layers     = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mimo"))   mimo       = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--epochs")) epochs     = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch"))  batch      = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr"))     lr         = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--wd"))     wd         = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))   seed       = (unsigned long)atol(argv[++i]);
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!train_path || !val_path) usage(argv[0]);

    MasldDataset *ds_train = masld_load(train_path);
    MasldDataset *ds_val   = masld_load(val_path);
    if (!ds_train || !ds_val) { fprintf(stderr, "Failed to load data\n"); return 1; }

    size_t N_train = masld_n_patients(ds_train);
    size_t N_val   = masld_n_patients(ds_val);
    size_t T = ANNITIA_MAX_TIMESTEPS;
    size_t F = ANNITIA_N_FEATURES;

    printf("Train: %zu patients | Val: %zu patients\n", N_train, N_val);

    AnnitiaConfig cfg = {
        .n_features = F,
        .dim        = dim,
        .state_size = state,
        .seq_len    = T,
        .n_layers   = layers,
        .mimo_rank  = mimo,
        .dt_scale   = 1.0f,
        .dt_min     = 0.001f,
        .dt_max     = 0.1f,
    };

    AnnitiaModel *model = annitia_create(&cfg);
    annitia_init(model, seed);

    MBOptimConfig opt = {
        .lr           = lr,
        .mu           = 0.9f,
        .beta2        = 0.999f,
        .eps          = 1e-8f,
        .clip_norm    = 1.0f,
        .weight_decay = wd,
    };
    annitia_enable_training(model, &opt, lr, wd);

    /* Shuffle indices */
    size_t *perm = malloc(N_train * sizeof(size_t));
    for (size_t i = 0; i < N_train; i++) perm[i] = i;
    unsigned shuf_seed = (unsigned)seed;

    int has_val = (N_val > 0);

    SurvivalBatch *b_train = masld_batch_alloc(batch, T, F);
    SurvivalBatch *b_val   = has_val ? masld_batch_alloc(N_val, T, F) : NULL;
    if (has_val) masld_get_batch(ds_val, b_val, 0, N_val);

    float *rh_val = has_val ? malloc(N_val * sizeof(float)) : NULL;
    float *rd_val = has_val ? malloc(N_val * sizeof(float)) : NULL;

    float best_score = -1.0f;

    if (has_val) {
        printf("\nEpoch | Train Loss | C-idx Hep | C-idx Dth | Score\n");
        printf("------|------------|-----------|-----------|------\n");
    } else {
        printf("\nEpoch | Train Loss  (pas de validation)\n");
        printf("------|-----------\n");
    }

    for (size_t ep = 0; ep < epochs; ep++) {
        shuffle_idx(perm, N_train, &shuf_seed);

        float total_loss = 0.0f;
        size_t n_batches = 0;

        for (size_t start = 0; start + batch <= N_train; start += batch) {
            masld_get_batch_idx(ds_train, b_train, perm + start, batch);
            total_loss += annitia_train_step(model, b_train);
            n_batches++;
        }
        float avg_loss = n_batches > 0 ? total_loss / (float)n_batches : 0.0f;

        if (has_val) {
            annitia_forward(model, b_val, rh_val, rd_val);
            float ci_hep = c_index(rh_val, b_val->time_hepatic, b_val->event_hepatic, N_val);
            float ci_dth = c_index(rd_val, b_val->time_death,   b_val->event_death,   N_val);
            float score  = 0.7f * ci_hep + 0.3f * ci_dth;

            char flag = ' ';
            if (score > best_score) {
                best_score = score;
                annitia_save(model, out_path);
                flag = '*';
            }
            printf("%5zu | %10.4f | %9.4f | %9.4f | %.4f %c\n",
                   ep + 1, avg_loss, ci_hep, ci_dth, score, flag);
        } else {
            /* Sans validation : sauvegarder à chaque époque */
            annitia_save(model, out_path);
            printf("%5zu | %10.4f\n", ep + 1, avg_loss);
        }
    }

    if (has_val)
        printf("\nBest: %.4f — checkpoint: %s\n", best_score, out_path);
    else
        printf("\nCheckpoint final: %s\n", out_path);

    free(perm);
    free(rh_val); free(rd_val);
    masld_batch_free(b_train);
    if (b_val) masld_batch_free(b_val);
    masld_free(ds_train);
    masld_free(ds_val);
    annitia_free(model);
    return 0;
}
