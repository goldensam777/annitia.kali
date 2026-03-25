/*
 * ensemble_main.c — Moyenne les prédictions de plusieurs modèles ANNITIA.
 *
 * Usage :
 *   ./annitia_ensemble --data test.bin --ids test_ids.bin --out submission.csv \
 *                      model1.bin model2.bin model3.bin ...
 *
 * Chaque modèle produit des scores de risque ; on fait la moyenne arithmétique.
 * Cela stabilise les prédictions face à la variance de l'initialisation.
 */

#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t *load_trustii_ids(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t n;
    if (fread(&n, sizeof(n), 1, f) != 1) { fclose(f); return NULL; }
    int32_t *ids = malloc(n * sizeof(int32_t));
    if (fread(ids, sizeof(int32_t), n, f) != n) {
        free(ids); fclose(f); return NULL;
    }
    fclose(f);
    *out_n = n;
    return ids;
}

int main(int argc, char **argv) {
    const char *data_path = NULL;
    const char *ids_path  = NULL;
    const char *out_path  = "submission.csv";

    /* Parse options (avant les chemins de modèles) */
    int model_start = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--data")) { data_path = argv[++i]; model_start = i + 1; }
        else if (!strcmp(argv[i], "--ids"))  { ids_path  = argv[++i]; model_start = i + 1; }
        else if (!strcmp(argv[i], "--out"))  { out_path  = argv[++i]; model_start = i + 1; }
        else break;
    }

    int n_models = argc - model_start;
    if (!data_path || n_models <= 0) {
        fprintf(stderr,
            "Usage: %s --data <file> [--ids <file>] [--out <file>] model1.bin ...\n",
            argv[0]);
        return 1;
    }

    MasldDataset *ds = masld_load(data_path);
    if (!ds) return 1;

    size_t N = masld_n_patients(ds);
    size_t T = ANNITIA_MAX_TIMESTEPS;
    size_t F = ANNITIA_N_FEATURES;

    int32_t *trustii_ids = NULL;
    size_t n_ids = 0;
    if (ids_path) {
        trustii_ids = load_trustii_ids(ids_path, &n_ids);
        if (trustii_ids && n_ids != N) {
            fprintf(stderr, "Warning: ids count %zu != patients %zu\n", n_ids, N);
            free(trustii_ids); trustii_ids = NULL;
        }
    }

    SurvivalBatch *batch = masld_batch_alloc(N, T, F);
    masld_get_batch(ds, batch, 0, N);

    float *rh_sum = calloc(N, sizeof(float));
    float *rd_sum = calloc(N, sizeof(float));
    float *rh_tmp = malloc(N * sizeof(float));
    float *rd_tmp = malloc(N * sizeof(float));

    int loaded = 0;
    for (int mi = 0; mi < n_models; mi++) {
        const char *mpath = argv[model_start + mi];
        AnnitiaModel *model = annitia_load(mpath, 0, NULL, 0.0f, 0.0f);
        if (!model) {
            fprintf(stderr, "Warning: cannot load %s — skipping\n", mpath);
            continue;
        }
        annitia_forward(model, batch, rh_tmp, rd_tmp);
        for (size_t i = 0; i < N; i++) {
            rh_sum[i] += rh_tmp[i];
            rd_sum[i] += rd_tmp[i];
        }
        annitia_free(model);
        loaded++;
        printf("Loaded model %d/%d: %s\n", loaded, n_models, mpath);
    }

    if (loaded == 0) { fprintf(stderr, "No models loaded\n"); return 1; }

    float inv = 1.0f / (float)loaded;
    for (size_t i = 0; i < N; i++) {
        rh_sum[i] *= inv;
        rd_sum[i] *= inv;
    }

    FILE *f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }
    fprintf(f, "trustii_id,risk_hepatic_event,risk_death\n");
    for (size_t i = 0; i < N; i++) {
        if (trustii_ids)
            fprintf(f, "%d,%.6f,%.6f\n", (int)trustii_ids[i], rh_sum[i], rd_sum[i]);
        else
            fprintf(f, "%zu,%.6f,%.6f\n", i, rh_sum[i], rd_sum[i]);
    }
    fclose(f);
    printf("Ensemble (%d modèles) → %s (%zu patients)\n", loaded, out_path, N);

    free(rh_sum); free(rd_sum); free(rh_tmp); free(rd_tmp);
    free(trustii_ids);
    masld_batch_free(batch);
    masld_free(ds);
    return 0;
}
