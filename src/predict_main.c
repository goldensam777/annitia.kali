#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usage :
 *   ./annitia_predict --model checkpoint.bin --data data/test.bin \
 *                    --ids data/test_ids.bin --out submission.csv
 *
 * test_ids.bin format (written by csv_parser):
 *   uint32_t  n_ids
 *   int32_t   ids[n_ids]
 */

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
    const char *model_path = NULL;
    const char *data_path  = NULL;
    const char *ids_path   = NULL;
    const char *out_path   = "submission.csv";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")) model_path = argv[++i];
        else if (!strcmp(argv[i], "--data"))  data_path  = argv[++i];
        else if (!strcmp(argv[i], "--ids"))   ids_path   = argv[++i];
        else if (!strcmp(argv[i], "--out"))   out_path   = argv[++i];
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    if (!model_path || !data_path) {
        fprintf(stderr,
            "Usage: %s --model <file> --data <file> [--ids <file>] [--out <file>]\n",
            argv[0]);
        return 1;
    }

    AnnitiaModel *model = annitia_load(model_path, 0, NULL, 0.0f, 0.0f);
    if (!model) return 1;

    MasldDataset *ds = masld_load(data_path);
    if (!ds) { annitia_free(model); return 1; }

    size_t N = masld_n_patients(ds);
    size_t T = ANNITIA_MAX_TIMESTEPS;
    size_t F = ANNITIA_N_FEATURES;

    /* Load trustii IDs if available */
    int32_t *trustii_ids = NULL;
    size_t n_ids = 0;
    if (ids_path) {
        trustii_ids = load_trustii_ids(ids_path, &n_ids);
        if (!trustii_ids) {
            fprintf(stderr, "Warning: could not load %s — using indices\n", ids_path);
        } else if (n_ids != N) {
            fprintf(stderr, "Warning: ids count %zu != patients %zu — using indices\n",
                    n_ids, N);
            free(trustii_ids); trustii_ids = NULL;
        }
    }

    SurvivalBatch *batch = masld_batch_alloc(N, T, F);
    masld_get_batch(ds, batch, 0, N);

    float *rh = malloc(N * sizeof(float));
    float *rd = malloc(N * sizeof(float));

    annitia_forward(model, batch, rh, rd);

    /* C-index if labels present (test set has no events → will print 0.5) */
    int has_labels = 0;
    for (size_t i = 0; i < N && !has_labels; i++)
        if (batch->event_hepatic[i] || batch->event_death[i]) has_labels = 1;

    if (has_labels) {
        float ci_hep = c_index(rh, batch->time_hepatic, batch->event_hepatic, N);
        float ci_dth = c_index(rd, batch->time_death,   batch->event_death,   N);
        printf("C-index hepatic = %.4f\n", ci_hep);
        printf("C-index death   = %.4f\n", ci_dth);
        printf("Score final     = %.4f\n", 0.7f * ci_hep + 0.3f * ci_dth);
    }

    /* Export CSV — format Trustii: trustii_id,risk_hepatic_event,risk_death */
    FILE *f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }
    fprintf(f, "trustii_id,risk_hepatic_event,risk_death\n");
    for (size_t i = 0; i < N; i++) {
        if (trustii_ids)
            fprintf(f, "%d,%.6f,%.6f\n", (int)trustii_ids[i], rh[i], rd[i]);
        else
            fprintf(f, "%zu,%.6f,%.6f\n", i, rh[i], rd[i]);
    }
    fclose(f);
    printf("Predictions saved to %s (%zu patients)\n", out_path, N);

    free(trustii_ids);
    free(rh); free(rd);
    masld_batch_free(batch);
    masld_free(ds);
    annitia_free(model);
    return 0;
}
