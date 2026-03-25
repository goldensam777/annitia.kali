#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Format binaire produit par preprocess.py :
 *
 * Header (24 bytes) :
 *   uint32_t magic        = 0x4D41534C ("MASL")
 *   uint32_t n_patients
 *   uint32_t seq_len      (T, padded)
 *   uint32_t n_features   (F)
 *   uint32_t version      = 1
 *   uint32_t _pad
 *
 * Pour chaque patient (row-major) :
 *   float    features[T * F]       — valeurs normalisées (0 si manquant)
 *   float    mask[T * F]           — 1=valide, 0=manquant
 *   float    time_gaps[T]          — jours depuis visite précédente
 *   int32_t  n_visits              — nombre réel de visites
 *   float    time_hepatic          — temps jusqu'à événement hépatique
 *   uint8_t  event_hepatic         — 1=événement, 0=censuré
 *   uint8_t  _pad1
 *   float    time_death            — temps jusqu'au décès
 *   uint8_t  event_death           — 1=décès, 0=censuré
 *   uint8_t  _pad2[3]
 * ------------------------------------------------------------------ */


struct MasldDataset {
    size_t   n_patients;
    size_t   seq_len;
    size_t   n_features;

    float   *features;      /* [N, T, F] */
    float   *mask;          /* [N, T, F] */
    float   *time_gaps;     /* [N, T]    */
    int     *n_visits;      /* [N]       */
    float   *time_hepatic;  /* [N]       */
    uint8_t *event_hepatic; /* [N]       */
    float   *time_death;    /* [N]       */
    uint8_t *event_death;   /* [N]       */
};

MasldDataset *masld_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "masld_load: cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic, n_patients, seq_len, n_features, version, _pad;
    if (fread(&magic,      sizeof(uint32_t), 1, f) != 1 || magic != MASLD_MAGIC ||
        fread(&n_patients, sizeof(uint32_t), 1, f) != 1 ||
        fread(&seq_len,    sizeof(uint32_t), 1, f) != 1 ||
        fread(&n_features, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version,    sizeof(uint32_t), 1, f) != 1 ||
        fread(&_pad,       sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "masld_load: invalid header in %s\n", path);
        fclose(f);
        return NULL;
    }

    MasldDataset *ds = calloc(1, sizeof(MasldDataset));
    ds->n_patients = n_patients;
    ds->seq_len    = seq_len;
    ds->n_features = n_features;

    size_t TF = seq_len * n_features;
    ds->features      = malloc(n_patients * TF * sizeof(float));
    ds->mask          = malloc(n_patients * TF * sizeof(float));
    ds->time_gaps     = malloc(n_patients * seq_len * sizeof(float));
    ds->n_visits      = malloc(n_patients * sizeof(int));
    ds->time_hepatic  = malloc(n_patients * sizeof(float));
    ds->event_hepatic = malloc(n_patients * sizeof(uint8_t));
    ds->time_death    = malloc(n_patients * sizeof(float));
    ds->event_death   = malloc(n_patients * sizeof(uint8_t));

    for (size_t i = 0; i < n_patients; i++) {
        uint8_t pad1, pad2[3];
        int32_t nv;

        fread(ds->features    + i * TF,       sizeof(float),   TF,       f);
        fread(ds->mask        + i * TF,       sizeof(float),   TF,       f);
        fread(ds->time_gaps   + i * seq_len,  sizeof(float),   seq_len,  f);
        fread(&nv,                             sizeof(int32_t), 1,        f);
        ds->n_visits[i] = (int)nv;
        fread(&ds->time_hepatic[i],            sizeof(float),   1,        f);
        fread(&ds->event_hepatic[i],           sizeof(uint8_t), 1,        f);
        fread(&pad1,                           sizeof(uint8_t), 1,        f);
        fread(&ds->time_death[i],              sizeof(float),   1,        f);
        fread(&ds->event_death[i],             sizeof(uint8_t), 1,        f);
        fread(pad2,                            sizeof(uint8_t), 3,        f);
    }

    fclose(f);
    return ds;
}

void masld_free(MasldDataset *ds) {
    if (!ds) return;
    free(ds->features);
    free(ds->mask);
    free(ds->time_gaps);
    free(ds->n_visits);
    free(ds->time_hepatic);
    free(ds->event_hepatic);
    free(ds->time_death);
    free(ds->event_death);
    free(ds);
}

size_t masld_n_patients(const MasldDataset *ds) {
    return ds ? ds->n_patients : 0;
}

SurvivalBatch *masld_batch_alloc(size_t batch_size, size_t seq_len, size_t n_features) {
    SurvivalBatch *b = calloc(1, sizeof(SurvivalBatch));
    size_t TF = seq_len * n_features;
    b->features      = malloc(batch_size * TF * sizeof(float));
    b->mask          = malloc(batch_size * TF * sizeof(float));
    b->time_gaps     = malloc(batch_size * seq_len * sizeof(float));
    b->n_visits      = malloc(batch_size * sizeof(int));
    b->time_hepatic  = malloc(batch_size * sizeof(float));
    b->event_hepatic = malloc(batch_size * sizeof(uint8_t));
    b->time_death    = malloc(batch_size * sizeof(float));
    b->event_death   = malloc(batch_size * sizeof(uint8_t));
    b->batch_size    = batch_size;
    b->seq_len       = seq_len;
    b->n_features    = n_features;
    return b;
}

void masld_batch_free(SurvivalBatch *b) {
    if (!b) return;
    free(b->features);
    free(b->mask);
    free(b->time_gaps);
    free(b->n_visits);
    free(b->time_hepatic);
    free(b->event_hepatic);
    free(b->time_death);
    free(b->event_death);
    free(b);
}

void masld_get_batch_idx(const MasldDataset *ds, SurvivalBatch *batch,
                          const size_t *idx, size_t batch_size) {
    size_t T  = ds->seq_len;
    size_t F  = ds->n_features;
    size_t TF = T * F;

    batch->batch_size = batch_size;
    batch->seq_len    = T;
    batch->n_features = F;

    for (size_t i = 0; i < batch_size; i++) {
        size_t src = idx[i];
        memcpy(batch->features   + i * TF,  ds->features   + src * TF,  TF * sizeof(float));
        memcpy(batch->mask       + i * TF,  ds->mask       + src * TF,  TF * sizeof(float));
        memcpy(batch->time_gaps  + i * T,   ds->time_gaps  + src * T,   T  * sizeof(float));
        batch->n_visits[i]       = ds->n_visits[src];
        batch->time_hepatic[i]   = ds->time_hepatic[src];
        batch->event_hepatic[i]  = ds->event_hepatic[src];
        batch->time_death[i]     = ds->time_death[src];
        batch->event_death[i]    = ds->event_death[src];
    }
}

void masld_get_batch(const MasldDataset *ds, SurvivalBatch *batch,
                     size_t start, size_t batch_size) {
    size_t T  = ds->seq_len;
    size_t F  = ds->n_features;
    size_t TF = T * F;

    if (start + batch_size > ds->n_patients)
        batch_size = ds->n_patients - start;

    batch->batch_size = batch_size;
    batch->seq_len    = T;
    batch->n_features = F;

    for (size_t i = 0; i < batch_size; i++) {
        size_t src = start + i;
        memcpy(batch->features   + i * TF,  ds->features   + src * TF,  TF * sizeof(float));
        memcpy(batch->mask       + i * TF,  ds->mask       + src * TF,  TF * sizeof(float));
        memcpy(batch->time_gaps  + i * T,   ds->time_gaps  + src * T,   T  * sizeof(float));
        batch->n_visits[i]       = ds->n_visits[src];
        batch->time_hepatic[i]   = ds->time_hepatic[src];
        batch->event_hepatic[i]  = ds->event_hepatic[src];
        batch->time_death[i]     = ds->time_death[src];
        batch->event_death[i]    = ds->event_death[src];
    }
}
