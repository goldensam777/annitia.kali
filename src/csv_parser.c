/*
 * csv_parser.c — ANNITIA Challenge
 * Parse le CSV wide ANNITIA (format réel) et produit des fichiers binaires
 * au format MASL pour masld_data.c.
 *
 * Usage (binaire annitia_preprocess) :
 *   ./annitia_preprocess --train DB-train.csv --test DB-test.csv --out data/
 *
 * Produit :
 *   data/train.bin, data/val.bin   — patients train (split 80/20)
 *   data/test.bin                  — patients test (sans targets)
 *   data/norm.bin                  — stats de normalisation (mean/std)
 *   data/test_ids.bin              — trustii_ids du test CSV
 */

#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Constantes                                                           */
/* ------------------------------------------------------------------ */

#define MAX_COLS      512
#define MAX_COL_LEN   128
#define MAX_LINE      65536
#define MAX_PATIENTS  4096
#define VAL_RATIO_PCT 20     /* 20% validation */
#define NORM_MAGIC    0x4E4F524D  /* "NORM" */

/* Features dynamiques par visite (dans l'ordre des colonnes features[t]) */
static const char *DYN_PREFIX[ANNITIA_DYN_FEATURES] = {
    "BMI_v",
    "alt_v", "ast_v", "bilirubin_v", "chol_v", "ggt_v",
    "gluc_fast_v", "plt_v", "triglyc_v",
    "aixp_aix_result_BM_3_v",
    "fibrotest_BM_2_v",
    "fibs_stiffness_med_BM_1_v",
};

/* Features statiques (une valeur par patient, répétée à chaque visite) */
static const char *STAT_COL[ANNITIA_STAT_FEATURES] = {
    "gender", "T2DM", "Hypertension", "Dyslipidaemia",
    "bariatric_surgery", "bariatric_surgery_age",
};

/* Colonnes spéciales */
static const char *AGE_PREFIX        = "Age_v";
static const char *COL_EVENT_HEP     = "evenements_hepatiques_majeurs";
static const char *COL_AGE_HEP       = "evenements_hepatiques_age_occur";
static const char *COL_EVENT_DTH     = "death";
static const char *COL_AGE_DTH       = "death_age_occur";
static const char *COL_TRUSTII_ID    = "trustii_id";

/* ------------------------------------------------------------------ */
/* Structures internes                                                  */
/* ------------------------------------------------------------------ */

/* Index des colonnes dans le CSV (rempli au parsing du header) */
typedef struct {
    /* dyn_idx[f][v] = index colonne pour feature f à visite v (0-based) */
    int dyn_idx[ANNITIA_DYN_FEATURES][ANNITIA_MAX_TIMESTEPS];
    int stat_idx[ANNITIA_STAT_FEATURES];
    int age_idx[ANNITIA_MAX_TIMESTEPS];
    int event_hep_idx;
    int age_hep_idx;
    int event_dth_idx;
    int age_dth_idx;
    int trustii_id_idx;
    int n_cols;
} ColIndex;

/* Un patient brut (avant normalisation) */
typedef struct {
    float features[ANNITIA_MAX_TIMESTEPS][ANNITIA_N_FEATURES];
    float mask[ANNITIA_MAX_TIMESTEPS][ANNITIA_N_FEATURES];
    float time_gaps[ANNITIA_MAX_TIMESTEPS];
    int   n_visits;
    float time_hep;
    uint8_t event_hep;
    float time_dth;
    uint8_t event_dth;
    int   trustii_id;
} RawPatient;

/* Stats de normalisation (mean/std par feature dynamique) */
typedef struct {
    float mean[ANNITIA_DYN_FEATURES];
    float std[ANNITIA_DYN_FEATURES];
} NormStats;

/* ------------------------------------------------------------------ */
/* Utilitaires CSV                                                      */
/* ------------------------------------------------------------------ */

/* Lit un champ CSV (gère les virgules et les champs vides).
 * Retourne le pointeur après la virgule ou \0. */
static const char *csv_next_field(const char *p, char *out, size_t out_size) {
    size_t i = 0;
    while (*p && *p != ',' && *p != '\n' && *p != '\r') {
        if (i < out_size - 1) out[i++] = *p;
        p++;
    }
    out[i] = '\0';
    if (*p == ',') p++;
    return p;
}

/* Parse un float depuis un champ CSV.
 * Retourne NAN si champ vide ou non numérique. */
static float parse_float(const char *s) {
    if (!s || s[0] == '\0') return NAN;
    char *end;
    float v = strtof(s, &end);
    if (end == s) return NAN;
    return v;
}

/* Parse un int depuis un champ CSV. Retourne -1 si invalide. */
static int parse_int(const char *s) {
    if (!s || s[0] == '\0') return -1;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s) return -1;
    return (int)v;
}

/* ------------------------------------------------------------------ */
/* Parsing du header                                                    */
/* ------------------------------------------------------------------ */

static int parse_header(const char *line, ColIndex *ci) {
    memset(ci, -1, sizeof(ColIndex));  /* -1 = colonne non trouvée */

    char field[MAX_COL_LEN];
    const char *p = line;
    int col = 0;

    while (*p && *p != '\n') {
        p = csv_next_field(p, field, sizeof(field));

        /* Features dynamiques : cherche "PREFIX_vN" */
        for (int f = 0; f < ANNITIA_DYN_FEATURES; f++) {
            size_t plen = strlen(DYN_PREFIX[f]);
            if (strncmp(field, DYN_PREFIX[f], plen) == 0) {
                int v = parse_int(field + plen);
                if (v >= 1 && v <= ANNITIA_MAX_TIMESTEPS)
                    ci->dyn_idx[f][v - 1] = col;
                break;
            }
        }

        /* Features statiques */
        for (int f = 0; f < ANNITIA_STAT_FEATURES; f++) {
            if (strcmp(field, STAT_COL[f]) == 0)
                ci->stat_idx[f] = col;
        }

        /* Age_vN */
        if (strncmp(field, AGE_PREFIX, strlen(AGE_PREFIX)) == 0) {
            int v = parse_int(field + strlen(AGE_PREFIX));
            if (v >= 1 && v <= ANNITIA_MAX_TIMESTEPS)
                ci->age_idx[v - 1] = col;
        }

        /* Targets */
        if (strcmp(field, COL_EVENT_HEP)  == 0) ci->event_hep_idx  = col;
        if (strcmp(field, COL_AGE_HEP)    == 0) ci->age_hep_idx    = col;
        if (strcmp(field, COL_EVENT_DTH)  == 0) ci->event_dth_idx  = col;
        if (strcmp(field, COL_AGE_DTH)    == 0) ci->age_dth_idx    = col;
        if (strcmp(field, COL_TRUSTII_ID) == 0) ci->trustii_id_idx = col;

        col++;
    }

    ci->n_cols = col;
    return col;
}

/* ------------------------------------------------------------------ */
/* Parsing d'une ligne patient                                          */
/* ------------------------------------------------------------------ */

static int parse_patient(const char *line, const ColIndex *ci, RawPatient *p) {
    /* Extraire tous les champs dans un tableau */
    char fields[MAX_COLS][MAX_COL_LEN];
    int n_fields = 0;
    const char *ptr = line;

    while (*ptr && *ptr != '\n' && *ptr != '\r' && n_fields < MAX_COLS) {
        ptr = csv_next_field(ptr, fields[n_fields], MAX_COL_LEN);
        n_fields++;
    }

    if (n_fields < 4) return 0;  /* ligne vide ou invalide */

    memset(p, 0, sizeof(RawPatient));

    /* Age à la première visite (baseline) */
    float age_v1 = NAN;
    if (ci->age_idx[0] >= 0 && ci->age_idx[0] < n_fields)
        age_v1 = parse_float(fields[ci->age_idx[0]]);
    if (isnan(age_v1)) age_v1 = 0.0f;

    /* Dernière visite observée = max(Age_v1..v22) */
    float last_age = age_v1;
    for (int t = 0; t < ANNITIA_MAX_TIMESTEPS; t++) {
        if (ci->age_idx[t] >= 0 && ci->age_idx[t] < n_fields) {
            float a = parse_float(fields[ci->age_idx[t]]);
            if (!isnan(a) && a > last_age) last_age = a;
        }
    }

    /* Features statiques (valeurs fixes) */
    float stat_vals[ANNITIA_STAT_FEATURES];
    for (int f = 0; f < ANNITIA_STAT_FEATURES; f++) {
        stat_vals[f] = NAN;
        if (ci->stat_idx[f] >= 0 && ci->stat_idx[f] < n_fields)
            stat_vals[f] = parse_float(fields[ci->stat_idx[f]]);
    }

    /* Construction des timesteps */
    int n_valid = 0;
    for (int t = 0; t < ANNITIA_MAX_TIMESTEPS; t++) {
        /* Time gap depuis baseline (en années) */
        if (ci->age_idx[t] >= 0 && ci->age_idx[t] < n_fields) {
            float at = parse_float(fields[ci->age_idx[t]]);
            if (!isnan(at)) p->time_gaps[t] = at - age_v1;
        }

        /* Features dynamiques */
        int has_dyn = 0;
        for (int f = 0; f < ANNITIA_DYN_FEATURES; f++) {
            int col = ci->dyn_idx[f][t];
            if (col >= 0 && col < n_fields) {
                float v = parse_float(fields[col]);
                if (!isnan(v)) {
                    p->features[t][f] = v;
                    p->mask[t][f]     = 1.0f;
                    has_dyn           = 1;
                }
            }
        }

        /* Features statiques (répétées si timestep valide) */
        if (has_dyn) {
            n_valid++;
            for (int f = 0; f < ANNITIA_STAT_FEATURES; f++) {
                if (!isnan(stat_vals[f])) {
                    p->features[t][ANNITIA_DYN_FEATURES + f] = stat_vals[f];
                    p->mask[t][ANNITIA_DYN_FEATURES + f]     = 1.0f;
                }
            }
        }
    }

    p->n_visits = n_valid;
    if (n_valid == 0) return 0;  /* patient sans aucune visite */

    /* Targets (train seulement) */
    if (ci->event_hep_idx >= 0 && ci->event_hep_idx < n_fields) {
        float ev = parse_float(fields[ci->event_hep_idx]);
        p->event_hep = (uint8_t)((!isnan(ev) && ev > 0.5f) ? 1 : 0);

        float age_occur = NAN;
        if (ci->age_hep_idx >= 0 && ci->age_hep_idx < n_fields)
            age_occur = parse_float(fields[ci->age_hep_idx]);

        if (p->event_hep && !isnan(age_occur))
            p->time_hep = age_occur - age_v1;
        else
            p->time_hep = last_age - age_v1;
        if (p->time_hep < 0.001f) p->time_hep = 0.001f;
    }

    if (ci->event_dth_idx >= 0 && ci->event_dth_idx < n_fields) {
        float ev = parse_float(fields[ci->event_dth_idx]);
        if (isnan(ev)) {
            /* Outcome inconnu → exclure du calcul de loss death */
            p->event_dth = 0;
            p->time_dth  = last_age - age_v1;
        } else {
            p->event_dth = (uint8_t)(ev > 0.5f ? 1 : 0);
            float age_occur = NAN;
            if (ci->age_dth_idx >= 0 && ci->age_dth_idx < n_fields)
                age_occur = parse_float(fields[ci->age_dth_idx]);

            if (p->event_dth && !isnan(age_occur))
                p->time_dth = age_occur - age_v1;
            else
                p->time_dth = last_age - age_v1;
        }
        if (p->time_dth < 0.001f) p->time_dth = 0.001f;
    }

    /* trustii_id (test CSV) */
    if (ci->trustii_id_idx >= 0 && ci->trustii_id_idx < n_fields) {
        int tid = parse_int(fields[ci->trustii_id_idx]);
        p->trustii_id = (tid >= 0) ? tid : 0;
    }

    return 1;
}

/* ------------------------------------------------------------------ */
/* Chargement d'un CSV complet                                          */
/* ------------------------------------------------------------------ */

static int load_csv(const char *path, RawPatient *patients, int max_patients) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Impossible d'ouvrir : %s\n", path); return -1; }

    char line[MAX_LINE];
    ColIndex ci;

    /* Header */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    /* Supprimer le BOM UTF-8 si présent */
    char *start = line;
    if ((unsigned char)start[0] == 0xEF &&
        (unsigned char)start[1] == 0xBB &&
        (unsigned char)start[2] == 0xBF) start += 3;

    parse_header(start, &ci);

    int n = 0;
    while (fgets(line, sizeof(line), f) && n < max_patients) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (parse_patient(line, &ci, &patients[n]))
            n++;
    }

    fclose(f);
    printf("  %s : %d patients chargés\n", path, n);
    return n;
}

/* ------------------------------------------------------------------ */
/* Calcul des stats de normalisation (features dynamiques uniquement)  */
/* ------------------------------------------------------------------ */

static void compute_norm_stats(const RawPatient *patients, int n, NormStats *stats) {
    double sum[ANNITIA_DYN_FEATURES]  = {0};
    double sum2[ANNITIA_DYN_FEATURES] = {0};
    long   cnt[ANNITIA_DYN_FEATURES]  = {0};

    for (int i = 0; i < n; i++) {
        for (int t = 0; t < ANNITIA_MAX_TIMESTEPS; t++) {
            for (int f = 0; f < ANNITIA_DYN_FEATURES; f++) {
                if (patients[i].mask[t][f] > 0.5f) {
                    double v = patients[i].features[t][f];
                    sum[f]  += v;
                    sum2[f] += v * v;
                    cnt[f]++;
                }
            }
        }
    }

    for (int f = 0; f < ANNITIA_DYN_FEATURES; f++) {
        if (cnt[f] < 2) {
            stats->mean[f] = 0.0f;
            stats->std[f]  = 1.0f;
        } else {
            double mean = sum[f] / cnt[f];
            double var  = sum2[f] / cnt[f] - mean * mean;
            stats->mean[f] = (float)mean;
            stats->std[f]  = (float)(var > 1e-8 ? sqrt(var) : 1.0);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Application de la normalisation (in-place)                          */
/* ------------------------------------------------------------------ */

static void apply_norm(RawPatient *patients, int n, const NormStats *stats) {
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < ANNITIA_MAX_TIMESTEPS; t++) {
            for (int f = 0; f < ANNITIA_DYN_FEATURES; f++) {
                if (patients[i].mask[t][f] > 0.5f) {
                    patients[i].features[t][f] =
                        (patients[i].features[t][f] - stats->mean[f]) / stats->std[f];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Écriture binaire MASL                                               */
/* ------------------------------------------------------------------ */

static void write_bin(const char *path, const RawPatient *patients, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Impossible d'écrire : %s\n", path); return; }

    uint32_t magic = MASLD_MAGIC;
    uint32_t np    = (uint32_t)n;
    uint32_t T     = ANNITIA_MAX_TIMESTEPS;
    uint32_t F     = ANNITIA_N_FEATURES;
    uint32_t ver   = 1;
    uint32_t pad   = 0;

    fwrite(&magic, 4, 1, f);
    fwrite(&np,    4, 1, f);
    fwrite(&T,     4, 1, f);
    fwrite(&F,     4, 1, f);
    fwrite(&ver,   4, 1, f);
    fwrite(&pad,   4, 1, f);

    for (int i = 0; i < n; i++) {
        const RawPatient *p = &patients[i];
        fwrite(p->features,  sizeof(float),   T * F, f);
        fwrite(p->mask,      sizeof(float),   T * F, f);
        fwrite(p->time_gaps, sizeof(float),   T,     f);
        int32_t nv = p->n_visits;
        fwrite(&nv,           sizeof(int32_t), 1,     f);
        fwrite(&p->time_hep,  sizeof(float),   1,     f);
        uint8_t tmp2[2] = { p->event_hep, 0 };
        fwrite(tmp2,          1, 2, f);
        fwrite(&p->time_dth,  sizeof(float),   1,     f);
        uint8_t tmp4[4] = { p->event_dth, 0, 0, 0 };
        fwrite(tmp4,          1, 4, f);
    }

    fclose(f);

    long size_kb = (long)ftell(f);  /* après fclose, ftell retourne -1 mais c'est OK */
    /* Calcul approximatif */
    long approx_kb = ((long)n * (ANNITIA_MAX_TIMESTEPS * ANNITIA_N_FEATURES * 2 * 4
                      + ANNITIA_MAX_TIMESTEPS * 4 + 24)) / 1024;
    printf("  %s : %d patients (~%ld KB)\n", path, n, approx_kb);
    (void)size_kb;
}

/* ------------------------------------------------------------------ */
/* Sauvegarde des stats de normalisation                               */
/* ------------------------------------------------------------------ */

static void write_norm(const char *path, const NormStats *stats) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = NORM_MAGIC;
    fwrite(&magic,       4, 1, f);
    fwrite(stats->mean,  sizeof(float), ANNITIA_DYN_FEATURES, f);
    fwrite(stats->std,   sizeof(float), ANNITIA_DYN_FEATURES, f);
    fclose(f);
    printf("  %s : stats de normalisation sauvegardées\n", path);
}

int annitia_load_norm(const char *path, NormStats *stats) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != NORM_MAGIC) { fclose(f); return -1; }
    fread(stats->mean, sizeof(float), ANNITIA_DYN_FEATURES, f);
    fread(stats->std,  sizeof(float), ANNITIA_DYN_FEATURES, f);
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Sauvegarde des trustii_ids (test)                                   */
/* ------------------------------------------------------------------ */

static void write_trustii_ids(const char *path, const RawPatient *patients, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t np = (uint32_t)n;
    fwrite(&np, 4, 1, f);
    for (int i = 0; i < n; i++) {
        int32_t tid = patients[i].trustii_id;
        fwrite(&tid, sizeof(int32_t), 1, f);
    }
    fclose(f);
    printf("  %s : %d trustii_ids\n", path, n);
}

/* ------------------------------------------------------------------ */
/* Shuffle Fisher-Yates                                                */
/* ------------------------------------------------------------------ */

static void shuffle_indices(int *idx, int n, unsigned seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* Affichage des statistiques dataset                                  */
/* ------------------------------------------------------------------ */

static void print_stats(const char *label, const RawPatient *patients, int n) {
    int n_hep = 0, n_dth = 0;
    for (int i = 0; i < n; i++) {
        if (patients[i].event_hep) n_hep++;
        if (patients[i].event_dth) n_dth++;
    }
    printf("  %-6s : %4d patients | hep: %d (%.1f%%) | dth: %d (%.1f%%)\n",
           label, n,
           n_hep, 100.0f * n_hep / n,
           n_dth, 100.0f * n_dth / n);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --train <csv> --test <csv> --out <dir> [--val_pct N] [--seed S]\n"
        "  --val_pct N   Pourcentage validation (défaut: 20)\n"
        "  --seed    S   Seed aléatoire (défaut: 42)\n",
        prog);
    exit(1);
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *test_path  = NULL;
    const char *out_dir    = NULL;
    int val_pct = VAL_RATIO_PCT;
    unsigned seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--train"))   train_path = argv[++i];
        else if (!strcmp(argv[i], "--test"))    test_path  = argv[++i];
        else if (!strcmp(argv[i], "--out"))     out_dir    = argv[++i];
        else if (!strcmp(argv[i], "--val_pct")) val_pct    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))    seed       = (unsigned)atoi(argv[++i]);
        else { fprintf(stderr, "Option inconnue : %s\n", argv[i]); usage(argv[0]); }
    }

    if (!train_path || !test_path || !out_dir) usage(argv[0]);

    /* Allocation */
    RawPatient *all_train = calloc(MAX_PATIENTS, sizeof(RawPatient));
    RawPatient *all_test  = calloc(MAX_PATIENTS, sizeof(RawPatient));
    if (!all_train || !all_test) { fprintf(stderr, "Mémoire insuffisante\n"); return 1; }

    /* Chargement */
    printf("\nChargement des données...\n");
    int n_train = load_csv(train_path, all_train, MAX_PATIENTS);
    int n_test  = load_csv(test_path,  all_test,  MAX_PATIENTS);
    if (n_train <= 0 || n_test <= 0) return 1;

    /* Stats événements */
    printf("\nStatistiques :\n");
    print_stats("train", all_train, n_train);
    print_stats("test",  all_test,  n_test);

    /* Normalisation sur les données train uniquement */
    NormStats stats;
    compute_norm_stats(all_train, n_train, &stats);
    apply_norm(all_train, n_train, &stats);
    apply_norm(all_test,  n_test,  &stats);

    /* Split train / val */
    int *idx = malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) idx[i] = i;
    shuffle_indices(idx, n_train, seed);

    int n_val = (n_train * val_pct) / 100;
    int n_trn = n_train - n_val;

    RawPatient *trn_buf = malloc(n_trn * sizeof(RawPatient));
    RawPatient *val_buf = malloc(n_val * sizeof(RawPatient));

    for (int i = 0; i < n_val; i++)
        val_buf[i] = all_train[idx[i]];
    for (int i = 0; i < n_trn; i++)
        trn_buf[i] = all_train[idx[n_val + i]];

    printf("\nSplit : %d train / %d val\n", n_trn, n_val);
    print_stats("trn", trn_buf, n_trn);
    print_stats("val", val_buf, n_val);

    /* Création du dossier de sortie (best-effort) */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
        (void)system(cmd);
    }

    /* Écriture */
    printf("\nÉcriture binaires...\n");
    char path[512];

    snprintf(path, sizeof(path), "%s/train.bin", out_dir);
    write_bin(path, trn_buf, n_trn);

    snprintf(path, sizeof(path), "%s/val.bin", out_dir);
    write_bin(path, val_buf, n_val);

    snprintf(path, sizeof(path), "%s/test.bin", out_dir);
    write_bin(path, all_test, n_test);

    snprintf(path, sizeof(path), "%s/norm.bin", out_dir);
    write_norm(path, &stats);

    snprintf(path, sizeof(path), "%s/test_ids.bin", out_dir);
    write_trustii_ids(path, all_test, n_test);

    free(all_train); free(all_test);
    free(trn_buf);   free(val_buf);
    free(idx);

    printf("\nDone. Données dans : %s/\n", out_dir);
    return 0;
}
