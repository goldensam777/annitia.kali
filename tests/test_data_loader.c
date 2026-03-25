#include "annitia.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int n_passed = 0;
static int n_failed = 0;

static void check_int(const char *name, int got, int expected) {
    if (got == expected) {
        printf("  PASS  %s = %d\n", name, got);
        n_passed++;
    } else {
        printf("  FAIL  %s: got=%d, expected=%d\n", name, got, expected);
        n_failed++;
    }
}

static void check_float(const char *name, float got, float expected, float tol) {
    float err = fabsf(got - expected);
    if (err <= tol) {
        printf("  PASS  %s = %.4f\n", name, got);
        n_passed++;
    } else {
        printf("  FAIL  %s: got=%.4f, expected=%.4f, err=%.4f\n",
               name, got, expected, err);
        n_failed++;
    }
}

/* ------------------------------------------------------------------ */
/* Écrit un fichier binaire de test avec 2 patients synthétiques       */
/* ------------------------------------------------------------------ */
static void write_test_bin(const char *path) {
    FILE *f = fopen(path, "wb");
    assert(f);

    uint32_t magic = 0x4D41534C;
    uint32_t n = 2, T = 4, F = 3, version = 1, pad = 0;
    fwrite(&magic,   4, 1, f);
    fwrite(&n,       4, 1, f);
    fwrite(&T,       4, 1, f);
    fwrite(&F,       4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&pad,     4, 1, f);

    /* Patient 0 */
    float feat0[12] = {1,2,3, 4,5,6, 0,0,0, 0,0,0};
    float mask0[12] = {1,1,1, 1,1,1, 0,0,0, 0,0,0};
    float tgap0[4]  = {0, 30, 60, 0};
    int32_t nv0 = 2;
    float th0 = 365.0f; uint8_t eh0 = 1;
    float td0 = 500.0f; uint8_t ed0 = 0;

    fwrite(feat0, sizeof(float), 12, f);
    fwrite(mask0, sizeof(float), 12, f);
    fwrite(tgap0, sizeof(float),  4, f);
    fwrite(&nv0,  sizeof(int32_t), 1, f);
    fwrite(&th0,  sizeof(float),   1, f);
    uint8_t tmp[2] = {eh0, 0}; fwrite(tmp, 1, 2, f);
    fwrite(&td0,  sizeof(float),   1, f);
    uint8_t tmp2[4] = {ed0, 0, 0, 0}; fwrite(tmp2, 1, 4, f);

    /* Patient 1 */
    float feat1[12] = {7,8,9, 0,0,0, 0,0,0, 0,0,0};
    float mask1[12] = {1,1,1, 0,0,0, 0,0,0, 0,0,0};
    float tgap1[4]  = {0, 0, 0, 0};
    int32_t nv1 = 1;
    float th1 = 200.0f; uint8_t eh1 = 0;
    float td1 = 200.0f; uint8_t ed1 = 1;

    fwrite(feat1, sizeof(float), 12, f);
    fwrite(mask1, sizeof(float), 12, f);
    fwrite(tgap1, sizeof(float),  4, f);
    fwrite(&nv1,  sizeof(int32_t), 1, f);
    fwrite(&th1,  sizeof(float),   1, f);
    uint8_t tmp3[2] = {eh1, 0}; fwrite(tmp3, 1, 2, f);
    fwrite(&td1,  sizeof(float),   1, f);
    uint8_t tmp4[4] = {ed1, 0, 0, 0}; fwrite(tmp4, 1, 4, f);

    fclose(f);
}

/* ------------------------------------------------------------------ */
/* Test : round-trip écriture → lecture → vérification                 */
/* ------------------------------------------------------------------ */
static void test_roundtrip(void) {
    printf("\n[test_data_loader_roundtrip]\n");

    const char *path = "/tmp/test_masld.bin";
    write_test_bin(path);

    MasldDataset *ds = masld_load(path);
    assert(ds);

    check_int("n_patients", (int)masld_n_patients(ds), 2);

    SurvivalBatch *b = masld_batch_alloc(2, 4, 3);
    masld_get_batch(ds, b, 0, 2);

    check_int("batch_size",    (int)b->batch_size, 2);
    check_int("seq_len",       (int)b->seq_len,    4);
    check_int("n_features",    (int)b->n_features, 3);

    /* Vérifier patient 0, timestep 0, feature 0 */
    check_float("p0_t0_f0",  b->features[0],  1.0f, 1e-6f);
    check_float("p0_t0_f1",  b->features[1],  2.0f, 1e-6f);
    check_float("p0_t0_f2",  b->features[2],  3.0f, 1e-6f);
    check_float("p0_t1_f0",  b->features[3],  4.0f, 1e-6f);

    /* Vérifier masque patient 0 */
    check_float("mask_p0_t0_f0",  b->mask[0], 1.0f, 1e-6f);
    check_float("mask_p0_t2_f0",  b->mask[6], 0.0f, 1e-6f);

    /* Vérifier targets patient 0 */
    check_float("time_hepatic_p0", b->time_hepatic[0],    365.0f, 1e-6f);
    check_int("event_hepatic_p0",  (int)b->event_hepatic[0], 1);
    check_int("event_death_p0",    (int)b->event_death[0],   0);

    /* Vérifier patient 1 */
    check_float("p1_t0_f0", b->features[4*3 + 0], 7.0f, 1e-6f);
    check_int("n_visits_p1", b->n_visits[1], 1);
    check_int("event_death_p1", (int)b->event_death[1], 1);

    masld_batch_free(b);
    masld_free(ds);
    printf("  (cleaned up)\n");
}

/* ------------------------------------------------------------------ */
/* Test : mask_last_pool                                                */
/* ------------------------------------------------------------------ */
static void test_mask_pool(void) {
    printf("\n[test_mask_pool]\n");

    float hidden[3*4] = {
        1,2,3,4,   /* t=0 */
        5,6,7,8,   /* t=1 */
        9,10,11,12 /* t=2 */
    };
    float tmask[3] = {1, 1, 0};  /* t=2 invalide */
    float out[4]   = {0};

    mask_last_pool(hidden, tmask, out, 3, 4);
    check_float("last_pool_d0", out[0], 5.0f, 1e-6f);
    check_float("last_pool_d1", out[1], 6.0f, 1e-6f);
    check_float("last_pool_d2", out[2], 7.0f, 1e-6f);
    check_float("last_pool_d3", out[3], 8.0f, 1e-6f);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== test_data_loader ===\n");
    test_roundtrip();
    test_mask_pool();
    printf("\n=== Results: %d passed, %d failed ===\n", n_passed, n_failed);
    return n_failed > 0 ? 1 : 0;
}
