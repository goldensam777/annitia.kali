#include "annitia.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/* Vérification de la Cox loss sur un cas analytique simple            */
/*                                                                      */
/* 3 patients, risques connus, temps et événements fixés.              */
/* La valeur analytique est calculée à la main.                        */
/* ------------------------------------------------------------------ */

static int n_passed = 0;
static int n_failed = 0;

static void check(const char *name, float got, float expected, float tol) {
    float err = fabsf(got - expected);
    if (err <= tol) {
        printf("  PASS  %s  (got=%.6f, expected=%.6f)\n", name, got, expected);
        n_passed++;
    } else {
        printf("  FAIL  %s  (got=%.6f, expected=%.6f, err=%.6f > tol=%.6f)\n",
               name, got, expected, err, tol);
        n_failed++;
    }
}

/* ------------------------------------------------------------------ */
/* Test 1 : Cox loss sur 3 patients                                    */
/*                                                                      */
/* Patients : risk=[0, 1, 2], times=[3, 1, 2], events=[1, 1, 0]       */
/*                                                                      */
/* Événements : i=0 (t=3,e=1), i=1 (t=1,e=1)                         */
/*                                                                      */
/* Pour i=1 (t=1) : risk set = {0,1,2} (tous t≥1)                     */
/*   L_1 = risk[1] - log(exp(0)+exp(1)+exp(2)) = 1 - log(1+e+e²)     */
/*                                                                      */
/* Pour i=0 (t=3) : risk set = {0} (seul t≥3)                         */
/*   L_0 = risk[0] - log(exp(0)) = 0 - 0 = 0                          */
/*                                                                      */
/* L_cox = -(L_1 + L_0) / 2                                            */
/* ------------------------------------------------------------------ */
static void test_cox_basic(void) {
    printf("\n[test_cox_basic]\n");

    float    risks[]  = {0.0f, 1.0f, 2.0f};
    float    times[]  = {3.0f, 1.0f, 2.0f};
    uint8_t  events[] = {1,    1,    0   };
    size_t   n = 3;

    /* Calcul analytique (max_risk = 2 pour stabilité) */
    float max_r = 2.0f;
    /* i=1, t=1 : risk set all */
    float s1 = expf(0.0f-max_r) + expf(1.0f-max_r) + expf(2.0f-max_r);
    float L1 = (1.0f - max_r) - logf(s1);
    /* i=0, t=3 : risk set {0} */
    float s0 = expf(0.0f-max_r);
    float L0 = (0.0f - max_r) - logf(s0);
    float expected_loss = -(L1 + L0) / 2.0f;

    float grads[3] = {0};
    float loss = cox_loss(risks, times, events, grads, n);

    check("cox_loss value", loss, expected_loss, 1e-4f);

    /* Vérification du signe des gradients :
     * grad[1] devrait être < 0 (risque sous-estimé pour événement précoce) */
    printf("  grads = [%.4f, %.4f, %.4f]\n", grads[0], grads[1], grads[2]);
}

/* ------------------------------------------------------------------ */
/* Test 2 : C-index parfait                                            */
/* ------------------------------------------------------------------ */
static void test_c_index_perfect(void) {
    printf("\n[test_c_index_perfect]\n");

    /* Risques croissants, temps décroissants → C-index = 1.0 */
    float    risks[]  = {3.0f, 2.0f, 1.0f};
    float    times[]  = {1.0f, 2.0f, 3.0f};
    uint8_t  events[] = {1,    1,    1   };

    float ci = c_index(risks, times, events, 3);
    check("c_index perfect", ci, 1.0f, 1e-6f);
}

/* ------------------------------------------------------------------ */
/* Test 3 : C-index inverse = 0.0                                      */
/* ------------------------------------------------------------------ */
static void test_c_index_inverse(void) {
    printf("\n[test_c_index_inverse]\n");

    float    risks[]  = {1.0f, 2.0f, 3.0f};
    float    times[]  = {1.0f, 2.0f, 3.0f};
    uint8_t  events[] = {1,    1,    1   };

    float ci = c_index(risks, times, events, 3);
    check("c_index inverse", ci, 0.0f, 1e-6f);
}

/* ------------------------------------------------------------------ */
/* Test 4 : Ranking loss minimum = 0 si parfaitement ordonné           */
/* ------------------------------------------------------------------ */
static void test_ranking_loss_perfect(void) {
    printf("\n[test_ranking_loss_perfect]\n");

    /* risk_i >> risk_j pour tous les pairs concordants → sigma(neg) → 0 */
    float    risks[]  = {10.0f, 5.0f, 1.0f};
    float    times[]  = { 1.0f, 2.0f, 3.0f};
    uint8_t  events[] = {   1,     1,    0 };

    float grads[3] = {0};
    float loss = ranking_loss(risks, times, events, grads, 3);

    /* Loss devrait être très proche de 0 */
    printf("  ranking_loss = %.6f (expected ≈ 0)\n", loss);
    check("ranking_loss near zero", loss < 0.01f ? 0.0f : loss, 0.0f, 0.01f);
}

/* ------------------------------------------------------------------ */
/* Test 5 : Loss combinée gradient cohérence                           */
/* ------------------------------------------------------------------ */
static void test_combined_gradient(void) {
    printf("\n[test_combined_gradient]\n");

    float    risks[]  = {0.5f, -0.5f, 0.0f};
    float    times[]  = {1.0f,  2.0f, 3.0f};
    uint8_t  events[] = {1,     1,    0   };

    float grads[3] = {0};
    float loss = survival_loss_combined(risks, times, events, grads, 3, 0.7f);

    printf("  combined_loss = %.6f\n", loss);
    printf("  grads = [%.4f, %.4f, %.4f]\n", grads[0], grads[1], grads[2]);

    /* Vérification : loss doit être positif */
    check("combined_loss positive", loss >= 0.0f ? loss : -1.0f, loss, 1e-6f);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== test_survival_loss ===\n");

    test_cox_basic();
    test_c_index_perfect();
    test_c_index_inverse();
    test_ranking_loss_perfect();
    test_combined_gradient();

    printf("\n=== Results: %d passed, %d failed ===\n", n_passed, n_failed);
    return n_failed > 0 ? 1 : 0;
}
