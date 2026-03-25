#include "annitia.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Cox partial likelihood                                               */
/*                                                                      */
/* Pour chaque patient i avec event_i = 1 :                            */
/*   L_i = risk_i - log( sum_{j: time_j >= time_i} exp(risk_j) )       */
/*   L_cox = -mean_i(L_i)                                              */
/*                                                                      */
/* Gradient par rapport à risk_k :                                      */
/*   dL/d_risk_k = -(event_k - sum_{i: event_i=1, time_i<=time_k}      */
/*                    exp(risk_k) / sum_{j: time_j>=time_i} exp(risk_j))*/
/*                                                                      */
/* Implémentation O(n²) — suffisant pour n ≤ 2000 patients.            */
/* ------------------------------------------------------------------ */
float cox_loss(const float *risks, const float *times,
               const uint8_t *events, float *grads, size_t n) {
    if (grads) memset(grads, 0, n * sizeof(float));

    /* Nombre d'événements observés */
    size_t n_events = 0;
    for (size_t i = 0; i < n; i++)
        if (events[i]) n_events++;
    if (n_events == 0) return 0.0f;

    /* Stabilité numérique : soustraire le max des risks */
    float max_risk = risks[0];
    for (size_t i = 1; i < n; i++)
        if (risks[i] > max_risk) max_risk = risks[i];

    float loss = 0.0f;

    for (size_t i = 0; i < n; i++) {
        if (!events[i]) continue;

        /* risk set : j tels que time_j >= time_i */
        float log_sum_exp = 0.0f;
        for (size_t j = 0; j < n; j++)
            if (times[j] >= times[i])
                log_sum_exp += expf(risks[j] - max_risk);

        if (log_sum_exp < 1e-20f) log_sum_exp = 1e-20f;
        loss += (risks[i] - max_risk) - logf(log_sum_exp);
    }

    loss = -loss / (float)n_events;

    /* Gradients */
    if (grads) {
        for (size_t k = 0; k < n; k++) {
            float g = 0.0f;
            /* Contribution de chaque événement i au gradient de risk_k */
            for (size_t i = 0; i < n; i++) {
                if (!events[i]) continue;
                if (times[k] < times[i]) continue;  /* k pas dans le risk set de i */

                float sum_exp = 0.0f;
                for (size_t j = 0; j < n; j++)
                    if (times[j] >= times[i])
                        sum_exp += expf(risks[j] - max_risk);

                if (sum_exp < 1e-20f) sum_exp = 1e-20f;
                g += expf(risks[k] - max_risk) / sum_exp;
            }
            grads[k] = (g - (float)events[k]) / (float)n_events;
        }
    }

    return loss;
}

/* ------------------------------------------------------------------ */
/* Pairwise ranking loss (proxy C-index différentiable)                */
/*                                                                      */
/* Pour chaque paire concordante (i,j) où event_i=1, time_i < time_j : */
/*   l_ij = sigmoid(risk_j - risk_i)   (pénalise si risk_i < risk_j)  */
/*   L_rank = mean_{paires concordantes}(l_ij)                         */
/*                                                                      */
/* Gradient : dl/d_risk_i = sigma(risk_j - risk_i) * (1 - sigma)       */
/*            dl/d_risk_j = -sigma(risk_j - risk_i) * (1 - sigma)      */
/* ------------------------------------------------------------------ */
float ranking_loss(const float *risks, const float *times,
                   const uint8_t *events, float *grads, size_t n) {
    if (grads) memset(grads, 0, n * sizeof(float));

    float loss    = 0.0f;
    float n_pairs = 0.0f;

    for (size_t i = 0; i < n; i++) {
        if (!events[i]) continue;

        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            if (times[j] <= times[i]) continue;  /* paire non concordante */

            /* sigma(risk_j - risk_i) : si risk_i > risk_j, sigma → 0, loss faible */
            float diff  = risks[j] - risks[i];
            float sigma = 1.0f / (1.0f + expf(-diff));
            loss += sigma;
            n_pairs += 1.0f;

            if (grads) {
                float ds = sigma * (1.0f - sigma);
                grads[i] -= ds;  /* augmenter risk_i → réduire loss */
                grads[j] += ds;
            }
        }
    }

    if (n_pairs < 1.0f) return 0.0f;

    loss /= n_pairs;
    if (grads)
        for (size_t k = 0; k < n; k++)
            grads[k] /= n_pairs;

    return loss;
}

/* ------------------------------------------------------------------ */
/* Loss combinée : alpha * cox + (1 - alpha) * ranking                 */
/* ------------------------------------------------------------------ */
float survival_loss_combined(const float *risks, const float *times,
                              const uint8_t *events, float *grads,
                              size_t n, float alpha) {
    float *g_cox  = NULL;
    float *g_rank = NULL;

    if (grads) {
        g_cox  = calloc(n, sizeof(float));
        g_rank = calloc(n, sizeof(float));
    }

    float l_cox  = cox_loss(risks, times, events, g_cox,  n);
    float l_rank = ranking_loss(risks, times, events, g_rank, n);

    if (grads) {
        for (size_t i = 0; i < n; i++)
            grads[i] = alpha * g_cox[i] + (1.0f - alpha) * g_rank[i];
        free(g_cox);
        free(g_rank);
    }

    return alpha * l_cox + (1.0f - alpha) * l_rank;
}

/* ------------------------------------------------------------------ */
/* C-index (évaluation, non différentiable)                            */
/*                                                                      */
/* C = (paires concordantes + 0.5 * paires liées) / paires comparables */
/* ------------------------------------------------------------------ */
float c_index(const float *risks, const float *times,
              const uint8_t *events, size_t n) {
    float concordant = 0.0f;
    float tied       = 0.0f;
    float comparable = 0.0f;

    for (size_t i = 0; i < n; i++) {
        if (!events[i]) continue;
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            if (times[j] <= times[i]) continue;  /* paire non comparable */

            comparable += 1.0f;
            float d = risks[i] - risks[j];
            if (d > 1e-8f)       concordant += 1.0f;
            else if (d > -1e-8f) tied       += 0.5f;
        }
    }

    if (comparable < 1.0f) return 0.5f;
    return (concordant + tied) / comparable;
}
