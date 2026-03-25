#include "annitia.h"
#include <string.h>

/* Applique le masque sur les features : met à 0 les valeurs manquantes.
 * features[T, F] et mask[T, F] sont row-major.
 * Opération in-place. */
void mask_apply(float *features, const float *mask, size_t T, size_t F) {
    size_t n = T * F;
    for (size_t i = 0; i < n; i++)
        features[i] *= mask[i];
}

/* Retourne l'index du dernier timestep valide pour un patient.
 * Un timestep est valide si au moins une feature est non-masquée.
 * mask[T, F] row-major. Retourne 0 si aucun timestep valide. */
size_t mask_last_valid(const float *mask, size_t T, size_t F) {
    for (size_t t = T; t > 0; t--) {
        const float *row = mask + (t - 1) * F;
        for (size_t f = 0; f < F; f++) {
            if (row[f] > 0.5f)
                return t - 1;
        }
    }
    return 0;
}

/* Construit un masque temporel [T] : 1 si le timestep t a au moins
 * une feature valide, 0 sinon. out[T] alloué par l'appelant. */
void mask_timestep(const float *mask, float *out, size_t T, size_t F) {
    for (size_t t = 0; t < T; t++) {
        const float *row = mask + t * F;
        float any = 0.0f;
        for (size_t f = 0; f < F; f++)
            any += row[f];
        out[t] = (any > 0.5f) ? 1.0f : 0.0f;
    }
}

/* Mean pooling sur les timesteps valides.
 * hidden[T, D] → out[D]. tmask[T] = masque temporel (0/1). */
void mask_mean_pool(const float *hidden, const float *tmask,
                    float *out, size_t T, size_t D) {
    memset(out, 0, D * sizeof(float));
    float count = 0.0f;
    for (size_t t = 0; t < T; t++) {
        if (tmask[t] > 0.5f) {
            const float *h = hidden + t * D;
            for (size_t d = 0; d < D; d++)
                out[d] += h[d];
            count += 1.0f;
        }
    }
    if (count > 0.0f) {
        for (size_t d = 0; d < D; d++)
            out[d] /= count;
    }
}

/* Last-valid-timestep pooling.
 * hidden[T, D] → out[D]. tmask[T] = masque temporel. */
void mask_last_pool(const float *hidden, const float *tmask,
                    float *out, size_t T, size_t D) {
    size_t last = 0;
    for (size_t t = 0; t < T; t++)
        if (tmask[t] > 0.5f) last = t;
    const float *h = hidden + last * D;
    memcpy(out, h, D * sizeof(float));
}
