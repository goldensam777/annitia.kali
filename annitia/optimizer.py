"""
optimizer.py — Surface de score paramétrée pour ANNITIA.

Modélise Score(θ) = f(w_ssm, epochs, mimo_rank) via Gaussian Process
à partir des points expérimentaux OOF.

Objectif : trouver θ* = argmax Score(θ) par gradient ascent sur la surface GP.

θ = (w_ssm, epochs, mimo_rank)
  w_ssm    ∈ [0, 1]    — poids SSM dans l'ensemble rank-average
  epochs   ∈ [1, 200]  — epochs d'entraînement SSM
  mimo_rank∈ {1, 4}    — rang MIMO du MambaBlock
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# ---------------------------------------------------------------------------
# Points expérimentaux (θ → Score OOF / soumission)
# ---------------------------------------------------------------------------
#
# Format : (w_ssm, epochs, mimo_rank, score)
# Source : RESULTS.md
#
OBSERVATIONS = [
    # XGBoost seul (w_ssm=0)
    (0.00, 100, 4, 0.8823),   # EXP-09 : XGBoost multi-seed, OOF

    # SSM seul (w_ssm=1)
    (1.00,  30, 1, 0.7819),   # EXP-08 : SSM mimo1 30ep
    (1.00,  50, 4, 0.8081),   # EXP-10 : SSM mimo4 50ep
    (1.00, 100, 4, 0.8417),   # EXP-12 : SSM mimo4 100ep

    # Ensemble (sweep EXP-10)
    (0.20,  50, 4, 0.8558),   # sweep EXP-10 : 20% SSM mimo4 50ep
]

# Soumissions officielles Trustii (remplir au fur et à mesure)
# Format : (w_ssm, epochs, mimo_rank, score_trustii)
SUBMISSIONS = [
    # (0.00, 100, 4, ???),   # submission_1 — à remplir après retour Trustii
    # (0.30, 100, 4, ???),   # submission_2 — à remplir après retour Trustii
]


def _build_X_y(observations=None):
    """Construit les matrices X (features) et y (scores) depuis les observations."""
    obs = OBSERVATIONS + (observations or []) + SUBMISSIONS
    X = np.array([[w, ep, mr] for w, ep, mr, _ in obs], dtype=np.float64)
    y = np.array([s for _, _, _, s in obs], dtype=np.float64)
    return X, y


def _normalize(X):
    """Normalise θ dans [0,1]³ pour le GP."""
    bounds = np.array([
        [0.0,   1.0],    # w_ssm
        [1.0, 200.0],    # epochs
        [1.0,   4.0],    # mimo_rank
    ])
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def fit_gp(extra_obs=None):
    """
    Fitte un Gaussian Process sur les observations.
    Retourne (gp, X_norm, y).
    """
    X, y = _build_X_y(extra_obs)
    X_norm = _normalize(X)

    kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,          # bruit numérique
        n_restarts_optimizer=10,
        normalize_y=True,
    )
    gp.fit(X_norm, y)
    return gp, X_norm, y


def predict_score(gp, w_ssm, epochs, mimo_rank):
    """Prédit le score et l'incertitude pour un θ donné."""
    X = np.array([[w_ssm, epochs, mimo_rank]], dtype=np.float64)
    X_norm = _normalize(X)
    mu, sigma = gp.predict(X_norm, return_std=True)
    return float(mu[0]), float(sigma[0])


def find_optimum(gp, n_restarts=20, extra_obs=None):
    """
    Trouve θ* = argmax Score(θ) par gradient ascent multi-départ.

    Espace de recherche :
      w_ssm    ∈ [0, 1]
      epochs   ∈ [30, 200]
      mimo_rank fixé à 4 (meilleur connu)
    """
    bounds_norm = [(0.0, 1.0), (0.0, 1.0), (1.0, 1.0)]  # mimo_rank normalisé = 1.0 = rank 4

    def neg_score(x_norm):
        mu = gp.predict(x_norm.reshape(1, -1), return_std=False)
        return -float(mu[0])

    def neg_score_grad(x_norm):
        # Gradient numérique
        eps = 1e-5
        g = np.zeros_like(x_norm)
        for i in range(len(x_norm)):
            xp, xm = x_norm.copy(), x_norm.copy()
            xp[i] += eps; xm[i] -= eps
            g[i] = (neg_score(xp) - neg_score(xm)) / (2 * eps)
        return g

    best_val = np.inf
    best_x   = None

    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        x0 = rng.uniform(0, 1, size=3)
        x0[2] = 1.0   # mimo_rank=4 fixé

        res = minimize(
            neg_score, x0, jac=neg_score_grad,
            method='L-BFGS-B', bounds=bounds_norm,
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x

    # Dénormaliser
    bounds_raw = np.array([[0.0, 1.0], [1.0, 200.0], [1.0, 4.0]])
    theta_star = best_x * (bounds_raw[:, 1] - bounds_raw[:, 0]) + bounds_raw[:, 0]

    return {
        "w_ssm":     float(theta_star[0]),
        "epochs":    float(theta_star[1]),
        "mimo_rank": int(round(theta_star[2])),
        "score_hat": float(-best_val),
    }


def gradient_map(gp, epochs=100, mimo_rank=4, n_grid=50):
    """
    Calcule la carte de score Score(w_ssm, epochs) à mimo_rank fixé.
    Retourne (w_grid, ep_grid, score_grid, grad_w, grad_ep).
    """
    w_vals  = np.linspace(0, 1,   n_grid)
    ep_vals = np.linspace(30, 200, n_grid)
    WW, EE  = np.meshgrid(w_vals, ep_vals)

    X_grid = np.column_stack([
        WW.ravel(),
        EE.ravel(),
        np.full(n_grid * n_grid, float(mimo_rank)),
    ])
    X_norm = _normalize(X_grid)
    scores, _ = gp.predict(X_norm, return_std=True)
    SS = scores.reshape(n_grid, n_grid)

    # Gradient numérique sur la grille
    grad_w  = np.gradient(SS, w_vals,  axis=1)
    grad_ep = np.gradient(SS, ep_vals, axis=0)

    return WW, EE, SS, grad_w, grad_ep


def plot_surface(gp, extra_obs=None, out_path="score_surface.png"):
    """
    Trace :
      - Surface Score(w_ssm, epochs) avec mimo_rank=4
      - Champ de gradient (flèches)
      - Points expérimentaux observés
      - θ* marqué
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    WW, EE, SS, GW, GEP = gradient_map(gp, mimo_rank=4)
    theta_star = find_optimum(gp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Heatmap + gradient ---
    ax = axes[0]
    cf = ax.contourf(WW, EE, SS, levels=30, cmap='RdYlGn')
    plt.colorbar(cf, ax=ax, label='Score estimé')

    # Flèches gradient (sous-échantillonnées)
    step = 5
    ax.quiver(
        WW[::step, ::step], EE[::step, ::step],
        GW[::step, ::step], GEP[::step, ::step],
        alpha=0.6, color='white', scale=3,
    )

    # Points expérimentaux
    X_obs, y_obs = _build_X_y(extra_obs)
    mask4 = X_obs[:, 2] == 4
    mask1 = X_obs[:, 2] == 1
    ax.scatter(X_obs[mask4, 0], X_obs[mask4, 1], c=y_obs[mask4],
               cmap='RdYlGn', vmin=SS.min(), vmax=SS.max(),
               edgecolors='black', s=80, zorder=5, label='mimo=4')
    ax.scatter(X_obs[mask1, 0], X_obs[mask1, 1], c=y_obs[mask1],
               cmap='RdYlGn', vmin=SS.min(), vmax=SS.max(),
               edgecolors='blue', s=80, marker='s', zorder=5, label='mimo=1')

    # θ*
    ax.scatter([theta_star['w_ssm']], [theta_star['epochs']],
               marker='*', s=300, c='gold', edgecolors='black',
               zorder=10, label=f"θ* (score≈{theta_star['score_hat']:.4f})")

    ax.set_xlabel('w_ssm (poids SSM dans l\'ensemble)')
    ax.set_ylabel('epochs SSM')
    ax.set_title('Surface Score(w_ssm, epochs) — mimo_rank=4')
    ax.legend(fontsize=8)

    # --- Coupe w_ssm fixé aux epochs optimales ---
    ax2 = axes[1]
    best_ep_idx = int(np.argmax(SS.max(axis=1)))
    best_ep = np.linspace(30, 200, 50)[best_ep_idx]
    w_line = np.linspace(0, 1, 100)
    X_line = np.column_stack([w_line, np.full(100, best_ep), np.full(100, 4.0)])
    score_line, sigma_line = gp.predict(_normalize(X_line), return_std=True)

    ax2.plot(w_line, score_line, 'b-', lw=2, label=f'Score (epochs≈{best_ep:.0f})')
    ax2.fill_between(w_line,
                     score_line - 2 * sigma_line,
                     score_line + 2 * sigma_line,
                     alpha=0.2, color='blue', label='±2σ (incertitude GP)')
    ax2.axvline(theta_star['w_ssm'], color='gold', ls='--',
                label=f"w* = {theta_star['w_ssm']:.3f}")
    ax2.scatter(X_obs[mask4, 0], y_obs[mask4],
                c='red', zorder=5, s=60, label='obs (mimo=4)')
    ax2.set_xlabel('w_ssm')
    ax2.set_ylabel('Score estimé')
    ax2.set_title(f'Coupe Score(w_ssm) à epochs={best_ep:.0f}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Surface sauvegardée : {out_path}")

    return theta_star


def report(extra_obs=None):
    """
    Rapport complet : fit GP → gradient → θ* → plot.
    """
    print("=== Optimiseur de surface de score ANNITIA ===\n")

    gp, X_norm, y = fit_gp(extra_obs)
    print(f"GP fitté sur {len(y)} observations")
    print(f"Kernel appris : {gp.kernel_}")
    print(f"Log-likelihood : {gp.log_marginal_likelihood_value_:.4f}\n")

    print("Prédictions sur les points connus :")
    X_raw, _ = _build_X_y(extra_obs)
    for i, (w, ep, mr, s_true) in enumerate(OBSERVATIONS + (extra_obs or []) + SUBMISSIONS):
        mu, sigma = predict_score(gp, w, ep, mr)
        print(f"  θ=({w:.2f}, {ep:3d}ep, mimo{mr}) → "
              f"vrai={s_true:.4f}  prédit={mu:.4f} ±{sigma:.4f}")

    print()
    theta_star = find_optimum(gp)
    print(f"θ* optimal trouvé :")
    print(f"  w_ssm     = {theta_star['w_ssm']:.3f}")
    print(f"  epochs    = {theta_star['epochs']:.1f}")
    print(f"  mimo_rank = {theta_star['mimo_rank']}")
    print(f"  score hat = {theta_star['score_hat']:.4f}")

    print()
    print("Gradient de Score vs w_ssm à epochs=100, mimo=4 :")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        eps = 0.01
        s_plus,  _ = predict_score(gp, min(w + eps, 1.0), 100, 4)
        s_minus, _ = predict_score(gp, max(w - eps, 0.0), 100, 4)
        grad = (s_plus - s_minus) / (2 * eps)
        mu, sigma = predict_score(gp, w, 100, 4)
        print(f"  w={w:.1f}  score={mu:.4f} ±{sigma:.4f}  ∂Score/∂w={grad:+.4f}")

    plot_surface(gp, extra_obs=extra_obs, out_path="score_surface.png")

    return gp, theta_star
