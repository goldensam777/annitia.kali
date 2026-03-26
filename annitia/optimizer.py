"""
optimizer.py — Surface de score 4D pour ANNITIA.

Modélise Score(θ) = f(w_ssm, epochs, mimo_rank, use_conv2d) via GP.

θ = (w_ssm, epochs, mimo_rank, use_conv2d)
  w_ssm      in [0, 1]    — SSM weight in rank-average ensemble
  epochs     in [1, 200]  — SSM training epochs
  mimo_rank  in {1, 4}    — MambaBlock MIMO rank
  use_conv2d in {0, 1}    — Conv2D preprocessing layer (new dimension)

Series of model functions:
  f0 : SSM only        (use_conv2d=0, various epochs/mimo)
  f1 : Conv2D + SSM    (use_conv2d=1, various epochs/mimo)
  f* : optimal mix     — argmax over all dimensions

Resolution objective: {theta : Score(theta) >= 0.95}
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# ---------------------------------------------------------------------------
# Observations (w_ssm, epochs, mimo_rank, use_conv2d, score)
# ---------------------------------------------------------------------------
OBSERVATIONS = [
    # --- f0 : SSM without Conv2D ---
    (0.00, 100, 4, 0, 0.8823),   # XGBoost alone (w_ssm=0)
    (1.00,  30, 1, 0, 0.7819),   # SSM mimo1 30ep
    (1.00, 100, 4, 0, 0.8464),   # SSM mimo4 100ep — EXP-12-bis (b_B fix)
    (1.00, 200, 4, 0, 0.8720),   # SSM mimo4 200ep seed=123 — EXP-14b (best seed)
    # Note: seed 42 at 200ep = 0.8422, seed 123 = 0.8720 (seed variance high)
    # 2-seed average (42+123): OOF=0.8849 > XGBoost

    # --- f1 : SSM with Conv2D (K=3) ---
    (1.00,  60, 1, 1, 0.8042),   # Conv2D K=3, mimo1, 60ep — OOF 5-fold
    (1.00, 100, 4, 1, 0.8226),   # Conv2D K=3, mimo4, 100ep — EXP-13
]

SUBMISSIONS = [
    # Fill after official Trustii scores
]

# Bounds for each dimension
BOUNDS = np.array([
    [0.0,   1.0],   # w_ssm
    [1.0, 200.0],   # epochs
    [1.0,   4.0],   # mimo_rank
    [0.0,   1.0],   # use_conv2d (continuous relaxation for GP)
])


# ---------------------------------------------------------------------------
# Series of functions to evaluate next (experimental plan)
# ---------------------------------------------------------------------------
# Each entry is a candidate configuration (w_ssm, epochs, mimo_rank, use_conv2d)
# ordered by expected information gain.
EXPERIMENT_SERIES = [
    # f1: Conv2D with more epochs
    (1.00, 100, 4, 1),   # Conv2D mimo4 100ep  — does it match SSM 100ep?
    (1.00, 150, 4, 1),   # Conv2D mimo4 150ep  — extrapolation zone
    # f1 ensemble
    (0.20,  60, 4, 1),   # 20% Conv2D SSM in ensemble
    (0.10,  60, 4, 1),   # 10% Conv2D SSM in ensemble
    # f0: SSM without Conv2D, higher epochs
    (1.00, 150, 4, 0),   # SSM 150ep baseline
    # f1: Conv2D with larger kernel
    (1.00,  60, 4, 1),   # Conv2D K=3 but mimo4 (current was mimo1)
]


# ---------------------------------------------------------------------------
# GP core
# ---------------------------------------------------------------------------

def _build_X_y(extra_obs=None):
    obs = OBSERVATIONS + (extra_obs or []) + SUBMISSIONS
    X = np.array([[w, ep, mr, c2d] for w, ep, mr, c2d, _ in obs], dtype=np.float64)
    y = np.array([s for _, _, _, _, s in obs], dtype=np.float64)
    return X, y


def _normalize(X):
    X = np.atleast_2d(X)
    return (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def _denormalize(X_norm):
    X_norm = np.atleast_2d(X_norm)
    return X_norm * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]


def fit_gp(extra_obs=None):
    X, y = _build_X_y(extra_obs)
    X_norm = _normalize(X)
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3, 0.5, 0.5], nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        n_restarts_optimizer=15,
        normalize_y=True,
    )
    gp.fit(X_norm, y)
    return gp, X_norm, y


def predict_score(gp, w_ssm, epochs, mimo_rank, use_conv2d=0):
    X = np.array([[w_ssm, epochs, mimo_rank, use_conv2d]], dtype=np.float64)
    mu, sigma = gp.predict(_normalize(X), return_std=True)
    return float(mu[0]), float(sigma[0])


# ---------------------------------------------------------------------------
# Optimum: argmax Score(theta)
# ---------------------------------------------------------------------------

def find_optimum(gp, n_restarts=30):
    """
    Find theta* = argmax Score(theta) over the full 4D space.
    Uses differential evolution (global) + L-BFGS-B (local refinement).
    """
    bounds_norm = [(0.0, 1.0)] * 4

    def neg_score(x_norm):
        mu = gp.predict(x_norm.reshape(1, -1), return_std=False)
        return -float(mu[0])

    # Global search via differential evolution
    res_de = differential_evolution(neg_score, bounds_norm, seed=42,
                                    maxiter=500, tol=1e-6, popsize=15)

    # Local refinement from best + random restarts
    best_val = res_de.fun
    best_x   = res_de.x

    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        x0 = rng.uniform(0, 1, size=4)
        res = minimize(neg_score, x0, method='L-BFGS-B', bounds=bounds_norm)
        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x

    theta_raw = _denormalize(best_x)[0]
    return {
        "w_ssm":      float(theta_raw[0]),
        "epochs":     float(theta_raw[1]),
        "mimo_rank":  int(round(theta_raw[2])),
        "use_conv2d": int(round(theta_raw[3])),
        "score_hat":  float(-best_val),
    }


# ---------------------------------------------------------------------------
# Solve {theta : Score(theta) >= target}
# ---------------------------------------------------------------------------

def solve_threshold(gp, target=0.95, n_samples=200_000):
    """
    Test convergence: estimate whether {theta : Score(theta) >= target} is empty.

    Strategy:
      1. Dense random sampling (Monte Carlo) to find candidate regions
      2. Local optimization from best candidates
      3. Return best found score and whether threshold is reachable

    Returns dict with:
      - reachable  : bool — True if Score(theta*) >= target
      - best_score : float — best score found (GP mean)
      - best_theta : dict — best configuration
      - gap        : float — how far from target (target - best_score)
      - n_above    : int  — number of MC samples above target (mean)
    """
    rng = np.random.default_rng(0)
    X_mc = rng.uniform(0, 1, size=(n_samples, 4))
    mu_mc = gp.predict(X_mc, return_std=False)

    n_above = int((mu_mc >= target).sum())
    best_mc_idx = np.argmax(mu_mc)
    best_mc_score = float(mu_mc[best_mc_idx])

    # Refine from top-k candidates
    top_k = np.argsort(mu_mc)[-50:]
    best_score = best_mc_score
    best_x_norm = X_mc[best_mc_idx]

    for idx in top_k:
        res = minimize(
            lambda x: -float(gp.predict(x.reshape(1, -1), return_std=False)[0]),
            X_mc[idx], method='L-BFGS-B',
            bounds=[(0, 1)] * 4,
        )
        if -res.fun > best_score:
            best_score = -res.fun
            best_x_norm = res.x

    theta_raw = _denormalize(best_x_norm)[0]
    best_theta = {
        "w_ssm":      float(theta_raw[0]),
        "epochs":     float(theta_raw[1]),
        "mimo_rank":  int(round(theta_raw[2])),
        "use_conv2d": int(round(theta_raw[3])),
        "score_hat":  best_score,
    }

    return {
        "reachable":   best_score >= target,
        "best_score":  best_score,
        "best_theta":  best_theta,
        "gap":         target - best_score,
        "n_above":     n_above,
        "n_samples":   n_samples,
    }


# ---------------------------------------------------------------------------
# 2D surface (slice at fixed mimo_rank and use_conv2d)
# ---------------------------------------------------------------------------

def gradient_map(gp, mimo_rank=4, use_conv2d=0, n_grid=50):
    w_vals  = np.linspace(0, 1,   n_grid)
    ep_vals = np.linspace(30, 200, n_grid)
    WW, EE  = np.meshgrid(w_vals, ep_vals)
    X_grid = np.column_stack([
        WW.ravel(), EE.ravel(),
        np.full(n_grid * n_grid, float(mimo_rank)),
        np.full(n_grid * n_grid, float(use_conv2d)),
    ])
    scores = gp.predict(_normalize(X_grid), return_std=False)
    SS = scores.reshape(n_grid, n_grid)
    grad_w  = np.gradient(SS, w_vals,  axis=1)
    grad_ep = np.gradient(SS, ep_vals, axis=0)
    return WW, EE, SS, grad_w, grad_ep


def plot_surface(gp, extra_obs=None, out_path="score_surface.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Score Surface — 4D GP (w_ssm, epochs, mimo_rank, use_conv2d)", fontsize=13)

    X_obs, y_obs = _build_X_y(extra_obs)
    theta_star = find_optimum(gp)

    for row, conv2d_val in enumerate([0, 1]):
        label = "f0: No Conv2D" if conv2d_val == 0 else "f1: Conv2D K=3"

        WW, EE, SS, GW, GEP = gradient_map(gp, mimo_rank=4, use_conv2d=conv2d_val)

        # Heatmap
        ax = axes[row][0]
        cf = ax.contourf(WW, EE, SS, levels=30, cmap='RdYlGn', vmin=0.75, vmax=0.95)
        plt.colorbar(cf, ax=ax, label='Score')
        step = 5
        ax.quiver(WW[::step, ::step], EE[::step, ::step],
                  GW[::step, ::step], GEP[::step, ::step],
                  alpha=0.5, color='white', scale=3)

        # Observations
        mask = X_obs[:, 3] == conv2d_val
        if mask.any():
            ax.scatter(X_obs[mask, 0], X_obs[mask, 1],
                       c=y_obs[mask], cmap='RdYlGn', vmin=0.75, vmax=0.95,
                       edgecolors='black', s=100, zorder=5)

        if int(round(theta_star['use_conv2d'])) == conv2d_val:
            ax.scatter([theta_star['w_ssm']], [theta_star['epochs']],
                       marker='*', s=400, c='gold', edgecolors='black',
                       zorder=10, label=f"θ* {theta_star['score_hat']:.4f}")
            ax.legend(fontsize=8)

        ax.set_xlabel('w_ssm'); ax.set_ylabel('epochs')
        ax.set_title(f'{label} — heatmap mimo=4')

        # Profile Score(w_ssm) at optimal epochs
        ax2 = axes[row][1]
        best_ep_idx = int(np.argmax(SS.max(axis=1)))
        best_ep = np.linspace(30, 200, 50)[best_ep_idx]
        w_line = np.linspace(0, 1, 100)
        X_line = np.column_stack([w_line, np.full(100, best_ep),
                                   np.full(100, 4.0), np.full(100, float(conv2d_val))])
        s_line, sig_line = gp.predict(_normalize(X_line), return_std=True)

        ax2.plot(w_line, s_line, 'b-', lw=2, label=f'Score (ep={best_ep:.0f})')
        ax2.fill_between(w_line, s_line - 2*sig_line, s_line + 2*sig_line,
                         alpha=0.2, color='blue', label='±2σ')
        ax2.axhline(0.95, color='red', ls='--', lw=1.5, label='target 0.95')
        ax2.axhline(0.8823, color='orange', ls=':', lw=1, label='XGBoost (0.8823)')
        if mask.any():
            ax2.scatter(X_obs[mask, 0], y_obs[mask], c='red', zorder=5, s=60)
        ax2.set_xlabel('w_ssm'); ax2.set_ylabel('Score')
        ax2.set_title(f'{label} — profile Score(w_ssm)')
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.70, 1.00)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Surface saved: {out_path}")
    return theta_star


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def report(extra_obs=None, target=0.95):
    print("=== ANNITIA Score Surface — 4D GP ===\n")

    gp, X_norm, y = fit_gp(extra_obs)
    print(f"GP fitted on {len(y)} observations")
    print(f"Kernel: {gp.kernel_}")
    print(f"Log-likelihood: {gp.log_marginal_likelihood_value_:.4f}\n")

    print("Predictions on known points:")
    all_obs = OBSERVATIONS + (extra_obs or []) + SUBMISSIONS
    for w, ep, mr, c2d, s_true in all_obs:
        mu, sigma = predict_score(gp, w, ep, mr, c2d)
        tag = "f0" if c2d == 0 else "f1"
        print(f"  {tag} θ=({w:.2f}, {ep:3d}ep, mimo{mr}) "
              f"true={s_true:.4f}  pred={mu:.4f} ±{sigma:.4f}")

    print()
    theta_star = find_optimum(gp)
    print("Optimal theta*:")
    print(f"  w_ssm      = {theta_star['w_ssm']:.3f}")
    print(f"  epochs     = {theta_star['epochs']:.1f}")
    print(f"  mimo_rank  = {theta_star['mimo_rank']}")
    print(f"  use_conv2d = {theta_star['use_conv2d']}")
    print(f"  score_hat  = {theta_star['score_hat']:.4f}")

    print(f"\n--- Solving {{theta : Score(theta) >= {target}}} ---")
    result = solve_threshold(gp, target=target)
    print(f"  Reachable  : {result['reachable']}")
    print(f"  Best score : {result['best_score']:.4f}")
    print(f"  Gap        : {result['gap']:+.4f}")
    print(f"  MC samples above target: {result['n_above']} / {result['n_samples']}")
    if result['reachable']:
        bt = result['best_theta']
        print(f"  Solution found:")
        print(f"    w_ssm={bt['w_ssm']:.3f}, epochs={bt['epochs']:.0f}, "
              f"mimo={bt['mimo_rank']}, conv2d={bt['use_conv2d']}")
    else:
        print(f"  Solution set is EMPTY in current parameter space.")
        print(f"  Closest point: {result['best_theta']}")

    print(f"\n--- Next experiments (information-maximizing series) ---")
    print(f"  {'Config':<45}  Expected Score")
    for w, ep, mr, c2d in EXPERIMENT_SERIES:
        mu, sigma = predict_score(gp, w, ep, mr, c2d)
        tag = "Conv2D" if c2d else "Base  "
        print(f"  {tag} w={w:.2f} ep={ep:3d} mimo={mr} c2d={c2d}  "
              f"mu={mu:.4f} ±{sigma:.4f}")

    plot_surface(gp, extra_obs=extra_obs, out_path="score_surface.png")

    return gp, theta_star, result
