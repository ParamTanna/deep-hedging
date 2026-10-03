"""Path generators, and how their parameters are estimated from real returns.

Two generators:

  GBM     dS = sigma S dW                          (constant volatility)
  Heston  dS = sqrt(v) S dW1
          dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

Both are driftless. That is a deliberate choice, explained in README: we work in
forward terms with a zero interest rate, so the Black-Scholes premium, the
simulated paths and the profit-and-loss accounting all sit in the same measure
and the hedge carries no financing cost. Drift is a first-order exposure and is
exactly what delta hedging removes, so it is not what this study is about.

Heston is calibrated by method of moments on a realised-variance proxy rather
than by option-implied fitting. We have no option chain, only the index, so
matching the moments of the variance process is the honest thing to do with
the data we actually have.
"""
from __future__ import annotations
import numpy as np

TRADING_DAYS = 252


# ---------------------------------------------------------------- generators

def gbm_paths(S0, sigma, T, n_steps, n_paths, rng):
    """Exact simulation of driftless GBM. No discretisation error in S."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z
    return np.hstack([np.full((n_paths, 1), float(S0)),
                      S0 * np.exp(np.cumsum(incr, axis=1))])


def heston_paths(S0, v0, kappa, theta, xi, rho, T, n_steps, n_paths, rng):
    """Heston with full-truncation Euler on the variance.

    Full truncation means we use max(v, 0) wherever v appears under a square
    root but let the state itself go slightly negative. It is the scheme with
    the smallest documented bias among the simple Euler variants (Lord et al.
    2010) and it cannot produce NaNs, which matters when this runs inside a
    training loop.
    """
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v = np.full(n_paths, float(v0))
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        S[:, t + 1] = S[:, t] * np.exp(-0.5 * vp * dt + np.sqrt(vp * dt) * z1)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    return S


# ---------------------------------------------------------------- calibration

def realised_variance(r, window=5, periods_per_year=TRADING_DAYS):
    """Annualised realised variance over a rolling window of daily returns.

    A single r_t^2 is an unbiased but extremely noisy estimate of daily
    variance. Averaging a short window cuts that noise without smearing the
    clustering we are trying to measure.
    """
    r = np.asarray(r, float)
    sq = r ** 2
    k = np.ones(window) / window
    rv = np.convolve(sq, k, mode="valid")
    return rv * periods_per_year


def calibrate_gbm(r, periods_per_year=TRADING_DAYS):
    """One parameter, one moment: sigma from the sample standard deviation."""
    r = np.asarray(r, float)
    return {"sigma": float(r.std(ddof=1) * np.sqrt(periods_per_year))}


def _acf1(x, lag=1):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-lag] * x[lag:]).sum() / (x ** 2).sum())


def return_moments(r, periods_per_year=TRADING_DAYS):
    """The four stylised facts we ask a simulator to reproduce.

    Chosen so that each one is sensitive to a different Heston parameter:
      annualised vol      -> theta   (pinned exactly, not fitted)
      excess kurtosis     -> xi      (vol-of-vol makes the tails fat)
      skewness            -> rho     (the leverage effect tilts the tails)
      ACF of r^2 at lag 1 -> kappa   (slow mean reversion = long clustering)
      ACF of r^2 at lag 5 -> kappa   (a second lag pins the decay rate)
    """
    r = np.asarray(r, float)
    s = r.std(ddof=1)
    z = (r - r.mean()) / s
    return np.array([s * np.sqrt(periods_per_year),
                     float((z ** 4).mean() - 3.0),
                     float((z ** 3).mean()),
                     _acf1(r ** 2, 1),
                     _acf1(r ** 2, 5)])


def _simulate_daily_returns(params, n_days, n_rep, z1, z2, periods_per_year=TRADING_DAYS):
    """Daily log returns from Heston, driven by pre-drawn normals.

    The caller supplies z1 and z2 so that every objective evaluation during
    calibration sees the SAME random numbers (common random numbers). Without
    this the objective is a noisy function of the parameters and no optimiser
    can descend it.
    """
    kappa, xi, rho, theta = params["kappa"], params["xi"], params["rho"], params["theta"]
    dt = 1.0 / periods_per_year
    v = np.full(n_rep, theta)
    out = np.empty((n_rep, n_days))
    for t in range(n_days):
        w2 = rho * z1[:, t] + np.sqrt(1.0 - rho ** 2) * z2[:, t]
        vp = np.maximum(v, 0.0)
        out[:, t] = -0.5 * vp * dt + np.sqrt(vp * dt) * z1[:, t]
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * w2
    return out


def calibrate_heston(r, periods_per_year=TRADING_DAYS, seed=0, n_rep=24, verbose=False):
    """Simulated method of moments for (kappa, xi, rho); theta is pinned.

    Why not plain moment matching on a realised-variance series? Because
    RV_t is a noisy estimate of the true variance, RV = v + noise. The noise
    inflates Var(RV), which blows up the vol-of-vol estimate, and deflates the
    autocorrelation of RV, which blows up the mean-reversion speed. We measured
    exactly that failure: kappa near 28 per year and a Feller condition
    violated by a wide margin.

    Instead we choose the parameters so that returns SIMULATED from the model
    reproduce the moments we actually observe in real returns. The noise then
    sits in both the real and simulated moments and largely cancels.

    theta is set to the sample variance so the annualised volatility matches by
    construction, leaving three free parameters against four target moments.
    """
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    n_days = len(r)
    target = return_moments(r, periods_per_year)
    theta = float(r.var(ddof=1) * periods_per_year)

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_rep, n_days))
    z2 = rng.standard_normal((n_rep, n_days))

    # moments are on wildly different scales, so weight by the target size
    scale = np.array([target[0], max(abs(target[1]), 1.0), max(abs(target[2]), 0.1),
                      max(abs(target[3]), 0.01), max(abs(target[4]), 0.01)])

    def unpack(u):
        # unconstrained -> constrained, so the optimiser never leaves the region
        kappa = 0.5 + 19.5 / (1 + np.exp(-u[0]))        # (0.5, 20)
        xi = 0.05 + 2.45 / (1 + np.exp(-u[1]))          # (0.05, 2.5)
        rho = -0.95 / (1 + np.exp(-u[2]))               # (-0.95, 0)
        return {"kappa": float(kappa), "xi": float(xi), "rho": float(rho), "theta": theta}

    def obj(u):
        p = unpack(u)
        sims = _simulate_daily_returns(p, n_days, n_rep, z1, z2, periods_per_year)
        m = np.mean([return_moments(s, periods_per_year) for s in sims], axis=0)
        return float(np.sum(((m - target) / scale) ** 2))

    best, best_f = None, np.inf
    for u0 in ([0.0, 0.0, 0.0], [-1.5, 1.0, -1.0], [1.5, -1.0, 1.0]):
        res = minimize(obj, u0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6})
        if res.fun < best_f:
            best, best_f = res.x, res.fun
        if verbose:
            print("   start", u0, "-> obj %.5f" % res.fun)

    p = unpack(best)
    sims = _simulate_daily_returns(p, n_days, n_rep, z1, z2, periods_per_year)
    fitted = np.mean([return_moments(s, periods_per_year) for s in sims], axis=0)
    feller = 2 * p["kappa"] * p["theta"] - p["xi"] ** 2
    p.update({"v0": theta, "long_run_vol": float(np.sqrt(theta)),
              "feller_2kt_minus_xi2": float(feller),
              "feller_satisfied": bool(feller > 0),
              "smm_objective": float(best_f),
              "target_moments": [float(x) for x in target],
              "fitted_moments": [float(x) for x in fitted],
              "moment_names": ["ann_vol", "excess_kurtosis", "skewness",
                               "acf1_r2", "acf5_r2"]})
    return p


# ---------------------------------------------------------------- horizon calibration

def _rolling_std(r, w):
    """Std of r[i : i + w] for every i, via cumulative sums (ddof = 1)."""
    c1 = np.concatenate([[0.0], np.cumsum(r)])
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    var = (s2 - s1 * s1 / w) / (w - 1)
    return np.sqrt(np.maximum(var, 0.0))


def horizon_stats(r, hist=60, steps=30, periods_per_year=TRADING_DAYS):
    """Statistics of the volatility SURPRISE at the hedging horizon.

    For every window: quoted vol = std of the `hist` returns before inception,
    realised vol = std of the `steps` returns of the episode. Exactly the
    protocol the samplers use. Returns the moments we calibrate to:

      median of log(realised / quoted)
      std    of log(realised / quoted)
      95th percentile of log(realised / quoted)    the tail that drives CVaR
      corr(episode log return, log ratio)          leverage, at this horizon
    """
    r = np.asarray(r, float)
    q = _rolling_std(r, hist)                 # window r[i : i + hist]
    e = _rolling_std(r, steps)
    n = len(r) - hist - steps + 1
    quoted = q[:n]
    real = e[hist:hist + n]
    lr = np.log(np.maximum(real, 1e-12) / np.maximum(quoted, 1e-12))
    c = np.concatenate([[0.0], np.cumsum(r)])
    ret = c[hist + steps:hist + steps + n] - c[hist:hist + n]
    return np.array([np.median(lr), lr.std(), np.percentile(lr, 95),
                     np.corrcoef(ret, lr)[0, 1]])


def calibrate_heston_horizon(r, periods_per_year=TRADING_DAYS, seed=0, n_rep=8,
                             hist=60, steps=30, verbose=False):
    """Simulated method of moments on the horizon statistics above.

    Why this target and not daily moments: to first order the P&L of a hedged
    short option is (1/2) Gamma S^2 (sigma_quoted^2 - sigma_realised^2) dt, so
    what a hedging simulator must get right is the distribution of
    realised / quoted volatility over the life of the option. Fitting daily
    kurtosis instead forced vol-of-vol to about 2, which made volatility far
    too erratic at the 30-day horizon. See README, "the simulator we threw
    away".

    theta is pinned to the sample variance, as before. Three free parameters
    (kappa, xi, rho) against four horizon moments.
    """
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    n_days = len(r)
    target = horizon_stats(r, hist, steps, periods_per_year)
    theta = float(r.var(ddof=1) * periods_per_year)
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_rep, n_days + 120))
    z2 = rng.standard_normal((n_rep, n_days + 120))
    scale = np.array([0.05, 0.05, 0.05, 0.1])

    def unpack(u):
        kappa = 0.2 + 29.8 / (1 + np.exp(-u[0]))      # (0.2, 30)
        xi = 0.02 + 2.48 / (1 + np.exp(-u[1]))        # (0.02, 2.5)
        rho = -0.95 / (1 + np.exp(-u[2]))             # (-0.95, 0)
        return {"kappa": float(kappa), "xi": float(xi), "rho": float(rho),
                "theta": theta}

    def sim_stats(p):
        sims = _simulate_daily_returns(p, n_days + 120, n_rep, z1, z2, periods_per_year)
        return np.mean([horizon_stats(s[120:], hist, steps) for s in sims], axis=0)

    def obj(u):
        return float(np.sum(((sim_stats(unpack(u)) - target) / scale) ** 2))

    best, best_f = None, np.inf
    for u0 in ([0.0, -1.0, 0.0], [-1.5, -2.0, 1.0], [1.5, 0.0, -1.0], [0.0, -3.0, 0.5]):
        res = minimize(obj, u0, method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-6})
        if verbose:
            print("   start", u0, "->", round(res.fun, 5), unpack(res.x))
        if res.fun < best_f:
            best, best_f = res.x, res.fun
    p = unpack(best)
    fitted = sim_stats(p)
    feller = 2 * p["kappa"] * p["theta"] - p["xi"] ** 2
    p.update({"v0": theta, "long_run_vol": float(np.sqrt(theta)),
              "feller_2kt_minus_xi2": float(feller), "feller_satisfied": bool(feller > 0),
              "smm_objective": float(best_f),
              "target_moments": target.tolist(), "fitted_moments": fitted.tolist(),
              "moment_names": ["median_log_ratio", "sd_log_ratio", "p95_log_ratio",
                               "corr_return_log_ratio"],
              "target": "horizon"})
    return p
