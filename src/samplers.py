"""Turn a data source into hedging episodes, using one protocol for all of them.

An episode is what one desk lives through when it sells one option:

    [ 60 trading days of history ][ 30 trading days hedged until expiry ]
                                  ^
                                  inception: the option is sold here

At inception the desk does not know the true volatility. It knows the recent
past. So it quotes a volatility from the 60-day history, prices the option with
Black-Scholes at that volatility, collects the premium, and then hedges. The
same protocol is applied to simulated GBM paths, simulated Heston paths and
real index history. Only that makes the simulated and real numbers comparable.

Every path is rescaled so the price at inception is exactly 100 and the strike
is 100 (an at-the-money call). Profit and loss is therefore per 100 of notional,
which lets NIFTY and the S&P 500 sit in the same table even though one trades
near 24,000 and the other near 6,000.

GBM is the one place we deviate from the protocol. For the clean experiments
we want the desk to know the true volatility, so that Black-Scholes delta is
genuinely optimal at zero cost and the zero-cost control has a known answer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

TRADING_DAYS = 252
HIST = 60          # days of history used to quote the volatility
STEPS = 30         # rebalance dates in one episode (daily, about six weeks)


@dataclass
class Episodes:
    S: torch.Tensor            # (B, HIST + STEPS + 1), S[:, HIST] == 100
    sigma_imp: torch.Tensor    # (B,) volatility the desk quotes at inception
    hist: int = HIST
    steps: int = STEPS
    dates: list | None = None  # inception dates, real data only

    @property
    def episode(self) -> torch.Tensor:
        """The hedged part only: (B, STEPS + 1)."""
        return self.S[:, self.hist:]

    def to(self, dtype=None, device=None):
        return Episodes(self.S.to(device=device, dtype=dtype),
                        self.sigma_imp.to(device=device, dtype=dtype),
                        self.hist, self.steps, self.dates)

    def subset(self, idx):
        d = None if self.dates is None else [self.dates[i] for i in idx]
        return Episodes(self.S[idx], self.sigma_imp[idx], self.hist, self.steps, d)


def _normalise(S: np.ndarray, hist: int) -> np.ndarray:
    return 100.0 * S / S[:, hist:hist + 1]


def realised_vol(S: np.ndarray, hist: int) -> np.ndarray:
    """Annualised standard deviation of the daily log returns in the history."""
    r = np.diff(np.log(S[:, :hist + 1]), axis=1)
    return r.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)


# ------------------------------------------------------------------ GBM

def vol_pool(returns_by_index: list[np.ndarray], window: int = HIST) -> np.ndarray:
    """Every trailing-window volatility that occurred in the training era.

    Drawing training volatilities from this pool means the policy is trained on
    the regimes a desk actually lived through before 2019, crises included,
    instead of a range we picked by hand.
    """
    out = []
    for r in returns_by_index:
        s = pd.Series(r).rolling(window).std().dropna().values * np.sqrt(TRADING_DAYS)
        out.append(s)
    return np.concatenate(out)


def gbm(n_paths: int, rng: np.random.Generator, sigma_pool: np.ndarray | None = None,
        sigma: float | None = None, hist: int = HIST, steps: int = STEPS) -> Episodes:
    if sigma is not None:
        sig = np.full(n_paths, float(sigma))
    else:
        sig = rng.choice(sigma_pool, size=n_paths, replace=True)
    dt = 1.0 / TRADING_DAYS
    n = hist + steps
    z = rng.standard_normal((n_paths, n))
    incr = -0.5 * (sig[:, None] ** 2) * dt + sig[:, None] * np.sqrt(dt) * z
    S = np.hstack([np.ones((n_paths, 1)), np.exp(np.cumsum(incr, axis=1))])
    S = _normalise(S, hist)
    return Episodes(torch.from_numpy(S), torch.from_numpy(sig), hist, steps)


# ------------------------------------------------------------------ Heston

def heston(n_paths: int, rng: np.random.Generator, p: dict, burn: int = 60,
           hist: int = HIST, steps: int = STEPS) -> Episodes:
    """Heston paths with a burn-in, then the standard history + episode.

    The variance starts at its long-run mean theta and runs for `burn` days
    before the history window opens. With the calibrated kappa near 18 per
    year the relaxation time 1/kappa is about 14 trading days, so 60 days of
    burn-in is four relaxation times: enough that the history window sees a
    variance drawn from (close to) the stationary distribution, not one pinned
    at theta.

    Here the desk does NOT know the true variance. It quotes the realised
    volatility of the 60-day history, exactly as it must on real data.
    """
    kappa, theta, xi, rho = p["kappa"], p["theta"], p["xi"], p["rho"]
    dt = 1.0 / TRADING_DAYS
    n = burn + hist + steps
    logS = np.zeros((n_paths, n + 1))
    v = np.full(n_paths, theta)
    for t in range(n):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        logS[:, t + 1] = logS[:, t] - 0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    S = np.exp(logS[:, burn:])
    S = _normalise(S, hist)
    sig = realised_vol(S, hist)
    return Episodes(torch.from_numpy(S), torch.from_numpy(sig), hist, steps)


# ------------------------------------------------------------------ real data

def real(df: pd.DataFrame, start: str, end: str | None = None,
         hist: int = HIST, steps: int = STEPS) -> Episodes:
    """Overlapping windows from real history, one starting on every trading day.

    A window qualifies if its INCEPTION date lies in [start, end] and the whole
    history plus episode fits in the data. Overlapping windows are the
    efficient way to use a short history, but neighbouring windows share most
    of their path, so they are strongly dependent. Any confidence interval
    built on them must respect that, which is why evaluation uses a block
    bootstrap rather than treating windows as independent.
    """
    px = df["adjclose"].to_numpy(float)
    dates = pd.to_datetime(df["date"]).to_numpy()
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else pd.Timestamp.max
    rows, incs = [], []
    for i in range(hist, len(px) - steps):
        d = pd.Timestamp(dates[i])
        if lo <= d <= hi:
            rows.append(px[i - hist:i + steps + 1])
            incs.append(str(d.date()))
    S = _normalise(np.asarray(rows), hist)
    sig = realised_vol(S, hist)
    return Episodes(torch.from_numpy(S), torch.from_numpy(sig), hist, steps, incs)
