"""The rule-based hedging strategies, in increasing order of sophistication.

Every function here returns a holdings tensor of shape (n_paths, n_steps) and
nothing else, so engine.pnl can score it. Every one is CAUSAL: the holding
chosen for the interval starting at t uses only information observable at t.
That is not a stylistic point. A strategy that peeked at S_{t+1} would look
spectacular and mean nothing.

Four strategies:

  unhedged      Hold nothing. The null model. It tells us what the risk is with
                no strategy at all, so every later number has a reference.

  bs_delta      Hold N(d1) at every rebalance date. This is the right primary
                baseline because it is provably optimal in the limit of
                continuous, costless trading, so it is not a straw man. Under
                cost it is deliberately naive: it chases every small move.

  band          Hold the delta, but only re-trade once the position has drifted
                more than w away from it. One free parameter, tuned on
                validation paths and never on test.

  whalley_wilmott
                The same idea but with the width derived rather than guessed.
                Whalley and Wilmott (1997) solve the singular stochastic
                control problem asymptotically for small proportional cost k
                under exponential utility with risk aversion gamma, and obtain
                a band of half-width

                    H_t = ( 1.5 * k * S_t * Gamma_t^2 / gamma ) ^ (1/3)

                The shape is the interesting part. H grows with the cost k, so
                you trade less when trading is dear. H grows with Gamma^2, so
                you tolerate more drift exactly where the delta is most curved
                and therefore most expensive to track. And H shrinks as risk
                aversion rises. A constant-width band has none of that
                structure, which is why this is the baseline the neural policy
                actually has to beat.

The Leland (1985) adjustment can be applied to any delta-based rule. It
inflates the volatility fed to the delta formula,

    sigma_adj^2 = sigma^2 * (1 + sqrt(2/pi) * 2k / (sigma * sqrt(dt)))

which flattens the delta profile and so reduces how much one trades. Leland
adjusts WHAT you aim at; the band adjusts WHEN you trade. They are
complementary, and we report both.
"""
from __future__ import annotations
import math
import torch
from engine import bs


def _taus(T: float, n_steps: int, dtype, device):
    """Time remaining at each rebalance date: T, T(1 - 1/n), ..., T/n."""
    i = torch.arange(n_steps, dtype=dtype, device=device)
    return T * (1.0 - i / n_steps)


def leland_sigma(sigma: float, cost_bps: float, T: float, n_steps: int) -> float:
    """Leland cost-adjusted volatility. Returns sigma unchanged at zero cost."""
    k = cost_bps * 1e-4
    if k <= 0:
        return float(sigma)
    dt = T / n_steps
    a = math.sqrt(2.0 / math.pi) * 2.0 * k / (sigma * math.sqrt(dt))
    return float(sigma * math.sqrt(1.0 + a))


def unhedged(S: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.zeros(S.shape[0], n_steps, dtype=S.dtype, device=S.device)


def _as_vec(sigma, n, dtype, device):
    sig = torch.as_tensor(sigma, dtype=dtype, device=device)
    return sig.expand(n) if sig.ndim == 0 else sig


def bs_delta(S: torch.Tensor, K: float, T: float, sigma, n_steps: int) -> torch.Tensor:
    tau = _taus(T, n_steps, S.dtype, S.device)
    sig = _as_vec(sigma, S.shape[0], S.dtype, S.device)
    out = torch.empty(S.shape[0], n_steps, dtype=S.dtype, device=S.device)
    for t in range(n_steps):
        _, d, _ = bs(S[:, t], K, tau[t], sig)
        out[:, t] = d
    return out


def band(S: torch.Tensor, K: float, T: float, sigma, n_steps: int,
         width: float):
    """Constant-width no-trade band. Returns (holdings, n_trades)."""
    tau = _taus(T, n_steps, S.dtype, S.device)
    sig = _as_vec(sigma, S.shape[0], S.dtype, S.device)
    held = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    out = torch.empty(S.shape[0], n_steps, dtype=S.dtype, device=S.device)
    trades = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    for t in range(n_steps):
        _, target, _ = bs(S[:, t], K, tau[t], sig)
        move = (target - held).abs() > width
        held = torch.where(move, target, held)
        trades = trades + move.to(S.dtype)
        out[:, t] = held
    return out, trades


def whalley_wilmott(S: torch.Tensor, K: float, T: float, sigma, n_steps: int,
                    cost_bps: float, gamma_risk: float):
    """Asymptotically optimal band, width derived from local gamma."""
    k = cost_bps * 1e-4
    tau = _taus(T, n_steps, S.dtype, S.device)
    sig = _as_vec(sigma, S.shape[0], S.dtype, S.device)
    held = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    out = torch.empty(S.shape[0], n_steps, dtype=S.dtype, device=S.device)
    trades = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    for t in range(n_steps):
        _, target, g = bs(S[:, t], K, tau[t], sig)
        if k <= 0:
            h_band = torch.zeros_like(target)
        else:
            h_band = (1.5 * k * S[:, t] * g ** 2 / gamma_risk) ** (1.0 / 3.0)
            h_band = torch.clamp(h_band, max=1.0)
        move = (target - held).abs() > h_band
        # when we do trade, move to the nearer edge of the band, not the centre
        edge = torch.where(target > held, target - h_band, target + h_band)
        held = torch.where(move, edge, held)
        trades = trades + move.to(S.dtype)
        out[:, t] = held
    return out, trades


def count_trades(holdings: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    """Number of rebalances actually executed, including the opening trade."""
    zero = torch.zeros(holdings.shape[0], 1, dtype=holdings.dtype,
                       device=holdings.device)
    prev = torch.cat([zero, holdings[:, :-1]], dim=1)
    return ((holdings - prev).abs() > tol).sum(dim=1).to(holdings.dtype)


def trailing_vol(S: torch.Tensor, window: int, periods_per_year: int,
                 fallback: float) -> torch.Tensor:
    """Causal trailing realised volatility at each rebalance date.

    At step t we use only the log returns observed strictly before t. For the
    first few steps there is not enough history inside the episode, so we fall
    back to the volatility estimated from the training period. That fallback is
    training-set information, which is allowed. What is not allowed is using
    returns from later in the episode, and we do not.
    """
    n = S.shape[1] - 1
    r = torch.log(S[:, 1:] / S[:, :-1])
    out = torch.full((S.shape[0], n), float(fallback), dtype=S.dtype,
                     device=S.device)
    for t in range(1, n):
        lo = max(0, t - window)
        seg = r[:, lo:t]
        if seg.shape[1] >= 2:
            out[:, t] = seg.std(dim=1, unbiased=True) * math.sqrt(periods_per_year)
    return out
