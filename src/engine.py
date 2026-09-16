"""Black-Scholes formulas and the profit-and-loss engine.

Design decision worth stating up front: there is ONE engine, written in
PyTorch, and every strategy is scored by it. The Black-Scholes delta, the
no-trade band and the neural policy all produce the same object, a tensor of
holdings, and all three are pushed through the same accounting code on the
same price paths. If the engine has a quirk, every strategy inherits it
equally, so the comparison stays fair. That is the single most important
property of this study, because the whole claim is a comparison.

Interest rate convention. We set r = 0 and work in forward terms. The reasons,
in order of importance:

  1. Consistency. The premium, the simulated paths and the accounting then all
     live in the same measure. With r > 0 the Black-Scholes premium is computed
     under a drift of r, so simulated paths must also drift at r, and the cash
     balance must accrue interest. Getting any one of those three wrong hands
     one strategy a free edge. Setting r = 0 removes the whole class of bug.
  2. Relevance. The quantity we study is the trade-off between hedging error
     and trading cost. A financing rate shifts every strategy by nearly the
     same amount because they all carry a similar average inventory, so it
     moves the level and not the ranking. We check that claim in an ablation
     rather than asserting it.
  3. Precedent. The deep hedging literature works in discounted terms for the
     same reason.

Sign convention: we are SHORT one call. We receive the premium at inception and
owe max(S_T - K, 0) at expiry.
"""
from __future__ import annotations
import math
import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


def _ncdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF. torch has erf, so we build Phi from it directly."""
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def _npdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs(S, K, tau, sigma):
    """Call price, delta and gamma at zero rate.

    tau may be a scalar or a tensor broadcastable against S. At tau = 0 the
    formulas are singular, so we floor tau. The floor only ever applies at the
    final rebalance date, where delta is already essentially 0 or 1.
    """
    S = torch.as_tensor(S, dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64).clamp_min(1e-10)
    sigma = torch.as_tensor(sigma, dtype=torch.float64)
    srt = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * tau) / srt
    d2 = d1 - srt
    price = S * _ncdf(d1) - K * _ncdf(d2)
    delta = _ncdf(d1)
    gamma = _npdf(d1) / (S * srt)
    return price, delta, gamma


def pnl(S: torch.Tensor, holdings: torch.Tensor, K: float, cost_bps: float,
        premium: torch.Tensor | float, liquidate_cost: bool = True):
    """Terminal profit and loss of a hedged short call.

    S         (n_paths, n_steps+1)  price path, S[:, 0] is inception
    holdings  (n_paths, n_steps)    units held over interval [t, t+1)
    premium   received at inception, scalar or (n_paths,)

    Accounting, step by step, exactly as a desk would do it:

        at t:    trade  = h_t - h_{t-1}            (h_{-1} = 0)
                 cash  -= trade * S_t              pay for the shares
                 cash  -= k * |trade| * S_t        pay the fee
        at T:    cash  += h_{n-1} * S_T            sell the inventory
                 cash  -= k * |h_{n-1}| * S_T      fee on that sale
                 cash  -= max(S_T - K, 0)          settle the option

    The fee on the final unwind is included by default. Leaving it out would
    quietly reward a strategy for carrying a large position into expiry, which
    is not free in reality.
    """
    k = cost_bps * 1e-4
    n_steps = holdings.shape[1]
    prev = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    cash = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    cost = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    turnover = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    for t in range(n_steps):
        trade = holdings[:, t] - prev
        fee = k * trade.abs() * S[:, t]
        cash = cash - trade * S[:, t] - fee
        cost = cost + fee
        turnover = turnover + trade.abs() * S[:, t]
        prev = holdings[:, t]
    ST = S[:, -1]
    cash = cash + prev * ST
    if liquidate_cost:
        fee = k * prev.abs() * ST
        cash = cash - fee
        cost = cost + fee
        turnover = turnover + prev.abs() * ST
    payoff = torch.clamp(ST - K, min=0.0)
    total = premium + cash - payoff
    return total, cost, turnover
