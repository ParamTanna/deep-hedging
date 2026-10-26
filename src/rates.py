"""Does the zero-interest-rate assumption change any conclusion?

The main study sets r = 0 (see engine.py for why). The argument was that a
financing rate shifts every strategy by nearly the same amount, because they
all carry a similar average inventory, so it moves the level of the results
but not their ranking. This module checks that claim instead of asserting it.

Doing r > 0 consistently means changing three things together, which is
exactly why the main study avoids it:

  1. the premium is the Black-Scholes price with rate r,
  2. simulated paths drift at r (the risk-neutral measure the premium uses),
  3. cash earns r between rebalances, and the premium accrues to expiry.

Getting any one of the three wrong hands some strategy a free edge. With
r = 0 everything here reduces exactly to engine.py, which test_reduces()
verifies to the last bit.
"""
from __future__ import annotations
import math
import numpy as np
import torch

from engine import _ncdf, _npdf
from samplers import Episodes, TRADING_DAYS, HIST, STEPS, _normalise


def bs_r(S, K, tau, sigma, r):
    S = torch.as_tensor(S)
    dt = S.dtype if S.is_floating_point() else torch.float64
    S = S.to(dt)
    tau = torch.as_tensor(tau, dtype=dt).clamp_min(1e-10)
    sigma = torch.as_tensor(sigma, dtype=dt)
    srt = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / srt
    d2 = d1 - srt
    price = S * _ncdf(d1) - K * torch.exp(-r * tau) * _ncdf(d2)
    return price, _ncdf(d1), _npdf(d1) / (S * srt)


def pnl_r(S, holdings, K, cost_bps, premium, r, dt=1.0 / TRADING_DAYS):
    """engine.pnl with interest on cash and on the premium."""
    k = cost_bps * 1e-4
    g = math.exp(r * dt)
    n = holdings.shape[1]
    prev = torch.zeros(S.shape[0], dtype=S.dtype)
    cash = torch.zeros(S.shape[0], dtype=S.dtype)
    cost = torch.zeros(S.shape[0], dtype=S.dtype)
    for t in range(n):
        if t > 0:
            cash = cash * g
        trade = holdings[:, t] - prev
        fee = k * trade.abs() * S[:, t]
        cash = cash - trade * S[:, t] - fee
        cost = cost + fee
        prev = holdings[:, t]
    cash = cash * g
    ST = S[:, -1]
    fee = k * prev.abs() * ST
    cash = cash + prev * ST - fee
    cost = cost + fee
    total = premium * math.exp(r * n * dt) + cash - torch.clamp(ST - K, min=0.0)
    return total, cost


def gbm_r(n_paths, rng, sigma_pool, r, hist=HIST, steps=STEPS):
    sig = rng.choice(sigma_pool, size=n_paths, replace=True)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_paths, hist + steps))
    incr = (r - 0.5 * sig[:, None] ** 2) * dt + sig[:, None] * np.sqrt(dt) * z
    S = np.hstack([np.ones((n_paths, 1)), np.exp(np.cumsum(incr, axis=1))])
    return Episodes(torch.from_numpy(_normalise(S, hist)), torch.from_numpy(sig), hist, steps)


def delta_r(S, K, T, sigma, n_steps, r):
    out = torch.empty(S.shape[0], n_steps, dtype=S.dtype)
    for t in range(n_steps):
        out[:, t] = bs_r(S[:, t], K, T * (1 - t / n_steps), sigma, r)[1]
    return out


def test_reduces():
    """At r = 0 every function here must equal its engine.py counterpart exactly."""
    import engine
    import strategies as st
    rng = np.random.default_rng(3)
    pool = np.array([0.1, 0.2, 0.35])
    ep = gbm_r(3000, rng, pool, 0.0)
    T = STEPS / TRADING_DAYS
    S = ep.episode
    p0, d0, g0 = engine.bs(S[:, 0], 100.0, T, ep.sigma_imp)
    p1, d1, g1 = bs_r(S[:, 0], 100.0, T, ep.sigma_imp, 0.0)
    H = st.bs_delta(S, 100.0, T, ep.sigma_imp, STEPS)
    a, _, _ = engine.pnl(S, H, 100.0, 10.0, p0)
    b, _ = pnl_r(S, delta_r(S, 100.0, T, ep.sigma_imp, STEPS, 0.0), 100.0, 10.0, p1, 0.0)
    gaps = [float((p0 - p1).abs().max()), float((d0 - d1).abs().max()),
            float((g0 - g1).abs().max()), float((a - b).abs().max())]
    return max(gaps)




# ------------------------------------------------------------------ the ablation

def rules_r(name, S, sig, T, n, cost, tuned, r):
    """The rule strategies with the rate-consistent delta."""
    k = cost * 1e-4
    out = torch.empty(S.shape[0], n, dtype=S.dtype)
    held = torch.zeros(S.shape[0], dtype=S.dtype)
    for t in range(n):
        tau = T * (1 - t / n)
        s = sig
        if name == "leland" and k > 0:
            s = sig * torch.sqrt(1 + math.sqrt(2 / math.pi) * 2 * k / (sig * math.sqrt(T / n)))
        _, d, g = bs_r(S[:, t], 100.0, tau, s, r)
        if name in ("bs_delta", "leland"):
            held = d
        elif name == "band":
            held = torch.where((d - held).abs() > tuned["band_w"], d, held)
        elif name == "ww":
            hb = torch.zeros_like(d) if k <= 0 else torch.clamp(
                (1.5 * k * S[:, t] * g ** 2 / tuned["ww_gamma"]) ** (1 / 3), max=1.0)
            move = (d - held).abs() > hb
            held = torch.where(move, torch.where(d > held, d - hb, d + hb), held)
        out[:, t] = held
    return out


def ablation(policies: dict, tuned: dict, cost=10.0, n_paths=50000, rates=(0.0, 0.065)):
    """CVaR of every strategy at r = 0 and at r = 6.5%, on matched paths.

    The same normals drive both worlds (same seed), so the only difference
    between the two columns is the rate.
    """
    import engine
    import policy as pol
    pool = np.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "..",
                                              "data", "processed", "vol_pool_pre2019.npy"))
    T = STEPS / TRADING_DAYS
    res = {}
    for r in rates:
        ep = gbm_r(n_paths, np.random.default_rng(4242), pool, r)
        S = ep.episode
        prem = bs_r(torch.full((S.shape[0],), 100.0, dtype=S.dtype), 100.0, T, ep.sigma_imp, r)[0]
        row = {}
        for nm in ("bs_delta", "leland", "band", "ww"):
            H = rules_r(nm, S, ep.sigma_imp, T, STEPS, cost, tuned, r)
            row[nm] = float(engine.cvar(pnl_r(S, H, 100.0, cost, prem, r)[0]))
        # policies trained at r = 0 see FORWARD prices F = S exp(r tau)
        tau = torch.tensor([T * (1 - t / STEPS) for t in range(STEPS)] + [0.0], dtype=S.dtype)
        Sf = ep.S.clone()
        Sf[:, HIST:] = S * torch.exp(r * tau)[None, :]
        ep_f = Episodes(Sf, ep.sigma_imp, ep.hist, ep.steps)
        for nm, p in policies.items():
            with torch.no_grad():
                H = pol.rollout(p, ep_f).double()
            row[nm] = float(engine.cvar(pnl_r(S, H, 100.0, cost, prem, r)[0]))
        res[str(r)] = row
    a, b = res[str(rates[0])], res[str(rates[1])]
    res["shift"] = {k: b[k] - a[k] for k in a}
    res["rank_r0"] = sorted(a, key=lambda k: -a[k])
    res["rank_r"] = sorted(b, key=lambda k: -b[k])
    res["ranking_unchanged"] = res["rank_r0"] == res["rank_r"]
    return res


if __name__ == "__main__":
    print("max difference from engine.py at r = 0: %.2e (must be ~0)" % test_reduces())
