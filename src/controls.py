"""Correctness controls for the engine and the rule-based strategies.

These are not unit tests of the "does it run" kind. Each one checks a property
that must hold if the mathematics is implemented correctly, and would fail
loudly if it were not. We run this before trusting any result.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import engine
import strategies as st
import simulators as sim

torch.set_default_dtype(torch.float64)

K, T, SIG, N = 100.0, 30 / 252, 0.20, 30
PREM = float(engine.bs(torch.tensor(100.0), K, torch.tensor(T),
                       torch.tensor(SIG))[0])


def paths(n_steps, n_paths, seed):
    return torch.tensor(sim.gbm_paths(100.0, SIG, T, n_steps, n_paths,
                                      np.random.default_rng(seed)))


def main():
    ok = True
    print("premium of the ATM 30-day call: %.6f\n" % PREM)

    # ---- 1. BS delta and gamma against finite differences of the price
    S = torch.tensor([100.0]); tau = torch.tensor([T]); sg = torch.tensor([SIG])
    p, d, g = engine.bs(S, K, tau, sg)
    h = 1e-5 * S
    pu, _, _ = engine.bs(S + h, K, tau, sg)
    pd_, _, _ = engine.bs(S - h, K, tau, sg)
    e_d = abs(float(d) / float((pu - pd_) / (2 * h)) - 1)
    e_g = abs(float(g) / float((pu - 2 * p + pd_) / h ** 2) - 1)
    print("1. delta vs finite difference   rel err %.2e" % e_d)
    print("   gamma vs finite difference   rel err %.2e" % e_g)
    ok &= e_d < 1e-6 and e_g < 1e-4

    # ---- 2. at zero cost, finer rebalancing must monotonically reduce risk
    print("\n2. zero cost: risk must fall monotonically as we rebalance more")
    prev = -1e18
    for n in (5, 10, 30, 60, 120):
        Sx = paths(n, 40000, 1)
        tot, _, _ = engine.pnl(Sx, st.bs_delta(Sx, K, T, SIG, n), K, 0.0, PREM)
        c = float(engine.cvar(tot))
        print("   n=%4d  CVaR95 %9.4f  sd %7.4f  mean %8.5f"
              % (n, c, float(tot.std()), float(tot.mean())))
        ok &= c > prev
        prev = c

    # ---- 3. at zero cost the mean P&L must be statistically zero
    Sx = paths(N, 200000, 2)
    tot, _, _ = engine.pnl(Sx, st.bs_delta(Sx, K, T, SIG, N), K, 0.0, PREM)
    se = float(tot.std()) / np.sqrt(tot.numel())
    z = abs(float(tot.mean()) / se)
    print("\n3. zero cost: mean P&L = %.5f, standard error %.5f, |z| = %.2f"
          % (float(tot.mean()), se, z))
    ok &= z < 4.0

    # ---- 4. hedging must beat doing nothing, by a lot
    tu, _, _ = engine.pnl(Sx, st.unhedged(Sx, N), K, 0.0, PREM)
    print("\n4. unhedged CVaR %9.4f  vs  delta CVaR %9.4f"
          % (float(engine.cvar(tu)), float(engine.cvar(tot))))
    ok &= engine.cvar(tu) < engine.cvar(tot)

    # ---- 5. a zero-width band must reproduce plain delta exactly
    hb, _ = st.band(Sx, K, T, SIG, N, 0.0)
    hd = st.bs_delta(Sx, K, T, SIG, N)
    diff = float((hb - hd).abs().max())
    print("\n5. band(width=0) vs bs_delta   max abs difference %.2e" % diff)
    ok &= diff < 1e-12

    # ---- 6. a very wide band must never trade
    _, tr = st.band(Sx, K, T, SIG, N, 10.0)
    print("6. band(width=10) mean trades  %.4f (must be 0)" % float(tr.mean()))
    ok &= float(tr.mean()) == 0.0

    # ---- 7. cost must be monotone in the cost rate for a fixed strategy
    print("\n7. same holdings, rising cost: fee paid must rise, P&L must fall")
    last_c, last_p = -1.0, 1e18
    for cb in (0.0, 5.0, 20.0, 50.0):
        t2, c2, _ = engine.pnl(Sx, hd, K, cb, PREM)
        mc, mp = float(c2.mean()), float(t2.mean())
        print("   %5.1f bps  mean cost %8.4f  mean P&L %9.4f" % (cb, mc, mp))
        ok &= mc > last_c and mp < last_p
        last_c, last_p = mc, mp

    # ---- 8. Leland volatility behaves as advertised
    print("\n8. Leland sigma: k=0 -> %.6f (must equal %.2f), k=20bps -> %.6f"
          % (st.leland_sigma(SIG, 0.0, T, N), SIG,
             st.leland_sigma(SIG, 20.0, T, N)))
    ok &= st.leland_sigma(SIG, 0.0, T, N) == SIG
    ok &= st.leland_sigma(SIG, 20.0, T, N) > SIG

    # ---- 9. Whalley-Wilmott collapses to plain delta as cost goes to zero
    hw, _ = st.whalley_wilmott(Sx, K, T, SIG, N, 0.0, 1.0)
    dw = float((hw - hd).abs().max())
    print("\n9. Whalley-Wilmott at k=0 vs bs_delta   max abs diff %.2e" % dw)
    ok &= dw < 1e-12

    # ---- 10. no look-ahead: holdings at t must not change if the future does
    Sa = paths(N, 2000, 7)
    Sb = Sa.clone()
    Sb[:, N // 2 + 1:] *= 1.05                 # perturb only the future
    ha = st.bs_delta(Sa, K, T, SIG, N)[:, : N // 2 + 1]
    hb2 = st.bs_delta(Sb, K, T, SIG, N)[:, : N // 2 + 1]
    la = float((ha - hb2).abs().max())
    hA, _ = st.band(Sa, K, T, SIG, N, 0.05)
    hB, _ = st.band(Sb, K, T, SIG, N, 0.05)
    lb = float((hA[:, : N // 2 + 1] - hB[:, : N // 2 + 1]).abs().max())
    print("\n10. causality: perturbing the future changes past holdings by")
    print("    bs_delta %.2e   band %.2e   (both must be 0)" % (la, lb))
    ok &= la == 0.0 and lb == 0.0

    print("\n" + ("ALL CONTROLS PASSED" if ok else "*** A CONTROL FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
