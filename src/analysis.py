"""What did the networks actually learn, and where do the strategies fail?

    python src/analysis.py

Writes results/analysis.json. Four analyses:

1. Zero-cost control. With no trading cost the right answer is (close to) the
   Black-Scholes delta. We measure how far the learned policy sits from it.

2. Band emergence. The plain MLP is given the current holding as an input and
   nothing else about no-trade regions. For a fixed state we sweep the current
   holding h_prev and look at the new holding. Where new == old, the policy
   chose not to trade. The set of such h_prev is the learned no-trade band.

3. The k^(1/3) law. Small-cost asymptotics (Whalley and Wilmott 1997) say the
   optimal band half-width grows like cost^(1/3). Derivation in README. We fit
   the exponent of the learned band width across our cost levels and compare.

4. Error analysis on real data. The worst episodes for each strategy, what the
   market did in them, and how much of the loss was hedging error versus fees.
"""
from __future__ import annotations
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import engine
import experiments as ex
import policy as pol
import samplers as sp
import strategies as st

T = sp.STEPS / sp.TRADING_DAYS
SIG_REF = 0.15                     # close to the median of the training vol pool
EPS = 1e-3                         # |new - old| below this counts as "no trade"


def policy_map(p, S, t, sigma, h_prev, rv_rel=None):
    """Evaluate a Markov policy on arbitrary states (no path needed)."""
    S = torch.as_tensor(S, dtype=torch.float32)
    sig = torch.full_like(S, sigma)
    hp = torch.as_tensor(h_prev, dtype=torch.float32)
    rv = None if not p.use_rv else sig * (1.0 + (rv_rel or 0.0))
    x = pol.state_features(S, t, sp.STEPS, sig, hp, rv)
    with torch.no_grad():
        h, _ = p.decide(x, hp, None)
    return h.numpy()


def learned_band(p, S_val, t, sigma=SIG_REF, n=801):
    """No-trade interval [lo, hi] of a Markov policy at one (S, t, sigma)."""
    hp = np.linspace(0.0, 1.0, n)
    hn = policy_map(p, np.full(n, S_val), t, sigma, hp)
    stay = np.abs(hn - hp) < EPS
    if not stay.any():
        # no exact no-trade set: report where the policy's target crosses h_prev
        k = int(np.argmin(np.abs(hn - hp)))
        return float(hp[k]), float(hp[k]), False
    idx = np.where(stay)[0]
    return float(hp[idx[0]]), float(hp[idx[-1]]), True


def bs_delta_at(S_val, t, sigma=SIG_REF):
    tau = T * (1 - t / sp.STEPS)
    _, d, g = engine.bs(torch.tensor([float(S_val)]), 100.0, tau, sigma)
    return float(d), float(g)


# ------------------------------------------------------------------ 1. zero cost

def zero_cost_control():
    out = {}
    test = ex.test_set("gbm")
    D = st.bs_delta(test.episode, 100.0, T, test.sigma_imp, test.steps)
    for job in ("gbm_mlp_c0_s0", "gbm_ntb_c0_s0"):
        if not os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
            continue
        p, _ = ex.load_policy(job)
        with torch.no_grad():
            H = pol.rollout(p, test).double()
        diff = (H - D).abs()
        tot_n = ex.score(H, test, 0)[0]
        tot_d = ex.score(D, test, 0)[0]
        out[job] = {"mean_abs_dev_from_delta": float(diff.mean()),
                    "p99_abs_dev": float(torch.quantile(diff.flatten()[:2_000_000], 0.99)),
                    "max_abs_dev": float(diff.max()),
                    "cvar_policy": float(engine.cvar(tot_n)),
                    "cvar_delta": float(engine.cvar(tot_d)),
                    "corr_with_delta": float(np.corrcoef(H.flatten().numpy(),
                                                         D.flatten().numpy())[0, 1])}
        # curves for the site: policy vs delta across spot, three times to expiry
        Sg = np.linspace(85, 115, 121)
        curves = {}
        for t in (0, 15, 25):
            dd = [bs_delta_at(s, t)[0] for s in Sg]
            hp = np.array(dd)                     # start on delta, zero cost ignores it
            curves[str(t)] = {"S": Sg.tolist(), "delta": dd,
                              "policy": policy_map(p, Sg, t, SIG_REF, hp).tolist()}
        out[job]["curves"] = curves
    return out


# ------------------------------------------------------------------ 2 & 3. bands

def response(p, S_val, t, sigma=SIG_REF, n=201):
    """New holding as a function of the current holding, at one (S, t, sigma)."""
    hp = np.linspace(0.0, 1.0, n)
    return hp, policy_map(p, np.full(n, S_val), t, sigma, hp)


def inertia(p, S_val, t, sigma=SIG_REF):
    """How much of its current position a policy keeps: d h_new / d h_prev.

    Measured at the policy's own resting point, where h_new = h_prev. Black-Scholes
    delta has inertia 0 (it trades to delta whatever it holds). A hard no-trade
    band has inertia 1 inside the band and 0 outside. A smooth network that is
    becoming cost-aware without forming a hard band shows inertia strictly
    between 0 and 1, rising with cost: partial adjustment toward its target.
    """
    hp, hn = response(p, S_val, t, sigma, 801)
    k = int(np.argmin(np.abs(hn - hp)))                  # resting point
    lo, hi = max(k - 40, 0), min(k + 40, len(hp) - 1)    # +-0.05 around it
    slope = float(np.polyfit(hp[lo:hi + 1], hn[lo:hi + 1], 1)[0])
    return {"rest": float(hp[k]), "slope": slope}


def band_maps():
    tuned = ex.load("tuned_rules.json")["gbm"]
    Sg = np.linspace(88, 112, 49)
    times = (0, 5, 10, 15, 20, 25, 28)          # 30, 25, 20, 15, 10, 5 and 2 days left
    maps, scaling, inert, curves = {}, {}, {}, {}
    for c in ex.COSTS:
        maps[str(c)], curves[str(c)] = {}, {}
        for kind in ("mlp", "ntb"):
            job = f"gbm_{kind}_c{c}_s0"
            if not os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
                continue
            p, _ = ex.load_policy(job)
            per_t = {}
            for t in times:
                lo, hi, exact, dl, wwl, wwh, bl, bh, rest = [], [], [], [], [], [], [], [], []
                for s in Sg:
                    a, b, e = learned_band(p, s, t)
                    d, g = bs_delta_at(s, t)
                    k = c * 1e-4
                    H = 0.0 if k == 0 else min((1.5 * k * s * g * g /
                                                tuned[str(c)]["ww_gamma"]) ** (1 / 3), 1.0)
                    w = tuned[str(c)]["band_w"]
                    lo.append(a); hi.append(b); exact.append(e); dl.append(d)
                    wwl.append(max(d - H, 0.0)); wwh.append(min(d + H, 1.0))
                    bl.append(max(d - w, 0.0)); bh.append(min(d + w, 1.0))
                    rest.append(inertia(p, s, t)["rest"])
                per_t[str(t)] = {"S": Sg.tolist(), "lo": lo, "hi": hi, "exact": exact,
                                 "delta": dl, "ww_lo": wwl, "ww_hi": wwh,
                                 "band_lo": bl, "band_hi": bh, "rest": rest}
            maps[str(c)][kind] = per_t
            hp, hn = response(p, 100.0, 15)
            curves[str(c)][kind] = {"h_prev": hp.tolist(), "h_new": hn.tolist()}
            if kind == "mlp":
                inert[str(c)] = {str(s): inertia(p, s, 15) for s in (96.0, 100.0, 104.0)}
        # the rules' response curves at the same state, for comparison
        d, g = bs_delta_at(100.0, 15)
        hp = np.linspace(0, 1, 201)
        k = c * 1e-4
        H = 0.0 if k == 0 else min((1.5 * k * 100.0 * g * g / tuned[str(c)]["ww_gamma"]) ** (1 / 3), 1.0)
        ww = np.where(hp < d - H, d - H, np.where(hp > d + H, d + H, hp))
        w = tuned[str(c)]["band_w"]
        band = np.where(np.abs(hp - d) > w, d, hp)
        curves[str(c)]["delta"] = {"h_prev": hp.tolist(), "h_new": [d] * len(hp)}
        curves[str(c)]["ww"] = {"h_prev": hp.tolist(), "h_new": ww.tolist()}
        curves[str(c)]["band"] = {"h_prev": hp.tolist(), "h_new": band.tolist()}

    # the cube-root test on the band network, whose band is explicit but whose
    # width is learned with no formula given
    pts = []
    for c in ex.COSTS:
        if c == 0 or "ntb" not in maps[str(c)]:
            continue
        p, _ = ex.load_policy(f"gbm_ntb_c{c}_s0")
        hw = []
        for s in np.linspace(98, 102, 9):
            S_ = torch.tensor([s], dtype=torch.float32)
            x = pol.state_features(S_, 15, sp.STEPS, torch.full_like(S_, SIG_REF), torch.zeros(1))
            with torch.no_grad():
                l, u = p.bounds(x[:, [0, 1, 2]])
            hw.append(0.5 * float(u - l))
        pts.append((c, float(np.mean(hw))))
    if len(pts) >= 3:
        x = np.log([q[0] for q in pts]); y = np.log([max(q[1], 1e-6) for q in pts])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        se = math.sqrt(((y - yhat) ** 2).sum() / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum())
        scaling["ntb"] = {"points": pts, "exponent": float(coef[0]), "exponent_se": float(se),
                          "r2": float(r2), "theory_exponent": 1 / 3}
    return maps, scaling, inert, curves


def effective_vol(p, t, sigma=SIG_REF):
    """The volatility at which Black-Scholes delta best matches where a policy rests.

    A policy that hedges 'as if volatility were higher' has a flatter holding
    curve across index levels than delta at the true volatility. We find the
    single sigma_eff minimising the squared gap between the policy's resting
    holdings and BS delta(sigma_eff) over index levels 90 to 110.
    """
    from scipy.optimize import minimize_scalar
    Sg = np.linspace(90, 110, 41)
    rest = np.array([inertia(p, s, t, sigma)["rest"] for s in Sg])
    tau = T * (1 - t / sp.STEPS)

    def gap(s_eff):
        _, d, _ = engine.bs(torch.tensor(Sg, dtype=torch.float64), 100.0, tau, s_eff)
        return float(((d.numpy() - rest) ** 2).mean())

    r = minimize_scalar(gap, bounds=(0.03, 0.8), method="bounded")
    return {"sigma_eff": float(r.x), "ratio": float(r.x / sigma), "rmse": float(np.sqrt(r.fun)),
            "rmse_at_true_sigma": float(np.sqrt(gap(sigma)))}


def effective_vols():
    out = {}
    for c in ex.COSTS:
        job = f"gbm_mlp_c{c}_s0"
        if os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
            p, _ = ex.load_policy(job)
            out[str(c)] = {str(t): effective_vol(p, t) for t in (0, 10, 20)}
    return out


# ------------------------------------------------------------------ 4. errors

def realised_vol_episode(ep):
    r = torch.log(ep.episode[:, 1:] / ep.episode[:, :-1])
    return (r.std(dim=1, unbiased=True) * math.sqrt(sp.TRADING_DAYS)).numpy()


def error_analysis():
    tuned = ex.load("tuned_rules.json")
    J = ex.jobs()
    out = {}
    for idx in ex.INDICES:
        ep = ex.real_set(idx, ex.SPLIT)
        learned = [f"gbm_mlp_c{ex.HEADLINE}_s0", f"gbm_ntb_c{ex.HEADLINE}_s0",
                   f"hes_{idx}_gru_c{ex.HEADLINE}_s0", f"hes_{idx}_mlprv_c{ex.HEADLINE}_s0"]
        learned = [j for j in learned if os.path.exists(os.path.join(ex.MODELS, j + ".pt"))]
        rows, extra = ex.evaluate_set(ep, ex.HEADLINE, tuned["gbm"][str(ex.HEADLINE)],
                                      learned, keep_paths=True)
        rv = realised_vol_episode(ep)
        ratio = rv / ep.sigma_imp.numpy()
        prem = pol.premium_bs(ep).numpy()
        payoff = torch.clamp(ep.episode[:, -1] - 100.0, min=0).numpy()
        res = {"n": int(ep.S.shape[0]), "vol_ratio": ratio.tolist(),
               "sigma_imp": ep.sigma_imp.numpy().tolist(), "rv": rv.tolist(),
               "dates": ep.dates, "strategies": {}}
        for name, e in extra.items():
            tot = e["tot"].numpy(); fee = e["fee"].numpy()
            worst = np.argsort(tot)[:8]
            res["strategies"][name] = {
                "pnl": tot.tolist(),
                "fee": fee.tolist(),
                "corr_pnl_volratio": float(np.corrcoef(tot, ratio)[0, 1]),
                "worst": [{"date": ep.dates[i], "pnl": float(tot[i]),
                           "fee": float(fee[i]),
                           "hedge_error": float(tot[i] + fee[i]),
                           "sigma_imp": float(ep.sigma_imp[i]),
                           "rv": float(rv[i]), "vol_ratio": float(ratio[i]),
                           "move_pct": float(ep.episode[i, -1] - 100.0)}
                          for i in worst],
            }
        # a thinned set of full episodes for the site's episode explorer
        keep = sorted(set(range(0, ep.S.shape[0], 5)) |
                      {int(i) for e in extra.values() for i in np.argsort(e["tot"].numpy())[:8]})
        show = [k for k in ("bs_delta", "band", "ww") + tuple(learned) if k in extra]
        res["episodes"] = [{"date": ep.dates[i],
                            "S": [round(float(v), 3) for v in ep.episode[i]],
                            "sigma_imp": round(float(ep.sigma_imp[i]), 4),
                            "rv": round(float(rv[i]), 4),
                            "H": {k: [round(float(v), 4) for v in extra[k]["H"][i]] for k in show},
                            "pnl": {k: round(float(extra[k]["tot"][i]), 4) for k in show}}
                           for i in keep]
        out[idx] = res
        print("error analysis", idx, "done")
    return out


def regime_buckets(cost=None):
    """Where does each strategy win or lose? Test paths split by volatility surprise.

    Interpretation only: the models are fixed, nothing is selected here. Paths
    are sorted into five equal-sized groups by realised over quoted volatility,
    and each strategy's mean P&L, mean fees and CVaR are computed inside each
    group. A strategy that wins only on quiet paths is saving fees; one that
    wins on violent paths is protecting the tail.
    """
    cost = ex.HEADLINE if cost is None else cost
    test = ex.test_set("gbm")
    tuned = ex.load("tuned_rules.json")["gbm"][str(cost)]
    ratio = realised_vol_episode(test) / test.sigma_imp.numpy()
    edges = np.quantile(ratio, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    group = np.clip(np.searchsorted(edges, ratio, side="right") - 1, 0, 4)
    res = {"edges": edges.tolist(), "strategies": {}}
    hold = {k: ex.rule_holdings(k, test, cost, tuned) for k in ("bs_delta", "ww")}
    for job in (f"gbm_mlp_c{cost}_s0", f"gbm_ntb_c{cost}_s0"):
        if os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
            with torch.no_grad():
                hold[job] = pol.rollout(ex.load_policy(job)[0], test).double()
    for k, H in hold.items():
        tot, fee, _, _ = ex.score(H, test, cost)
        tot, fee = tot.numpy(), fee.numpy()
        rows = []
        for g in range(5):
            m = group == g
            x = torch.from_numpy(tot[m])
            rows.append({"mean": float(tot[m].mean()), "fee": float(fee[m].mean()),
                         "cvar95": float(engine.cvar(x)), "n": int(m.sum())})
        res["strategies"][k] = rows
    return res


def extras():
    """Numbers quoted in the prose that are not test metrics.

    sim_check compares plain delta hedging in the Heston world with the real
    TRAINING-ERA windows. Hedging CVaR was never a calibration target, so a
    match here is a (modest, in-sample) validation of the simulator.
    """
    out = {"vol_pool_n": int(len(ex.vol_pool()))}
    test = ex.test_set("gbm")
    prem = pol.premium_bs(test)
    out["avg_premium_gbm_test"] = float(prem.mean())
    chk = {}
    daily = ex.load("heston_calibration.json")
    for idx in ex.INDICES:
        hv = ex.val_set("heston", idx)                       # horizon calibration
        hd = sp.heston(ex.N_VAL, np.random.default_rng(ex.SEED_VAL + 77), daily[idx])
        rp = ex.real_set(idx, "2000-01-01", "2018-11-15")    # training era only
        c = {}
        for nm, e in (("heston", hv), ("heston_daily", hd), ("real", rp)):
            tot = ex.score(ex.rule_holdings("bs_delta", e, ex.HEADLINE), e, ex.HEADLINE)[0]
            c[nm] = float(engine.cvar(tot))
            ratio = realised_vol_episode(e) / e.sigma_imp.numpy()
            c["p95_" + nm] = float(np.percentile(ratio, 95))
        chk[idx] = c
    out["sim_check"] = chk
    return out


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "14")))
    res = {"extras": extras()}
    print("extras done")
    res["zero_cost"] = zero_cost_control()
    print("zero-cost control done")
    res["band_maps"], res["band_scaling"], res["inertia"], res["response_curves"] = band_maps()
    print("band maps done:", {k: round(v["exponent"], 3) for k, v in res["band_scaling"].items()})
    print("MLP inertia at the money:", {c: round(v["100.0"]["slope"], 3) for c, v in res["inertia"].items()})
    res["errors"] = error_analysis()
    res["effective_vol"] = effective_vols()
    print("effective vol:", {c: round(v["10"]["ratio"], 3) for c, v in res["effective_vol"].items()})
    res["regimes"] = regime_buckets()
    print("regime buckets done")
    import rates
    pols = {}
    for kind in ("mlp", "ntb"):
        job = f"gbm_{kind}_c{ex.HEADLINE}_s0"
        if os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
            pols[job] = ex.load_policy(job)[0]
    res["rate_ablation"] = rates.ablation(pols, ex.load("tuned_rules.json")["gbm"][str(ex.HEADLINE)],
                                          cost=ex.HEADLINE)
    print("rate ablation: ranking unchanged =", res["rate_ablation"]["ranking_unchanged"])
    ex.save("analysis.json", res)
