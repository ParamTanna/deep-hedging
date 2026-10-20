"""The full study, stage by stage. Every stage writes JSON into results/.

    python src/experiments.py prepare      data checks, calibration, simulator validation
    python src/experiments.py tune         tune the band width and the WW risk aversion
    python src/experiments.py train JOB    train one learned policy
    python src/experiments.py train-all    every job, a few at a time
    python src/experiments.py evaluate     score everything on the test sets

Nothing here reads the test sets until `evaluate`, and `evaluate` changes no
parameter. That ordering is the whole of the evaluation design, so it is
enforced by the code layout rather than left to discipline.

Decisions fixed before any test number existed (see README for the reasons):
  * option: at-the-money call, 30 trading days, daily rebalancing
  * metric: CVaR at 95% of terminal P&L net of all fees, per 100 notional
  * headline cost 10 bps, sweep over 0, 5, 10, 20, 50 bps
  * validation 20,000 episodes (seed 1001), test 100,000 episodes (seed 2002)
  * real data: everything estimated up to 2018-12-31; test = inception 2019 on
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch

import data
import engine
import policy as pol
import samplers as sp
import simulators as sim
import strategies as st

RES = os.path.join(ROOT, "results")
MODELS = os.path.join(RES, "models")
os.makedirs(MODELS, exist_ok=True)

COSTS = [0, 5, 10, 20, 50]
HEADLINE = 10
N_VAL, N_TEST = 20000, 100000
SEED_VAL, SEED_TEST = 1001, 2002
SPLIT = "2019-01-01"
INDICES = ("NIFTY50", "SP500")
T = sp.STEPS / sp.TRADING_DAYS
BAND_GRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]
WW_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
ITERS = int(os.environ.get("ITERS", "1000"))
TRADE_TOL = 1e-3          # a "trade" is a change in holding above 0.1% of notional


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ============================================================ worlds

def clean_df(name):
    df, rep = data.clean(data.fetch(name))
    return df, rep


def vol_pool():
    return np.load(os.path.join(ROOT, "data", "processed", "vol_pool_pre2019.npy"))


def heston_params(index):
    """The Heston world used for training and testing is the HORIZON calibration.

    heston_calibration.json (daily moments) is kept only as the documented
    failure: it matched daily kurtosis and got the 30-day volatility surprise
    badly wrong. See simulators.calibrate_heston_horizon.
    """
    return load("heston_horizon_calibration.json")[index]


def episodes(world, n, seed, index=None):
    rng = np.random.default_rng(seed)
    if world == "gbm":
        return sp.gbm(n, rng, sigma_pool=vol_pool())
    if world == "heston":
        return sp.heston(n, rng, heston_params(index))
    raise ValueError(world)


def sampler_for(world, index=None):
    if world == "gbm":
        pool = vol_pool()
        return lambda n, rng: sp.gbm(n, rng, sigma_pool=pool)
    p = heston_params(index)
    return lambda n, rng: sp.heston(n, rng, p)


def val_set(world, index=None):
    off = 0 if index is None else 10 * (1 + INDICES.index(index))
    return episodes(world, N_VAL, SEED_VAL + off, index)


def test_set(world, index=None):
    off = 0 if index is None else 10 * (1 + INDICES.index(index))
    return episodes(world, N_TEST, SEED_TEST + off, index)


def real_set(index, start, end=None):
    df, _ = clean_df(index)
    return sp.real(df, start, end)


# ============================================================ rule strategies

def rule_holdings(name, ep, cost, tuned=None):
    S = ep.episode
    sig = ep.sigma_imp
    n = ep.steps
    if name == "unhedged":
        return st.unhedged(S, n)
    if name == "bs_delta":
        return st.bs_delta(S, 100.0, T, sig, n)
    if name == "leland":
        sl = torch.tensor([st.leland_sigma(float(s), cost, T, n) for s in sig],
                          dtype=S.dtype)
        return st.bs_delta(S, 100.0, T, sl, n)
    if name == "band":
        return st.band(S, 100.0, T, sig, n, tuned["band_w"])[0]
    if name == "ww":
        return st.whalley_wilmott(S, 100.0, T, sig, n, cost, tuned["ww_gamma"])[0]
    raise ValueError(name)


def score(H, ep, cost):
    tot, fee, turn = pol.episode_pnl(H, ep, cost, torch.float64)
    trades = ((H.double() - torch.cat([torch.zeros(H.shape[0], 1, dtype=torch.float64),
                                       H.double()[:, :-1]], 1)).abs() > TRADE_TOL).sum(1)
    return tot, fee, turn, trades.double()


def tune_rules(val_ep, cost):
    """Pick the band width and the WW gamma that maximise validation CVaR95."""
    out = {}
    best = None
    for w in BAND_GRID:
        tot = score(st.band(val_ep.episode, 100.0, T, val_ep.sigma_imp, val_ep.steps, w)[0],
                    val_ep, cost)[0]
        c = float(engine.cvar(tot))
        if best is None or c > best[0]:
            best = (c, w)
    out["band_w"], out["band_val_cvar"] = best[1], best[0]
    best = None
    for g in WW_GRID:
        H = st.whalley_wilmott(val_ep.episode, 100.0, T, val_ep.sigma_imp, val_ep.steps,
                               cost, g)[0]
        c = float(engine.cvar(score(H, val_ep, cost)[0]))
        if best is None or c > best[0]:
            best = (c, g)
    out["ww_gamma"], out["ww_val_cvar"] = best[1], best[0]
    edge = {"band_at_grid_edge": out["band_w"] in (BAND_GRID[0], BAND_GRID[-1]),
            "ww_at_grid_edge": out["ww_gamma"] in (WW_GRID[0], WW_GRID[-1])}
    out.update(edge)
    return out


# ============================================================ jobs

def jobs():
    J = {}
    for c in COSTS:
        J[f"gbm_mlp_c{c}_s0"] = dict(world="gbm", kind="mlp", cost=c, seed=0)
        J[f"gbm_ntb_c{c}_s0"] = dict(world="gbm", kind="ntb", cost=c, seed=0)
    for s in (1, 2):
        J[f"gbm_mlp_c{HEADLINE}_s{s}"] = dict(world="gbm", kind="mlp", cost=HEADLINE, seed=s)
        J[f"gbm_ntb_c{HEADLINE}_s{s}"] = dict(world="gbm", kind="ntb", cost=HEADLINE, seed=s)
    J[f"gbm_mlpq_c{HEADLINE}_s0"] = dict(world="gbm", kind="mlp", cost=HEADLINE, seed=0,
                                         objective="quadratic")
    for w in (8, 16, 32, 128):
        J[f"gbm_mlp_w{w}_c{HEADLINE}_s0"] = dict(world="gbm", kind="mlp", cost=HEADLINE,
                                                 seed=0, width=w)
    for idx in INDICES:
        J[f"hes_{idx}_mlprv_c{HEADLINE}_s0"] = dict(world="heston", index=idx, kind="mlp",
                                                    use_rv=True, cost=HEADLINE, seed=0)
        J[f"hes_{idx}_ntbrv_c{HEADLINE}_s0"] = dict(world="heston", index=idx, kind="ntb",
                                                    use_rv=True, cost=HEADLINE, seed=0)
        J[f"hes_{idx}_gru_c{HEADLINE}_s0"] = dict(world="heston", index=idx, kind="gru",
                                                  cost=HEADLINE, seed=0)
    return J


def build(spec):
    return pol.make_policy(spec["kind"], width=spec.get("width", 64),
                           depth=spec.get("depth", 3), use_rv=spec.get("use_rv", False))


def load_policy(job):
    ck = torch.load(os.path.join(MODELS, job + ".pt"), weights_only=False)
    p = build(ck["spec"])
    p.load_state_dict(ck["state"])
    p.eval()
    return p, ck


def train_job(job):
    spec = jobs()[job]
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    world, idx = spec["world"], spec.get("index")
    val = val_set(world, idx)
    p = build(spec)
    batch = int(os.environ.get("BATCH", "8192")) if spec["kind"] != "gru" else 4096
    iters = ITERS if spec["kind"] != "gru" else max(ITERS // 2, 300)
    print(f"[{job}] start  iters={iters} batch={batch}", flush=True)
    res = pol.train(p, sampler_for(world, idx), spec["cost"], val,
                    objective=spec.get("objective", "cvar"), iters=iters, batch=batch,
                    seed=spec["seed"], eval_every=25,
                    log=lambda s: print(f"[{job}] {s}", flush=True))
    torch.save({"state": p.state_dict(), "spec": spec, "train": res},
               os.path.join(MODELS, job + ".pt"))
    print(f"[{job}] done   best val CVaR {res['best_val_cvar']:.4f}  "
          f"({res['train_secs']:.0f}s)", flush=True)


def train_all(workers=3, threads=4, only=None):
    order = only if only is not None else list(jobs())
    todo = [j for j in order if j in jobs()
            and not os.path.exists(os.path.join(MODELS, j + ".pt"))]
    print(f"{len(todo)} jobs to train, {workers} at a time, {threads} threads each")
    env = dict(os.environ, THREADS=str(threads), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    logdir = os.path.join(RES, "logs")
    os.makedirs(logdir, exist_ok=True)

    def run(j):
        with open(os.path.join(logdir, j + ".log"), "w") as f:
            r = subprocess.run([sys.executable, __file__, "train", j], stdout=f,
                               stderr=subprocess.STDOUT, env=env)
        print(f"  {j:32s} exit {r.returncode}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(run, todo))
    print("failures:", sum(c != 0 for c in codes))


# ============================================================ prepare

def stage_prepare():
    t0 = time.time()
    report = {}
    pre_r = {}
    for idx in INDICES:
        raw = data.fetch(idx)
        df, rep = data.clean(raw)
        r_all = data.log_returns(df)
        dates = df["date"].iloc[1:].reset_index(drop=True)
        pre = r_all.values[(dates < SPLIT).values]
        post = r_all.values[(dates >= SPLIT).values]
        pre_r[idx] = pre
        gaps = df["date"].diff().dt.days.dropna()
        rep.update({
            "adjclose_equals_close": bool(np.allclose(df["adjclose"], df["close"])),
            "zero_volume_days": int((df["volume"].fillna(0) == 0).sum()),
            "zero_change_days": int((r_all.abs() < 1e-12).sum()),
            "calendar_gaps_over_5_days": int((gaps > 5).sum()),
            "largest_gap_days": int(gaps.max()),
            "largest_gap_ends": str(df["date"][gaps.idxmax()].date()),
            "pre2019": data.describe_returns(pre),
            "post2019": data.describe_returns(post),
            "acf_r2_pre2019": data.acf(pre ** 2, 20).tolist(),
            "acf_r_pre2019": data.acf(pre, 20).tolist(),
            "chi2_crit_20df_5pct": 31.410,
        })
        report[idx] = rep
        print(f"{idx}: {rep['rows_kept']} rows, {rep['first_date']} to {rep['last_date']}")
    save("data_report.json", report)

    # calibrations, training era only
    cal = {}
    for idx in INDICES:
        cal[idx] = sim.calibrate_heston(pre_r[idx])
        cal[idx]["gbm_sigma"] = sim.calibrate_gbm(pre_r[idx])["sigma"]
        print(f"{idx}: heston (daily moments) kappa {cal[idx]['kappa']:.2f} "
              f"xi {cal[idx]['xi']:.2f} rho {cal[idx]['rho']:.3f}")
    save("heston_calibration.json", cal)
    hcal = {}
    for idx in INDICES:
        hcal[idx] = sim.calibrate_heston_horizon(pre_r[idx])
        print(f"{idx}: heston (horizon) kappa {hcal[idx]['kappa']:.2f} "
              f"xi {hcal[idx]['xi']:.2f} rho {hcal[idx]['rho']:.3f}")
    save("heston_horizon_calibration.json", hcal)

    # simulator validation: do simulated returns reproduce the stylised facts?
    val = {}
    rng = np.random.default_rng(7)
    for idx in INDICES:
        n = len(pre_r[idx])
        p = cal[idx]
        g = sim.gbm_paths(1.0, p["gbm_sigma"], n / 252, n, 24, rng)
        h = sim.heston_paths(1.0, p["theta"], p["kappa"], p["theta"], p["xi"], p["rho"],
                             n / 252, n, 24, rng)
        q = hcal[idx]
        h2 = sim.heston_paths(1.0, q["theta"], q["kappa"], q["theta"], q["xi"], q["rho"],
                              n / 252, n, 24, rng)
        out = {}
        for nm, P in (("gbm", g), ("heston_daily", h), ("heston_horizon", h2)):
            rr = np.diff(np.log(P), axis=1)
            m = np.mean([sim.return_moments(x) for x in rr], axis=0)
            a = np.mean([data.acf(x ** 2, 20) for x in rr], axis=0)
            hz = np.mean([sim.horizon_stats(x) for x in rr], axis=0)
            lr_all = []
            for x in rr[:8]:
                qv = sim._rolling_std(x, 60)
                ev = sim._rolling_std(x, 30)
                k = len(x) - 90 + 1
                lr_all.append(np.log(ev[60:60 + k] / qv[:k]))
            out[nm] = {"moments": m.tolist(), "acf_r2": a.tolist(), "horizon": hz.tolist(),
                       "log_ratio_sample": np.concatenate(lr_all)[::7].round(4).tolist()}
        x = pre_r[idx]
        qv, ev = sim._rolling_std(x, 60), sim._rolling_std(x, 30)
        k = len(x) - 90 + 1
        out["real_pre2019"] = {"moments": sim.return_moments(x).tolist(),
                               "acf_r2": data.acf(x ** 2, 20).tolist(),
                               "horizon": sim.horizon_stats(x).tolist(),
                               "log_ratio_sample": np.log(ev[60:60 + k] / qv[:k]).round(4).tolist()}
        out["moment_names"] = ["ann_vol", "excess_kurtosis", "skewness", "acf1_r2", "acf5_r2"]
        out["horizon_names"] = ["median_log_ratio", "sd_log_ratio", "p95_log_ratio",
                                "corr_return_log_ratio"]
        val[idx] = out
    save("simulator_validation.json", val)
    print(f"prepare done in {time.time() - t0:.0f}s")


# ============================================================ tune

def stage_tune():
    out = {"gbm": {}, "heston": {}}
    v = val_set("gbm")
    for c in COSTS:
        out["gbm"][str(c)] = tune_rules(v, c)
        print("gbm", c, out["gbm"][str(c)])
    for idx in INDICES:
        v = val_set("heston", idx)
        out["heston"][idx] = tune_rules(v, HEADLINE)
        print("heston", idx, out["heston"][idx])
    save("tuned_rules.json", out)


# ============================================================ evaluate

def bootstrap_cvar(tots: dict, n_boot=1000, block=None, seed=0, ref=None):
    """Percentile CIs for CVaR95, paired across strategies.

    Paired: in each resample every strategy is scored on the SAME episodes, so
    the CI of a difference reflects only the strategies, not which paths
    happened to be drawn. With block set, a moving-block bootstrap is used for
    overlapping real windows.
    """
    names = list(tots)
    N = next(iter(tots.values())).numel()
    rng = np.random.default_rng(seed)
    alpha_n = max(1, int(round(0.05 * N)))
    samples = {k: [] for k in names}
    diffs = {k: [] for k in names} if ref else None
    stack = torch.stack([tots[k] for k in names], 0)
    for _ in range(n_boot):
        if block:
            nb = math.ceil(N / block)
            starts = rng.integers(0, N - block + 1, nb)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:N]
        else:
            idx = rng.integers(0, N, N)
        x = stack[:, torch.from_numpy(idx)]
        cv = torch.topk(x, alpha_n, dim=1, largest=False).values.mean(1)
        for i, k in enumerate(names):
            samples[k].append(float(cv[i]))
        if ref:
            r = float(cv[names.index(ref)])
            for i, k in enumerate(names):
                diffs[k].append(float(cv[i]) - r)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in samples.items()}
    dci = None
    if ref:
        dci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
               for k, v in diffs.items()}
    return ci, dci


def evaluate_set(ep, cost, tuned, learned_jobs, ref="bs_delta", block=None,
                 keep_paths=False):
    tots, rows, extra = {}, {}, {}
    for name in ("unhedged", "bs_delta", "leland", "band", "ww"):
        H = rule_holdings(name, ep, cost, tuned)
        tot, fee, turn, tr = score(H, ep, cost)
        tots[name] = tot
        rows[name] = engine.metrics(tot, fee, turn, tr)
        if keep_paths:
            extra[name] = {"tot": tot, "fee": fee, "H": H}
    for job in learned_jobs:
        p, _ = load_policy(job)
        with torch.no_grad():
            H = pol.rollout(p, ep).double()
        tot, fee, turn, tr = score(H, ep, cost)
        tots[job] = tot
        rows[job] = engine.metrics(tot, fee, turn, tr)
        if keep_paths:
            extra[job] = {"tot": tot, "fee": fee, "H": H}
    ci, dci = bootstrap_cvar(tots, n_boot=600 if block else 400, block=block, ref=ref)
    for k in rows:
        rows[k]["cvar95_ci"] = ci[k]
        rows[k]["cvar95_minus_" + ref + "_ci"] = dci[k]
    return (rows, extra) if keep_paths else rows


def stage_evaluate():
    tuned = load("tuned_rules.json")
    J = jobs()
    out = {"gbm": {}, "heston": {}, "real": {}}

    test = test_set("gbm")
    for c in COSTS:
        learned = [j for j in (f"gbm_mlp_c{c}_s0", f"gbm_ntb_c{c}_s0")
                   if os.path.exists(os.path.join(MODELS, j + ".pt"))]
        if c == HEADLINE:
            learned += [j for j in J if J[j]["world"] == "gbm" and J[j]["cost"] == c
                        and j not in learned and os.path.exists(os.path.join(MODELS, j + ".pt"))]
        out["gbm"][str(c)] = evaluate_set(test, c, tuned["gbm"][str(c)], learned)
        print("gbm", c, "done")

    for idx in INDICES:
        test = test_set("heston", idx)
        learned = [f"gbm_mlp_c{HEADLINE}_s0", f"gbm_ntb_c{HEADLINE}_s0"] + \
                  [j for j in J if J[j].get("index") == idx]
        learned = [j for j in learned if os.path.exists(os.path.join(MODELS, j + ".pt"))]
        out["heston"][idx] = evaluate_set(test, HEADLINE, tuned["heston"][idx], learned)
        print("heston", idx, "done")

    for idx in INDICES:
        out["real"][idx] = {}
        learned = [f"gbm_mlp_c{HEADLINE}_s0", f"gbm_ntb_c{HEADLINE}_s0"] + \
                  [j for j in J if J[j].get("index") == idx]
        learned = [j for j in learned if os.path.exists(os.path.join(MODELS, j + ".pt"))]
        for era, (a, b) in {"test_2019_on": (SPLIT, None),
                            "calib_era_2009_2018": ("2009-01-01", "2018-12-31")}.items():
            ep = real_set(idx, a, b)
            for fam, tun in (("gbm_tuned", tuned["gbm"][str(HEADLINE)]),
                             ("heston_tuned", tuned["heston"][idx])):
                rows = evaluate_set(ep, HEADLINE, tun, learned, block=30)
                out["real"][idx][f"{era}|{fam}"] = rows
            out["real"][idx][f"{era}|n_windows"] = int(ep.S.shape[0])
            out["real"][idx][f"{era}|first"] = ep.dates[0]
            out["real"][idx][f"{era}|last"] = ep.dates[-1]
            print("real", idx, era, ep.S.shape[0], "windows")
    save("metrics.json", out)


# ============================================================ main

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    torch.set_num_threads(int(os.environ.get("THREADS", "14")))
    if cmd == "prepare":
        stage_prepare()
    elif cmd == "tune":
        stage_tune()
    elif cmd == "train":
        train_job(sys.argv[2])
    elif cmd == "train-all":
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        train_all(int(os.environ.get("WORKERS", "3")), int(os.environ.get("THREADS", "4")),
                  only)
    elif cmd == "evaluate":
        stage_evaluate()
    else:
        print(__doc__)
