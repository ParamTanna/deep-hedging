"""Collect everything in results/ into the files the website reads.

    python src/export.py

docs/data/site.json        every number quoted on the page
docs/data/band_maps.json   learned and analytical no-trade bands, for the explorer
docs/data/episodes_*.json  a thinned set of real episodes, for the episode explorer

The site never computes a result itself. It only displays what this script
copies out of results/, so the page cannot disagree with the experiments.
The real price paths shipped here are rescaled to 100 at each sale date, so
the site carries no raw index levels.
"""
from __future__ import annotations
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import experiments as ex

OUT = os.path.join(ex.ROOT, "docs", "data")
os.makedirs(OUT, exist_ok=True)


def r(x, nd=4):
    """Round floats recursively so the JSON stays small."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return x


def training_histories():
    """Validation curves, plus the budget and hardware each model actually got.

    The facts quoted on the site (iterations, batch, GPU) are read from the
    checkpoints rather than from constants, so they cannot drift apart.
    """
    out, meta = {}, {}
    for f in sorted(os.listdir(ex.MODELS)):
        if f.endswith(".pt"):
            ck = torch.load(os.path.join(ex.MODELS, f), weights_only=False)
            out[f[:-3]] = [{"iter": h["iter"], "val_cvar": h["val_cvar"]}
                           for h in ck["train"]["history"]]
            meta[f[:-3]] = {"iters": ck.get("iters"), "batch": ck.get("batch"),
                            "hardware": ck.get("hardware", "cpu"),
                            "secs": ck["train"]["train_secs"],
                            "best_val_cvar": ck["train"]["best_val_cvar"]}
    return out, meta


def derived(site):
    """Summary numbers quoted in the prose, computed here so they are traceable."""
    d = {}
    H = str(ex.HEADLINE)
    g = site["gbm"][H]
    base = g["bs_delta"]["cvar95"]
    for k in ("leland", "band", "ww", f"gbm_mlp_c{H}_s0", f"gbm_ntb_c{H}_s0",
              f"gbm_mlpq_c{H}_s0"):
        if k in g:
            key = k.replace(f"_c{H}_s0", "").replace("gbm_", "")
            d[f"gbm_gain_{key}"] = g[k]["cvar95"] - base
            d[f"gbm_gain_pct_{key}"] = (g[k]["cvar95"] - base) / abs(base)
            d[f"gbm_fee_cut_pct_{key}"] = 1 - g[k]["mean_cost"] / g["bs_delta"]["mean_cost"]
    if "ww" in g and f"gbm_mlp_c{H}_s0" in g:
        d["gbm_mlp_vs_ww"] = g[f"gbm_mlp_c{H}_s0"]["cvar95"] - g["ww"]["cvar95"]
        d["gbm_mlp_vs_ww_pct"] = d["gbm_mlp_vs_ww"] / abs(g["ww"]["cvar95"])
    if "ww" in g and f"gbm_ntb_c{H}_s0" in g:
        d["gbm_ntb_vs_ww"] = g[f"gbm_ntb_c{H}_s0"]["cvar95"] - g["ww"]["cvar95"]
        d["gbm_ntb_vs_ww_pct"] = d["gbm_ntb_vs_ww"] / abs(g["ww"]["cvar95"])
    z = site["gbm"].get("0", {})
    if "gbm_mlp_c0_s0" in z:
        d["zero_gain_mlp"] = z["gbm_mlp_c0_s0"]["cvar95"] - z["bs_delta"]["cvar95"]
        if "gbm_gain_mlp" in d:
            d["cost_part_of_gain_mlp"] = d["gbm_gain_mlp"] - d["zero_gain_mlp"]
    # seed spread at the headline cost: how much a different seed moves the test CVaR
    for kind in ("mlp", "ntb"):
        v = [g[f"gbm_{kind}_c{H}_s{s}"]["cvar95"] for s in (0, 1, 2) if f"gbm_{kind}_c{H}_s{s}" in g]
        if len(v) == 3:
            d[f"seed_spread_{kind}"] = max(v) - min(v)
    # the network against the best rule, in simulation
    con = site.get("contrasts") or {}
    for c in (H, "50"):
        k = f"gbm_{c}"
        if k in con:
            best = con[k]["best_rule_cvar"]
            for kind in ("mlp", "ntb"):
                net = site["gbm"][str(c)][f"gbm_{kind}_c{c}_s0"]["cvar95"]
                d[f"gbm{c}_{kind}_vs_best_rule"] = net - best
                d[f"gbm{c}_{kind}_vs_best_rule_pct"] = (net - best) / abs(best)
    # real data from 2019 on
    for idx in ex.INDICES:
        R = site["real"][idx]["test_2019_on|gbm_tuned"]
        delta = R["bs_delta"]["cvar95"]
        nets = {k: v["cvar95"] for k, v in R.items() if k.startswith("hes_")}
        best_net = max(nets, key=nets.get)
        e = {"delta": delta, "best_net": best_net, "best_net_cvar": nets[best_net],
             "hes_gain_pct_min": min((v - delta) / abs(delta) for v in nets.values()),
             "hes_gain_pct_max": max((v - delta) / abs(delta) for v in nets.values()),
             "gbm_mlp_gain_pct": (R[f"gbm_mlp_c{H}_s0"]["cvar95"] - delta) / abs(delta)}
        c = con.get(f"real_{idx}")
        if c:
            e["best_rule"] = c["best_rule"]
            e["best_rule_cvar"] = c["best_rule_cvar"]
            e["best_net_vs_rule_pct"] = (nets[best_net] - c["best_rule_cvar"]) / abs(c["best_rule_cvar"])
            e["n_nets"] = len(c["minus_best_rule_ci"])
            e["n_sig_vs_rule"] = sum(ci[0] > 0 for ci in c["minus_best_rule_ci"].values())
        d[f"real_{idx}"] = e
    hes = [d[f"real_{i}"][k] for i in ex.INDICES for k in ("hes_gain_pct_min", "hes_gain_pct_max")]
    d["real_hes_gain_min_all"], d["real_hes_gain_max_all"] = min(hes), max(hes)
    # the zero-cost MLP's mean on validation, at the checkpoint selection picked,
    # in standard errors: the size of the selection bias the test set removes
    tr = site.get("training", {}).get("gbm_mlp_c0_s0")
    meta = site.get("training_meta", {}).get("gbm_mlp_c0_s0")
    if tr and z:
        import torch
        ck = torch.load(os.path.join(ex.MODELS, "gbm_mlp_c0_s0.pt"), weights_only=False)
        best = max(ck["train"]["history"], key=lambda r: r["val_cvar"])
        sd = z["gbm_mlp_c0_s0"]["std"]
        d["zero_val_mean_mlp"] = best["val_mean"]
        d["zero_val_mean_z_mlp"] = best["val_mean"] / (sd / np.sqrt(ex.N_VAL))
        d["n_checkpoints"] = len(ck["train"]["history"])
        d["zero_test_mean_z_mlp"] = z["gbm_mlp_c0_s0"]["mean"] / (sd / np.sqrt(z["gbm_mlp_c0_s0"]["n_paths"]))
    ex_ = site.get("explain_mlp")
    if ex_:
        d["explain_c0"] = ex_["c0"]
        d["explain_share_0"] = ex_["by_cost"]["0"]["share_explained"]
        d["explain_share_10"] = ex_["by_cost"][H]["share_explained"]
        d["explain_share_50"] = ex_["by_cost"]["50"]["share_explained"]
    br = site.get("band_regime")
    if br:
        d["ntb_halfwidth_min"] = site["band_scaling"]["ntb"]["points"][0][1]
        d["ntb_halfwidth_max"] = site["band_scaling"]["ntb"]["points"][-1][1]
    ce = site.get("calib_era")
    if ce and "jump_window" in ce:
        j = ce["jump_window"]
        d["jump_pct"] = math.exp(ce["largest_move"]["log_return"]) - 1
        d["jump_gap"] = j["delta_held"] - j["net_held"]
        d["jump_cost"] = (j["delta_held"] - j["net_held"]) * j["index_move"]
        net = f"hes_{ex.INDICES[0]}_mlprv_c{H}_s0"
        yrs = ce["worst_years"][net]
        d["calib_net_share_0809"] = (yrs.get("2008", 0) + yrs.get("2009", 0)) / sum(yrs.values())
        yd = ce["worst_years"]["bs_delta"]
        d["calib_delta_share_0809"] = (yd.get("2008", 0) + yd.get("2009", 0)) / sum(yd.values())

    # the sim-to-real ladder: one strategy's CVaR in every world, at the headline cost
    lad = {}
    for idx in ex.INDICES:
        real = site["real"][idx]
        worlds = {"gbm_sim": site["gbm"][H], "heston_sim": site["heston"][idx],
                  "real_calib": real.get("calib_era_pre2019|gbm_tuned", {}),
                  "real_test": real.get("test_2019_on|gbm_tuned", {})}
        rows = {}
        for k in worlds["real_test"]:
            rows[k] = {w: (v[k]["cvar95"] if k in v else None) for w, v in worlds.items()}
        lad[idx] = rows
    d["sim2real"] = lad
    # martingale check: at zero cost every causal strategy has E[P&L] = 0, so
    # the test-set mean of each must be statistically indistinguishable from 0
    if z:
        d["zero_cost_mean_z"] = {k: r["mean"] / (r["std"] / np.sqrt(r["n_paths"]))
                                 for k, r in z.items()}
        d["zero_cost_max_abs_z"] = max(abs(v) for v in d["zero_cost_mean_z"].values())
    # how often each strategy chooses not to trade (a change of holding below 0.1%)
    d["no_trade_share"] = {c: {k: 1 - r["mean_trades"] / 30 for k, r in rows.items()}
                           for c, rows in site["gbm"].items()}
    # seed spread at the headline cost
    for kind in ("mlp", "ntb"):
        v = [g[j]["cvar95"] for j in g if j.startswith(f"gbm_{kind}_c{H}_s")]
        if len(v) > 1:
            d[f"seed_mean_{kind}"] = float(np.mean(v))
            d[f"seed_sd_{kind}"] = float(np.std(v, ddof=1))
            d[f"seed_n_{kind}"] = len(v)
    return d


def model_weights():
    """Trained GBM policies as plain lists, for the in-browser laboratory.

    9 significant digits is the full precision of the float32 weights the
    networks were trained with, so nothing is lost. tests/check_lab.js checks
    that the shipped file reproduces the Python P&L.
    """
    g9 = lambda v: float(f"{v:.9g}")
    out = {}
    for c in ex.COSTS:
        for kind in ("mlp", "ntb"):
            job = f"gbm_{kind}_c{c}_s0"
            if not os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
                continue
            p = ex.load_policy(job)[0]
            lin = [m for m in p.net if isinstance(m, torch.nn.Linear)]
            out[f"{kind}_{c}"] = {"kind": kind, "layers": [
                {"W": [[g9(v) for v in row] for row in m.weight.detach().tolist()],
                 "b": [g9(v) for v in m.bias.detach().tolist()]} for m in lin]}
    tuned = {str(c): {k: v for k, v in r.items() if k in ("band_w", "ww_gamma")}
             for c, r in ex.load("tuned_rules.json")["gbm"].items()}
    return {"models": out, "tuned": tuned}


def main():
    hist, tmeta = training_histories()

    def one(key):
        vals = {v[key] for v in tmeta.values()}
        assert len(vals) == 1, f"models disagree on {key}: {vals}"
        return vals.pop()

    m = ex.load("metrics.json")
    a = ex.load("analysis.json")
    site = {
        "headline": ex.HEADLINE, "costs": ex.COSTS,
        "n_val": ex.N_VAL, "n_test": ex.N_TEST, "split": ex.SPLIT,
        "data_report": ex.load("data_report.json"),
        "sim_validation": ex.load("simulator_validation.json"),
        "calibration": {"daily": ex.load("heston_calibration.json"),
                        "horizon": ex.load("heston_horizon_calibration.json")},
        "tuned": ex.load("tuned_rules.json"),
        "gbm": m["gbm"], "heston": m["heston"], "real": m["real"],
        "zero_cost": a["zero_cost"], "band_scaling": a["band_scaling"],
        "regimes": a.get("regimes"), "rate_ablation": a.get("rate_ablation"),
        "inertia": a.get("inertia"),
        "explain_mlp": a.get("explain_mlp"), "band_regime": a.get("band_regime"),
        "contrasts": a.get("contrasts"), "calib_era": a.get("calib_era"),
        "effective_vol": a.get("effective_vol"),
        "training": hist, "training_meta": tmeta,
        "iters": one("iters"), "batch": one("batch"), "hardware": one("hardware"),
    }
    site["vol_pool_n"] = a["extras"]["vol_pool_n"]
    site["sim_check"] = a["extras"]["sim_check"]
    site["avg_premium_gbm_test"] = a["extras"]["avg_premium_gbm_test"]
    t = site["tuned"]["gbm"][str(ex.HEADLINE)]
    site["tuned_val"] = {"ww": t["ww_val_cvar"], "band": t["band_val_cvar"]}
    errs = {}
    for idx, e in a["errors"].items():
        errs[idx] = {k: e[k] for k in ("n", "vol_ratio", "sigma_imp", "rv", "dates")}
        errs[idx]["strategies"] = {k: {"pnl": v["pnl"], "corr_pnl_volratio": v["corr_pnl_volratio"],
                                       "worst": v["worst"]}
                                   for k, v in e["strategies"].items()}
        with open(os.path.join(OUT, f"episodes_{idx}.json"), "w") as f:
            json.dump(e["episodes"], f, separators=(",", ":"))
    site["errors"] = errs
    site["derived"] = derived(site)
    un = site["gbm"][str(ex.HEADLINE)]["unhedged"]["cvar95"]
    site["derived"]["unhedged_cvar_abs"] = abs(un)
    site["derived"]["unhedged_cvar_over_premium"] = abs(un) / site["avg_premium_gbm_test"]
    with open(os.path.join(OUT, "site.json"), "w") as f:
        json.dump(r(site), f, separators=(",", ":"))
    with open(os.path.join(OUT, "band_maps.json"), "w") as f:
        json.dump(r({"maps": a["band_maps"], "curves": a.get("response_curves", {})}), f,
                  separators=(",", ":"))
    with open(os.path.join(OUT, "models.json"), "w") as f:
        json.dump(model_weights(), f, separators=(",", ":"))
    for fn in os.listdir(OUT):
        print(f"  {fn:24s} {os.path.getsize(os.path.join(OUT, fn)) / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
