"""Static figures for the README and the notebook.

    python src/figures.py

Reads the same JSON files as the website, writes PNGs to results/figures/.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as ex

FIG = os.path.join(ex.RES, "figures")
os.makedirs(FIG, exist_ok=True)

C = {"unhedged": "#9a9a92", "bs_delta": "#4a4a46", "leland": "#8c6bb1", "band": "#d39a2f",
     "ww": "#2e6f8e", "mlp": "#b4462e", "ntb": "#2f7d4f", "gru": "#c2185b",
     "mlprv": "#e07b39", "ntbrv": "#3aa38a", "mlpq": "#a0522d"}
NAME = {"unhedged": "Unhedged", "bs_delta": "Black-Scholes delta", "leland": "Leland delta",
        "band": "Constant band", "ww": "Whalley-Wilmott band", "mlp": "Deep hedge (MLP)",
        "ntb": "No-trade band net", "gru": "Recurrent (GRU)", "mlprv": "MLP + vol feature",
        "ntbrv": "Band net + vol feature", "mlpq": "MLP, quadratic loss"}
IDXN = {"NIFTY50": "NIFTY 50", "SP500": "S&P 500"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 11.5,
})


def kind(job):
    for k in ("mlpq", "mlprv", "ntbrv", "gru", "mlp", "ntb"):
        if f"_{k}_" in job or job.startswith(f"gbm_{k}_"):
            return k
    return job


def fig_surprise(v):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    bins = np.arange(-1.2, 1.6, 0.05)
    for ax, idx in zip(axs, ex.INDICES):
        s = v[idx]
        ax.hist(s["real_pre2019"]["log_ratio_sample"], bins=bins, density=True, color="#bbb",
                label="Real, pre-2019")
        for k, lab, col, lw in (("gbm", "GBM", C["unhedged"], 1.6),
                                ("heston_daily", "Heston, daily-moment fit", C["band"], 1.8),
                                ("heston_horizon", "Heston, horizon fit", C["ww"], 2.4)):
            h, e = np.histogram(s[k]["log_ratio_sample"], bins=bins, density=True)
            ax.step(e[:-1], h, where="post", color=col, lw=lw, label=lab)
        ax.axvline(0, color="#888", ls=":", lw=1)
        ax.set_title(IDXN[idx])
        ax.set_xlabel("log(realised vol in episode / quoted vol)")
    axs[0].set_ylabel("density")
    axs[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("How wrong was the volatility quote? Real data against three simulators",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_surprise.png"), bbox_inches="tight")
    plt.close(fig)


def fig_sweep(m):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    costs = ex.COSTS
    for k in ("bs_delta", "leland", "band", "ww", "mlp", "ntb"):
        ys, gs, lo, hi = [], [], [], []
        for c in costs:
            rows = m["gbm"][str(c)]
            jid = f"gbm_{k}_c{c}_s0" if k in ("mlp", "ntb") else k
            r = rows[jid]
            ys.append(r["cvar95"])
            gs.append(r["cvar95"] - rows["bs_delta"]["cvar95"])
            d = r["cvar95_minus_bs_delta_ci"]
            lo.append(gs[-1] - d[0]); hi.append(d[1] - gs[-1])
        lw = 2.6 if k in ("mlp", "ntb") else 1.6
        axs[0].plot(costs, ys, "o-", color=C[k], lw=lw, ms=4, label=NAME[k])
        axs[1].errorbar(costs, gs, yerr=[lo, hi], fmt="o-", color=C[k], lw=lw, ms=4, capsize=2)
    axs[0].set_xlabel("transaction cost (bps)"); axs[0].set_ylabel("CVaR95 of P&L (higher is better)")
    axs[0].set_title("Tail loss across the cost sweep (GBM test)")
    axs[1].axhline(0, color="#888", lw=1)
    axs[1].set_xlabel("transaction cost (bps)"); axs[1].set_ylabel("CVaR95 minus delta's")
    axs[1].set_title("Gain over Black-Scholes delta, 95% paired CI")
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sweep.png"), bbox_inches="tight")
    plt.close(fig)


def fig_bands(bm, curves):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, c in zip(axs[:2], ("10", "50")):
        S = np.array(bm[c]["ntb"]["15"]["S"])
        n, m = bm[c]["ntb"]["15"], bm[c]["mlp"]["15"]
        ax.fill_between(S, n["ww_lo"], n["ww_hi"], color=C["ww"], alpha=0.18, label="Whalley-Wilmott band")
        ax.plot(S, n["lo"], color=C["ntb"], lw=2.2, label="Band net, learned edges")
        ax.plot(S, n["hi"], color=C["ntb"], lw=2.2)
        ax.plot(S, m["rest"], color=C["mlp"], lw=1.8, ls="--", label="MLP, where it settles")
        ax.plot(S, n["delta"], color="#222", lw=1.1, ls=":", label="Black-Scholes delta")
        ax.set_title(f"{c} bps, 15 days to expiry"); ax.set_xlabel("index level (strike 100)")
    axs[0].set_ylabel("units of index held"); axs[0].legend(fontsize=8, loc="upper left")
    ax = axs[2]
    cv = curves["50"]
    for k, lab, col, ls in (("delta", "Black-Scholes delta", "#222", ":"), ("ww", "Whalley-Wilmott", C["ww"], "-"),
                            ("ntb", "Band net", C["ntb"], "-"), ("mlp", "MLP", C["mlp"], "--")):
        if k in cv:
            ax.plot(cv[k]["h_prev"], cv[k]["h_new"], color=col, ls=ls, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
    ax.set_xlabel("current holding"); ax.set_ylabel("new holding")
    ax.set_title("50 bps, at the money: response to current holding"); ax.legend(fontsize=8)
    fig.suptitle("What the networks learned about not trading", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_bands.png"), bbox_inches="tight")
    plt.close(fig)


def fig_scaling(sc, inert=None):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axs[0]
    if "ntb" in sc:
        x = np.array([p[0] for p in sc["ntb"]["points"]]); y = np.array([p[1] for p in sc["ntb"]["points"]])
        ax.loglog(x, y, "o", color=C["ntb"], ms=7,
                  label=f"band net: slope {sc['ntb']['exponent']:.2f} $\\pm$ {sc['ntb']['exponent_se']:.2f}")
        xs = np.array([x.min() * 0.85, x.max() * 1.15])
        a = np.log(y).mean() - sc["ntb"]["exponent"] * np.log(x).mean()
        ax.loglog(xs, np.exp(a) * xs ** sc["ntb"]["exponent"], color=C["ntb"], lw=1.2)
        a3 = np.log(y).mean() - np.log(x).mean() / 3
        ax.loglog(xs, np.exp(a3) * xs ** (1 / 3), ":", color="#555", lw=2, label="theory: slope 1/3")
    ax.set_xlabel("transaction cost (bps)"); ax.set_ylabel("learned half-width at the money")
    ax.set_title("The cube-root law"); ax.legend(fontsize=8.5)
    ax = axs[1]
    if inert:
        cs = sorted(int(c) for c in inert)
        ax.plot(cs, [inert[str(c)]["100.0"]["slope"] for c in cs], "o-", color=C["mlp"], lw=2)
    ax.set_xlabel("transaction cost (bps)"); ax.set_ylabel("d(new holding) / d(current holding)")
    ax.set_title("The MLP's inertia grows with cost"); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scaling.png"), bbox_inches="tight")
    plt.close(fig)


def fig_real(m):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharex=False)
    H = ex.HEADLINE
    for ax, idx in zip(axs, ex.INDICES):
        rows = m["real"][idx]["test_2019_on|gbm_tuned"]
        ids = [k for k in rows if k != "unhedged" and not k.endswith(("_s1", "_s2"))
               and "_w" not in k and "mlpq" not in k]
        ids.sort(key=lambda k: rows[k]["cvar95"])
        y = np.arange(len(ids))
        v = np.array([rows[k]["cvar95"] for k in ids])
        lo = v - np.array([rows[k]["cvar95_ci"][0] for k in ids])
        hi = np.array([rows[k]["cvar95_ci"][1] for k in ids]) - v
        cols = [C.get(kind(k), "#888") for k in ids]
        ax.barh(y, v, color=cols, xerr=[lo, hi], error_kw={"elinewidth": 1, "capsize": 2})
        labs = [NAME.get(kind(k), k) + (" (Heston)" if k.startswith("hes_") else "") for k in ids]
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
        ax.set_title(f"{IDXN[idx]}, options sold 2019 onward, {H} bps")
        ax.set_xlabel("CVaR95 of P&L per 100 notional (right is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_real.png"), bbox_inches="tight")
    plt.close(fig)


def fig_volscatter(a):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, idx in zip(axs, ex.INDICES):
        e = a["errors"][idx]
        x = np.array(e["vol_ratio"]); y = np.array(e["strategies"]["bs_delta"]["pnl"])
        sc = ax.scatter(x, y, c=e["sigma_imp"], s=6, cmap="viridis", alpha=0.7)
        ax.axvline(1, color="#888", ls=":", lw=1)
        ax.set_title(f"{IDXN[idx]}: delta hedging, real test episodes")
        ax.set_xlabel("realised vol in episode / quoted vol")
        r = e["strategies"]["bs_delta"]["corr_pnl_volratio"]
        ax.text(0.97, 0.05, f"correlation {r:.2f}", transform=ax.transAxes, ha="right")
    axs[0].set_ylabel("P&L per 100 notional")
    fig.colorbar(sc, ax=axs, label="quoted vol", shrink=0.85)
    fig.savefig(os.path.join(FIG, "fig_volscatter.png"), bbox_inches="tight")
    plt.close(fig)


def fig_zero(a):
    z = a["zero_cost"].get("gbm_mlp_c0_s0")
    if not z:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for t, col in (("0", C["mlp"]), ("15", C["ntb"]), ("25", C["ww"])):
        c = z["curves"][t]
        ax.plot(c["S"], c["delta"], color="#222", lw=1, ls=":")
        ax.plot(c["S"], c["policy"], color=col, lw=2.2, label=f"MLP, {30 - int(t)} days left")
    ax.plot([], [], color="#222", ls=":", label="Black-Scholes delta")
    ax.set_xlabel("index level (strike 100)"); ax.set_ylabel("units held")
    ax.set_title("Zero-cost control: the network learns delta")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_zero.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    m = ex.load("metrics.json"); a = ex.load("analysis.json")
    fig_surprise(ex.load("simulator_validation.json"))
    fig_sweep(m)
    fig_bands(a["band_maps"], a["response_curves"])
    fig_scaling(a["band_scaling"], a.get("inertia"))
    fig_real(m)
    fig_volscatter(a)
    fig_zero(a)
    print("figures:", sorted(os.listdir(FIG)))
