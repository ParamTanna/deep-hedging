"""Write the fixture that tests/check_lab.js compares the browser port against."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch
import engine, strategies as st, samplers as sp, policy as pol, experiments as ex

torch.set_num_threads(2)


def weights(p):
    """A policy's weights as plain lists, in the format docs/lab.js reads."""
    lin = [m for m in p.net if isinstance(m, torch.nn.Linear)]
    return {"kind": p.kind,
            "layers": [{"W": m.weight.detach().double().tolist(),
                        "b": m.bias.detach().double().tolist()} for m in lin]}


if __name__ == "__main__":
    cost = ex.HEADLINE
    tuned = ex.load("tuned_rules.json")["gbm"][str(cost)]
    ep = sp.gbm(400, np.random.default_rng(99), sigma_pool=ex.vol_pool())
    S = ep.episode.double(); sig = ep.sigma_imp.double()
    T = sp.STEPS / sp.TRADING_DAYS
    prem = pol.premium_bs(ep, torch.float64)
    out = {"paths": S.tolist(), "sigmas": sig.tolist(), "cost": cost,
           "ww_gamma": tuned["ww_gamma"], "premium": prem.tolist()}
    Hd = st.bs_delta(S, 100.0, T, sig, sp.STEPS)
    Hw = st.whalley_wilmott(S, 100.0, T, sig, sp.STEPS, cost, tuned["ww_gamma"])[0]
    out["pnl_delta"] = engine.pnl(S, Hd, 100.0, cost, prem)[0].tolist()
    out["pnl_ww"] = engine.pnl(S, Hw, 100.0, cost, prem)[0].tolist()
    for kind in ("mlp", "ntb"):
        job = f"gbm_{kind}_c{cost}_s0"
        if not os.path.exists(os.path.join(ex.MODELS, job + ".pt")):
            continue
        p = ex.load_policy(job)[0].double()
        with torch.no_grad():
            H = pol.rollout(p, ep.to(torch.float64), torch.float64)
        out["w_" + kind] = weights(p)
        out["H_" + kind] = H[:5].tolist()
        out["pnl_" + kind] = engine.pnl(S, H, 100.0, cost, prem)[0].tolist()
    xs = np.linspace(-6, 6, 241)
    out["erf_x"] = xs.tolist(); out["erf_y"] = [math.erf(v) for v in xs]
    Sg = torch.tensor([80.0, 95.0, 100.0, 104.0, 120.0], dtype=torch.float64)
    p_, d_, g_ = engine.bs(Sg, 100.0, 0.05, 0.22)
    out["bs_S"] = Sg.tolist(); out["bs_p"] = p_.tolist(); out["bs_d"] = d_.tolist(); out["bs_g"] = g_.tolist()
    json.dump(out, open(os.path.join(HERE, "lab_fixture.json"), "w"))
    print("fixture written:", sorted(k for k in out if k.startswith(("pnl_", "w_"))))
