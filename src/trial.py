"""Convergence trial: how long does one policy need, and is it in the right range?

Run once to choose the training budget. Not part of the final pipeline.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import samplers as sp, policy as pol, strategies as st, engine

torch.set_num_threads(int(os.environ.get("THREADS", "14")))
COST = float(os.environ.get("COST", "10"))
KIND = os.environ.get("KIND", "mlp")
ITERS = int(os.environ.get("ITERS", "800"))
BATCH = int(os.environ.get("BATCH", "8192"))

pool = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                            "vol_pool_pre2019.npy"))
val = sp.gbm(20000, np.random.default_rng(1001), sigma_pool=pool)
T = sp.STEPS / sp.TRADING_DAYS
S = val.episode
prem = pol.premium_bs(val)

def score(H, name):
    tot, c, _ = engine.pnl(S, H.double(), 100.0, COST, prem)
    print("  %-26s CVaR95 %8.4f  mean %8.4f  sd %7.4f  cost %7.4f"
          % (name, engine.cvar(tot), tot.mean(), tot.std(), c.mean()))
    return float(engine.cvar(tot))

print("validation baselines at %.0f bps (20,000 GBM episodes, sigma from pool)" % COST)
score(st.unhedged(S, sp.STEPS), "unhedged")
score(st.bs_delta(S, 100.0, T, val.sigma_imp, sp.STEPS), "BS delta")
best = max((score(st.band(S, 100.0, T, val.sigma_imp, sp.STEPS, w)[0], "band w=%.3f" % w), w)
           for w in (0.01, 0.02, 0.03, 0.05, 0.08))
print("  -> best band width %.3f" % best[1])
for g in (0.1, 0.3, 1.0, 3.0):
    score(st.whalley_wilmott(S, 100.0, T, val.sigma_imp, sp.STEPS, COST, g)[0], "Whalley-Wilmott g=%.1f" % g)

sampler = lambda n, rng: sp.gbm(n, rng, sigma_pool=pool)
p = pol.make_policy(KIND)
print("\ntraining %s, %d iters, batch %d" % (KIND, ITERS, BATCH))
t0 = time.time()
res = pol.train(p, sampler, COST, val, objective="cvar", iters=ITERS, batch=BATCH,
                eval_every=25)
print("done in %.0fs, best val CVaR %.4f" % (time.time() - t0, res["best_val_cvar"]))
