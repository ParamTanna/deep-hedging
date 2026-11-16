"""Build notebooks/deep_hedging.ipynb from the cells below, then execute it.

    python notebooks/build_notebook.py

The notebook does not retrain the 25 policies (that takes hours). It loads the
trained models in results/models, reruns the cheap parts of the pipeline live,
and reads every reported number from results/*.json, the same files the
website uses. One short training run is included so a reader can watch the
loss go down.
"""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Deep hedging under transaction costs

ES 335 project, Param Tanna (24110236) and Pranav Magarde (24110269).

This notebook walks through the study in the order we actually did it. The heavy lifting lives in `src/`; here we call it, look at what comes out, and explain what we concluded at each step. Every number printed below is either computed live in this notebook or read from `results/`, the same files the project website draws from.

The question: we sell a 30-day at-the-money call on an index and hedge it once a day. Under a proportional transaction cost, which hedging policy gives the smallest tail loss (CVaR at 95% of terminal P&L, net of fees), and does a neural network trained on that objective rediscover the no-trade band that theory predicts?
""")

code(r"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from IPython.display import Image, display
import data, engine, strategies as st, samplers as sp, simulators as sim, policy as pol
import experiments as ex
torch.set_num_threads(8)
pd.set_option('display.width', 140, 'display.precision', 4)
RES = os.path.join('..', 'results')
load = lambda f: json.load(open(os.path.join(RES, f)))
print('torch', torch.__version__, '| numpy', np.__version__)
""")

md(r"""
## 1. The data, and the two facts that decide everything

Daily history of the NIFTY 50 and the S&P 500 from Yahoo Finance. One row is one trading day; we use only the adjusted close. `data.fetch` caches the download in `data/raw/`, which is not committed (we do not redistribute the raw prices).
""")

code(r"""
rep = load('data_report.json')
rows = ['rows_downloaded', 'duplicate_dates_dropped', 'nonpositive_or_missing_close_dropped',
        'rows_kept', 'first_date', 'last_date', 'adjclose_equals_close', 'zero_volume_days',
        'largest_gap_days', 'largest_gap_ends']
pd.DataFrame({i: [rep[i][r] for r in rows] for i in rep}, index=rows)
""")

md(r"""
NIFTY on Yahoo only starts in September 2007, not 2000 as our proposal assumed. The 31 dropped NIFTY rows are exchange holidays that come back as empty rows (one is 27 November 2008, the Mumbai attacks). The longest S&P gap ends on 17 September 2001, the post-9/11 closure; that return is real and stays in. Adjusted close equals close, so these are price indices with no dividends, which matches our zero-dividend model.

Now the statistics that matter. Plain geometric Brownian motion says daily log returns are independent normal draws. Excess kurtosis should then be 0, and squared returns should be uncorrelated.
""")

code(r"""
facts = ['annualised_vol', 'skewness', 'excess_kurtosis', 'ljung_box_returns_20', 'ljung_box_abs_returns_20']
pd.DataFrame({(i, era): [rep[i][era][k] for k in facts] for i in rep for era in ('pre2019', 'post2019')},
             index=facts).round(3)
""")

md(r"""
Both assumptions fail badly. Kurtosis is 12 and 8.5 before 2019, and the Ljung-Box statistic on absolute returns is in the thousands against a 5% critical value of 31.4: calm days follow calm days, violent days follow violent ones. Note also that the test period (2019 on) is harsher than the training period: NIFTY's kurtosis jumps to 21. March 2020 lives in the test set.

## 2. The engine has to be right before anything else matters

There is one P&L engine and every strategy, learned or not, is scored by it. Before trusting a single comparison we run ten checks. Each one tests a property that must hold if the mathematics is implemented correctly.
""")

code(r"""
import controls
assert controls.main() == 0
""")

md(r"""
Check 2 is the one we like most. At zero cost the hedging error should shrink like the square root of the rebalancing interval, so four times as many rebalances should halve the tail. From 30 to 120 rebalances the CVaR goes from about -1.006 to -0.506: a ratio of 1.99.

## 3. Episodes: one protocol for every data source

An episode is 60 trading days of history, then 30 days of hedging. At the sale the desk quotes the realised volatility of the history, prices the call with Black-Scholes and starts hedging. Real windows and simulated windows go through exactly the same code.
""")

code(r"""
df, _ = data.clean(data.fetch('NIFTY50'))
real_test = sp.real(df, '2019-01-01')
print(real_test.S.shape[0], 'real NIFTY test episodes; first sold', real_test.dates[0], ', last', real_test.dates[-1])
print('quoted vol across them: median %.3f, 5%% %.3f, 95%% %.3f' % tuple(np.percentile(real_test.sigma_imp.numpy(), [50, 5, 95])))
print('every path starts at 100:', bool(torch.allclose(real_test.S[:, sp.HIST], torch.tensor(100., dtype=real_test.S.dtype))))
""")

md(r"""
## 4. The simulator we threw away

Training needs hundreds of thousands of episodes, and real history has a couple of hundred non-overlapping ones. So we fit a price generator on the training era. Our first Heston fit matched daily moments. It looked fine, and it was wrong for hedging: to first order a hedged option's P&L is driven by how far realised volatility strays from the quoted one, and the daily-moment fit made that surprise far too large. We refit on the statistic that actually matters, the distribution of realised over quoted volatility at the 30-day horizon.
""")

code(r"""
cal_d, cal_h = load('heston_calibration.json'), load('heston_horizon_calibration.json')
pd.DataFrame({(i, fit): [c[i][k] for k in ('kappa', 'theta', 'xi', 'rho', 'feller_satisfied')]
              for i in cal_d for fit, c in (('daily fit', cal_d), ('horizon fit', cal_h))},
             index=['kappa', 'theta', 'xi', 'rho', 'Feller holds'])
""")

code(r"""
display(Image(os.path.join(RES, 'figures', 'fig_surprise.png')))
chk = load('analysis.json')['extras']['sim_check']
pd.DataFrame(chk).T.round(3)
""")

md(r"""
The last table is the check that convinced us. Delta hedging's tail loss was never a calibration target, yet under the horizon fit (`heston`) it lands within a few percent of the real training-era value (`real`), while the daily fit (`heston_daily`) roughly doubles it for NIFTY. This is still the calibration period, so we treat it as a sanity check, not evidence. The evidence is the 2019+ test.

## 5. Baselines, and one short training run

The rule-based strategies, weakest to strongest: unhedged, Black-Scholes delta, Leland delta, a constant no-trade band, and the Whalley-Wilmott band, whose width follows gamma and, in continuous trading, grows like cost^(1/3). The band width and the Whalley-Wilmott risk aversion were tuned on validation paths only.
""")

code(r"""
tuned = load('tuned_rules.json')
pd.DataFrame(tuned['gbm']).T[['band_w', 'band_val_cvar', 'ww_gamma', 'ww_val_cvar']]
""")

md(r"""
Whalley-Wilmott beats the constant band at every cost level: a width that adapts to gamma beats one fixed number. It is the real bar for the network.

Here is a deliberately short training run so you can watch the idea work: 120 iterations, nothing like the 1,500 the real models get. The network sees the state and its current holding; the loss is CVaR of the simulated P&L; the gradient goes back through all 30 decisions.
""")

code(r"""
pool = ex.vol_pool()
val = sp.gbm(5000, np.random.default_rng(1001), sigma_pool=pool)
demo = pol.make_policy('mlp')
res = pol.train(demo, lambda n, rng: sp.gbm(n, rng, sigma_pool=pool), 10, val,
                iters=120, batch=4096, eval_every=20, seed=7, log=None)
h = pd.DataFrame(res['history'])
bs_val = float(engine.cvar(ex.score(ex.rule_holdings('bs_delta', val, 10), val, 10)[0]))
ax = h.plot(x='iter', y='val_cvar', marker='o', figsize=(6, 3.5), legend=False, title='A short demo run at 10 bps')
ax.axhline(bs_val, color='k', ls=':', label='Black-Scholes delta'); ax.set_ylabel('validation CVaR95'); ax.legend()
print('delta on this validation set: %.3f, demo network after 120 iterations: %.3f' % (bs_val, h.val_cvar.max()))
""")

md(r"""
## 6. Results on the simulated test set

Everything below is read from `results/metrics.json`, produced by `python src/experiments.py evaluate` on 100,000 held-out episodes that played no part in training or tuning. Intervals are paired bootstrap 95% intervals.
""")

code(r"""
m = load('metrics.json')
def table(rows, keep=None):
    out = []
    for k, r in rows.items():
        if keep and k not in keep: continue
        d = r['cvar95_minus_bs_delta_ci']
        out.append([k, r['cvar95'], f"[{r['cvar95_ci'][0]:.3f}, {r['cvar95_ci'][1]:.3f}]",
                    r['cvar95'] - rows['bs_delta']['cvar95'], f"[{d[0]:.3f}, {d[1]:.3f}]",
                    r['mean'], r['std'], r['mean_cost'], r['mean_trades']])
    return pd.DataFrame(out, columns=['strategy', 'CVaR95', 'CI', 'vs delta', 'CI of diff',
                                      'mean', 'sd', 'fees', 'trades']).set_index('strategy').round(4)
table(m['gbm']['10'], keep=['unhedged', 'bs_delta', 'leland', 'band', 'ww', 'gbm_mlp_c10_s0',
                             'gbm_ntb_c10_s0', 'gbm_mlpq_c10_s0'])
""")

code(r"""
display(Image(os.path.join(RES, 'figures', 'fig_sweep.png')))
""")

md(r"""
**Reading the table.** Both networks beat every rule at 10 bps. Delta is the wrong yardstick for a claim about learning, so the next cell compares each network with the best rule at each cost, using the same paired bootstrap. At every nonzero cost both intervals exclude zero; at 10 bps the networks are ahead of Leland's rule by 6.0% and 6.6% of its tail loss, and at 50 bps the band network is ahead of Whalley-Wilmott by 10.5%.
""")

code(r"""
con = a_all = load('analysis.json')['contrasts']
pd.DataFrame({c: {'best rule': con[f'gbm_{c}']['best_rule'],
                  **{j.split('_')[1] + ' minus best rule (95% CI)': f'[{v[0]:.3f}, {v[1]:.3f}]'
                     for j, v in con[f'gbm_{c}']['minus_best_rule_ci'].items()}}
              for c in (5, 10, 20, 50)}).T
""")

md(r"""
The MLP trained on a quadratic loss (`gbm_mlpq_c10_s0` above) improves on delta's tail by only 5.7%, against 9.8% for the same network trained on CVaR. Choosing the loss is choosing which mistakes matter. Seeds move the 10 bps result by at most 0.004.

## 7. What did the network learn?

First the control. Trained at zero cost, the network should stay close to Black-Scholes delta, the frictionless answer. With daily rebalancing and a tail objective, delta is not exactly optimal, so a small, smooth deviation is allowed; a large or erratic one would mean our simulation or our gradients were wrong.
""")

code(r"""
a = load('analysis.json')
pd.DataFrame(a['zero_cost']).T.drop(columns='curves', errors='ignore').round(4)
""")

code(r"""
display(Image(os.path.join(RES, 'figures', 'fig_zero.png')))
display(Image(os.path.join(RES, 'figures', 'fig_bands.png')))
""")

md(r"""
To see what the MLP learned about not trading, we ask it, at a fixed index level and time, what it would hold tomorrow for every possible current holding. A hard no-trade band would show up as a stretch where the answer equals the current holding. The MLP does not do that. It trades most of the way to a target and keeps a fraction of the gap, and that fraction (its *inertia*) is what we measure. At zero cost it should be zero.
""")

code(r"""
inert = pd.DataFrame({c: {S: v[S]['slope'] for S in v} for c, v in a['inertia'].items()}).T
inert.index.name = 'cost (bps)'; inert.columns = ['S = 96', 'S = 100', 'S = 104']
inert.round(3)
""")

md(r"""
Where the MLP *rests* is informative too. We find the volatility at which Black-Scholes delta best matches its resting holdings. A ratio above 1 means it hedges as if volatility were higher than it is: a flatter holding curve that pre-positions for large moves.
""")

code(r"""
ev = pd.DataFrame({c: {f'{30 - int(t)} days left': v[t]['ratio'] for t in v} for c, v in a['effective_vol'].items()}).T
ev.index.name = 'cost (bps)'
ev.round(3)
""")

md(r"""
The band network has the band built in, but not its width: that it has to learn. Whalley-Wilmott says the optimal width grows like cost^(1/3), for continuous trading. Here is the fit.
""")

code(r"""
display(Image(os.path.join(RES, 'figures', 'fig_scaling.png')))
pd.DataFrame({k: {kk: v[kk] for kk in ('exponent', 'exponent_se', 'r2')} for k, v in a['band_scaling'].items()}).T
""")

md(r"""
**The exponent is 0.77, not 1/3.** Our first suspicion was the network, so the next cell refits the two rules whose widths were tuned on validation separately at each cost, with no network involved. They also grow faster than the cube root.
""")

code(r"""
br = a['band_regime']
pd.Series({'band network exponent': a['band_scaling']['ntb']['exponent'],
           'tuned constant band exponent': br['band_exponent'],
           'tuned Whalley-Wilmott exponent': br['ww_exponent'],
           "one day's move in delta at the money": br['one_day_delta_move'],
           'band network half-width at 50 bps': a['band_scaling']['ntb']['points'][-1][1]}).round(3)
""")

md(r"""
The cube-root argument assumes the holding drifts smoothly into the edge of the band, which needs continuous trading. With one decision a day, delta moves further in a day than the whole band is wide, so the edge is crossed by jumps and the choice is made step by step: keep a gap $x$ and carry risk of order $x^2$ for a day, or pay $k|x|$ to close it. That threshold grows like $k$ itself. The measured exponents sit between the continuous-time $1/3$ and the one-step $1$.

**What the MLP's volatility inflation buys.** The MLP rests near delta computed at 1.14 times the true volatility even at zero cost. Hedging at a higher volatility costs a fixed amount on every path and pays back $\tfrac12(\hat\sigma^2-\sigma^2)\int S^2\Gamma_{\hat\sigma}\,dt$, which is large on paths that stay near the strike, exactly where daily rebalancing leaks most. The next cell (validation set only) scores delta at a scaled volatility, and Leland's rule with its volatility scaled by the MLP's zero-cost factor, next to the MLP itself.
""")

code(r"""
ex_ = a['explain_mlp']
display(pd.Series(ex_['zero_cost_delta_at_c'], name='zero cost: CVaR of delta at sigma x c').round(4))
pd.DataFrame(ex_['by_cost']).T.round(4)
""")

md(r"""
The rule recovers 78% of the MLP's gain over delta at zero cost, 87% at 10 bps and 76% at 50 bps. What it misses grows with cost and is the inertia measured above. The cost part of the inflation grows linearly in the cost, like Leland's, but hardly depends on the level of volatility, where Leland's falls like $1/\sigma$ (`a['explain_mlp']['leland_scaling']`): Leland's direction, not Leland's formula.

## 8. Real markets, 2019 onward

Now the part that matters most and that we could not tune on. Every strategy, with every parameter frozen, on every NIFTY and S&P option sold from 1 January 2019. Neighbouring windows share 29 of their 30 days, so intervals come from a moving-block bootstrap with 30-day blocks.
""")

code(r"""
for idx in ('NIFTY50', 'SP500'):
    print(idx, '-', m['real'][idx]['test_2019_on|n_windows'], 'episodes')
    display(table(m['real'][idx]['test_2019_on|gbm_tuned']))
display(Image(os.path.join(RES, 'figures', 'fig_real.png')))
""")

code(r"""
def ladder(idx):
    rows = {}
    worlds = {'GBM sim': m['gbm']['10'], 'Heston sim': m['heston'][idx],
              'real, calibration era': m['real'][idx]['calib_era_pre2019|gbm_tuned'],
              'real, 2019 on': m['real'][idx]['test_2019_on|gbm_tuned']}
    for k in worlds['real, 2019 on']:
        rows[k] = {w: (v[k]['cvar95'] if k in v else np.nan) for w, v in worlds.items()}
    return pd.DataFrame(rows).T.round(3)
for idx in ('NIFTY50', 'SP500'):
    print(idx); display(ladder(idx))
""")

md(r"""
**What survives.** Networks trained on the calibrated Heston simulator cut delta's tail loss by 14% to 16% on NIFTY and 11% to 12% on the S&P, every interval excluding zero. The same architectures trained on GBM gain 8% and 2%, the second indistinguishable from zero: the simulator a network learns in shows up in how it trades real markets. Against the best rule, picked by its score on this very test set (which flatters it), the margin is smaller and mostly not significant:
""")

code(r"""
for idx in ('NIFTY50', 'SP500'):
    c = con[f'real_{idx}']
    print(idx, '- best rule:', c['best_rule'], round(c['best_rule_cvar'], 3))
    display(pd.DataFrame(c['minus_best_rule_ci'], index=['2.5%', '97.5%']).T.round(3))
""")

md(r"""
In the ladder above, delta's tail in the Heston worlds sits close to its tail in the real calibration years, which is what the horizon calibration was for. On NIFTY's calibration era the Heston-trained networks trail delta; section 9 traces that to one day.

## 9. Where the losses come from

To first order a hedged option loses money when realised volatility beats the quote. Every dot below is one real test episode.
""")

code(r"""
rg = a['regimes']; e = rg['edges']
names = ['quietest', 'quiet', 'middle', 'rough', 'roughest']
pd.DataFrame({k: [r['mean'] for r in v] for k, v in rg['strategies'].items()},
             index=[f'{names[g]} ({e[g]:.2f} to {e[g+1]:.2f})' for g in range(5)]).round(3)
""")

md(r"""
That table splits the 100,000 simulated test paths into fifths by how much realised volatility exceeded the quote, and shows each strategy's mean P&L inside each fifth. It tells us *where* a strategy wins: on quiet paths (saving fees) or on rough ones (protecting the tail).

Now the real data.
""")

code(r"""
display(Image(os.path.join(RES, 'figures', 'fig_volscatter.png')))
w = a['errors']['NIFTY50']['strategies']['bs_delta']['worst'][:5] + a['errors']['SP500']['strategies']['bs_delta']['worst'][:5]
pd.DataFrame(w)[['date', 'pnl', 'hedge_error', 'fee', 'sigma_imp', 'rv', 'vol_ratio', 'move_pct']].round(3)
""")

md(r"""
**The worst episodes are the same for every strategy:** options sold in early March 2020. On NIFTY, delta's worst episodes outside 2020 were sold in late April 2024 and run through 4 June 2024, the general election result, when the index fell about 6%. On the S&P the Heston-trained GRU's worst loss is 6.40 against delta's 10.00; the gain is concentrated where it matters.

**The one day the simulator never showed the networks.** The next cell looks at NIFTY's calibration era, where the Heston-trained networks trail delta.
""")

code(r"""
ce = a['calib_era']
display(pd.DataFrame(ce['worst_years']).fillna(0).astype(int).T)
print('largest one-day move in the data:', ce['largest_move'])
ce['jump_window']
""")

md(r"""
Most of the networks' worst windows were sold in 2008 and 2009 (83%, against 46% for delta), and the very worst contain 18 May 2009, the first trading day after the general election result, when NIFTY rose 18%. The day before, the network held 0.65 of the index where delta held 0.74: the flatter hedge that helps everywhere else left it short of an in-the-money option that jumped, at a cost of about 1.69 per 100 notional. Heston prices move continuously and practically never jump like that, so the networks never learned to fear it. Jumps are the first thing we would add to the simulator.

The full discussion of limitations and the interactive versions of every figure are on the project website.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = os.path.join(HERE, "deep_hedging.ipynb")
nbf.write(nb, path)
if os.environ.get("NO_EXEC") != "1":
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    nbf.write(nb, path)
print("wrote", path)
