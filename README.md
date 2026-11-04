# Deep hedging under transaction costs

**ES 335 Machine Learning, IIT Gandhinagar.** Param Tanna (24110236) and Pranav Magarde (24110269).

Project website: **https://REPO_OWNER.github.io/deep-hedging/** (interactive figures, full write-up)

We sell a 30-day at-the-money call option on an equity index and hedge it once a day. Black-Scholes says exactly how much of the index to hold, but only if trading is free. With a proportional transaction cost there is no formula for the best strategy. We train neural network policies to minimise the tail risk of the hedged position directly, compare them with Black-Scholes delta and two cost-aware rules from the literature, and test everything on held-out simulations and on real NIFTY 50 and S&P 500 history from 2019 onward. We also check whether the network rediscovers the no-trade band that theory predicts, without being told it exists.

<!-- RESULTS-SUMMARY -->

## The question

For a short 30-day at-the-money call, rebalanced daily under a proportional transaction cost, which hedging policy gives the best tail outcome, measured by **CVaR at 95% of terminal profit and loss net of all fees**? And does a neural policy trained only on that objective learn the no-trade structure that cost-aware rules build in by hand?

* **Unit of analysis.** One episode: one option sold and hedged to expiry. Inside it the policy makes 30 decisions, one per trading day.
* **Target.** None in the supervised sense. Nobody knows the right hedge under costs, which is the whole problem. Policies are judged by the distribution of terminal P&L.
* **Information at decision time.** Prices up to today, the current holding, time to expiry, the volatility quoted at the sale, recent realised volatility. Never a future price; `src/controls.py` and the policy tests check this by perturbing future prices.
* **Limits of the claim.** One contract type, proportional costs without market impact, zero rates and dividends, a volatility quote estimated from past returns (we have no option market data), and a real test period of about seven years.

## Why this is hard, from first principles

1. **Delta hedging cancels only the first-order risk.** Over one step, `dC = Delta dS + (1/2) Gamma dS^2 + Theta dt`. Holding `Delta` units cancels the first term. What is left, `(1/2) Gamma S^2 (sigma^2 dt - r_t^2)`, has mean zero if the volatility you quoted is the one that happens, and its standard deviation shrinks like `sqrt(dt)`. Rebalance four times as often and the error halves. Our engine reproduces this (tail risk falls from -1.006 to -0.506 between 30 and 120 rebalances at 20% volatility).
2. **Costs make continuous hedging infinitely expensive.** Each step delta moves by about `Gamma dS`, so you trade about `Gamma S sigma sqrt(dt)` and pay `k` times that. Over `T/dt` steps the bill grows like `1/sqrt(dt)`. Error shrinks with frequency, cost grows with it, and the best balance depends on the cost.
3. **Trade when it matters, not on a clock.** Let the holding drift up to `H` from delta before trading. The risk you carry grows like `H^2`; the rate at which you hit the band edge and pay falls like `1/H`. Minimising their sum gives `H^3 proportional to k S Gamma^2 / gamma`, so **the optimal band width grows like cost^(1/3)**. This is the Whalley-Wilmott band.
4. **Why learn.** That result is asymptotic in small costs, uses exponential utility instead of tail risk, assumes continuous monitoring and a known volatility. Our problem is the exact one: CVaR of a 30-step daily hedge with a volatility that has to be guessed. A policy trained on that objective optimises what we actually care about.

## Data

| | NIFTY 50 | S&P 500 |
|---|---|---|
| Source | Yahoo Finance chart endpoint, symbol `^NSEI` | same, `^GSPC` |
| Rows | one trading day | one trading day |
| Range used | 2007-09-17 to 2026-11-13 | 2000-01-03 to 2026-11-13 |
| Columns used | date, adjusted close | date, adjusted close |

Access: no key or login. `src/data.py` builds the request; the URL is in the code. **Raw prices are not stored in this repository**; `python src/experiments.py prepare` downloads them into `data/raw/`. The website only shows paths rescaled to 100 at each sale date.

Quality findings that shaped the analysis: Yahoo's NIFTY history starts in 2007 (our proposal assumed 2000); the 31 dropped NIFTY rows are exchange holidays emitted as empty rows (one is 27 November 2008, during the Mumbai attacks); the longest S&P gap is the post-9/11 closure and is kept as a genuine return; adjusted close equals close (price indices, no dividends); NIFTY volume is mostly unreported and is never used. Returns have excess kurtosis of 12 (NIFTY) and 8.5 (S&P) before 2019, and the Ljung-Box statistic on absolute returns is in the thousands against a critical value of 31.4: fat tails and strong volatility clustering, both of which plain GBM rules out.

## Method

**Episodes.** Every data source goes through one protocol: 60 trading days of history, then 30 days of hedging. At the sale the desk quotes the realised volatility of the history, prices the call with Black-Scholes, collects the premium and hedges. Paths are rescaled to 100 at the sale; the strike is 100.

**Split.** Time split at 2019-01-01, made before anything was fitted. Every parameter estimated from data uses pre-2019 returns only. Real test = every episode sold from 2019 onward.

**Simulators.** GBM with each path's volatility drawn from the 7,419 trailing 60-day volatilities seen before 2019, and Heston with stochastic volatility. Our first Heston fit matched daily moments and turned out to be badly wrong for hedging: to make daily tails fat it pushed vol-of-vol to about 2, so a month's realised volatility overshot the quote far more often than in reality. We refit by simulated method of moments on the statistic that actually drives hedging P&L, the distribution of realised over quoted volatility at the 30-day horizon. Vol-of-vol fell to 0.74 (NIFTY) and 0.91 (S&P), the leverage effect came out negative for both, the Feller condition now holds, and the tail loss of plain delta hedging in the simulator matches the real training era within a few percent, although that loss was never a calibration target.

**Strategies.**

| | Idea | Free parameter |
|---|---|---|
| Unhedged | hold nothing | none |
| Black-Scholes delta | hold N(d1) every day | none |
| Leland delta | delta at a cost-inflated volatility | none |
| Constant band | trade back to delta when the drift exceeds `w` | `w`, tuned on validation |
| Whalley-Wilmott | gamma-dependent band, trade to its edge | `gamma`, tuned on validation |
| MLP (deep hedging) | network maps state and current holding to a holding; nothing about bands built in | weights |
| No-trade band network | network outputs a band, hedge = clamp(holding, band) | weights |
| GRU | recurrent network reading the return history, for worlds where volatility is hidden | weights |

The networks are trained without labels. A batch of 8,192 episodes is simulated, the policy makes all 30 decisions, the same accounting engine that scores the baselines computes the P&L, and the gradient of its CVaR (Rockafellar-Uryasev form, with the VaR level learned jointly) flows back through every decision. 1,500 iterations of Adam with cosine decay; the checkpoint with the best validation CVaR is kept.

## Evaluation design

* **Validation** 20,000 simulated episodes (seed 1001) select every checkpoint and every rule parameter. **Simulated test** 100,000 episodes (seed 2002) and the **real test** (2019 onward) are scored once, at the end.
* Every strategy is scored by the same engine on the same paths, with the same cost and premium. Rules are tuned on the same validation set as the networks they are compared with.
* **Metric:** CVaR at 95% of terminal P&L net of fees, per 100 notional. Also mean, standard deviation, fees and trade count.
* **Uncertainty:** paired bootstrap intervals for every CVaR and every difference from delta. Real windows overlap by 29 of 30 days, so real-data intervals use a moving-block bootstrap with 30-day blocks.
* **Controls** that must pass before any result is trusted: `python src/controls.py` runs ten (Greeks against finite differences, the `sqrt(dt)` law, zero mean P&L at zero cost, limiting cases of every rule, causality). The network is additionally trained at zero cost, where the right answer is known.

<!-- RESULTS-DETAIL -->

## Reproducing everything

Python 3.13; CPU is enough.

```bash
git clone https://github.com/REPO_OWNER/deep-hedging
cd deep-hedging
pip install -r requirements.txt

python src/controls.py               # ten correctness checks, all must pass
python src/experiments.py prepare    # download data, quality report, calibration (2-3 min)
python src/experiments.py tune       # band width and Whalley-Wilmott gamma, validation only
python src/experiments.py train-all  # 25 policies (about 2 hours on a laptop CPU)
python src/experiments.py evaluate   # every test number, with bootstrap intervals
python src/analysis.py               # zero-cost control, learned bands, error analysis
python src/export.py                 # data files for the website in docs/data/
python src/figures.py                # static figures in results/figures/
```

The trained policies are committed in `results/models/`, so `evaluate`, `analysis`, `export` and `figures` run without retraining. Seeds are fixed everywhere (validation 1001, test 2002, training 0, 1, 2). `train-all` accepts `WORKERS` and `THREADS` environment variables.

## Repository map

| Path | What it holds |
|---|---|
| `src/data.py` | download, cleaning, log returns, ACF and Ljung-Box written out from their definitions |
| `src/simulators.py` | GBM, Heston (full-truncation Euler), both Heston calibrations |
| `src/samplers.py` | the episode protocol shared by simulated and real data |
| `src/engine.py` | Black-Scholes formulas, the single P&L engine, CVaR |
| `src/strategies.py` | delta, Leland, constant band, Whalley-Wilmott, all causal |
| `src/policy.py` | MLP, no-trade band network, GRU, training loop |
| `src/controls.py` | the ten correctness checks |
| `src/experiments.py` | prepare, tune, train, evaluate |
| `src/analysis.py` | zero-cost control, learned bands, the cube-root fit, error analysis |
| `src/export.py`, `src/figures.py` | website data and static figures |
| `notebooks/deep_hedging.ipynb` | the study as a narrative, with outputs |
| `results/` | every number as JSON, trained models, figures |
| `docs/` | the website (GitHub Pages serves this folder) |

## Use of AI tools

We used Claude (Anthropic) as a coding and writing assistant throughout the project: to implement and debug the simulation, training and evaluation code, to check derivations, and to draft and edit the README, notebook text and website. Every number in this repository comes from code in this repository, every component passes the correctness controls, and we checked the outputs ourselves. Decisions such as discarding the first Heston calibration came from examining the results. We take responsibility for all of it and can explain any part.

## References

1. Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy* 81(3).
2. Leland, H. E. (1985). Option pricing and replication with transactions costs. *Journal of Finance* 40(5).
3. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies* 6(2).
4. Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs. *Mathematical Finance* 7(3).
5. Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3).
6. Lord, R., Koekkoek, R. and van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance* 10(2).
7. Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep hedging. *Quantitative Finance* 19(8).
8. Imaki, S., Imajo, K., Ito, K., Minami, K. and Nakagawa, K. (2021). No-transaction band network: a neural network architecture for efficient deep hedging. arXiv:2103.01775.
9. Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika* 65(2).
10. Data: Yahoo Finance daily history for `^NSEI` and `^GSPC`, retrieved November 2026.

No external code was copied into this repository. The architectures follow the cited papers; the implementations are our own.
