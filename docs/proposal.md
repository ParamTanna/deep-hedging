# Project proposal

The proposal as submitted for ES 335 (Machine Learning, IIT Gandhinagar). Kept unchanged as a record of what we set out to do; the README lists what changed once we had the data in hand.

**Title:** Deep Hedging Under Transaction Costs
**Course:** ES 335, undergraduate track, pair
**Team:** Param Tanna (24110236), Pranav Magarde (24110269)

## Research question

Under proportional transaction costs, how do different hedging strategies for a European call compare in the full distribution of terminal profit and loss, and does a neural policy trained directly on that objective learn the no-trade behaviour that cost-aware analytical rules build in by hand? In short, we are asking which hedging rule to use when rebalancing is not free, and what a learned policy does differently from the Black-Scholes delta.

## Problem and intended claim

Black-Scholes delta hedging is optimal only in a frictionless market where you can rebalance continuously at no cost. Every real desk pays spread and impact on each trade, so following the delta exactly means paying for a lot of small, pointless adjustments. The practical question is how much hedging error you accept in exchange for how much trading cost, and once costs are included there is no closed-form answer.

We are not forecasting prices. We are comparing policies: rules that map the current state of a position to how much of the underlying to hold. What we study is the distribution of profit and loss at expiry, especially its left tail, since the point of hedging is to avoid large losses rather than to earn a return on average.

A useful answer is a clear, evidence-backed statement of where each strategy sits on the cost-versus-risk trade-off at realistic cost levels, with every comparison run on identical price paths. If the learned policy only matches a well-tuned band rule, that is still worth reporting: it would say the analytical rule already captures most of what is available, and we would show the evidence for that rather than claim an improvement we did not get.

## Data plan

Real market data comes from Yahoo Finance through the yfinance package, which needs no key or login. We will pull daily history for the NIFTY 50 (^NSEI) and the S&P 500 (^GSPC) from 2000 to the present, roughly 6,000 trading days per index. One row there is one trading day: date, open, high, low, close, adjusted close and volume. We work from adjusted close, convert to daily log returns for stationarity, and check the usual problems before using any of it, including holiday and missing-day gaps, zero-volume days, duplicate dates, and whether the return series actually shows the fat tails and volatility clustering our simulator assumes (excess kurtosis, and the autocorrelation of absolute returns).

That history is used for three things: estimating realised volatility, calibrating the path simulator, and supplying genuine out-of-sample paths for the final comparison.

Training data is generated. We simulate price paths under geometric Brownian motion and under a Heston stochastic-volatility model whose parameters are calibrated to the real returns. One row of the modelling dataset is a single path-timestep: the state of a hedging episode at one rebalance time, holding time to maturity, moneyness, the current hedge position, and recent realised volatility. One episode is one option hedged from inception to expiry. We plan on about 100,000 episodes of roughly 30 rebalance steps each, so around three million transitions, which is small enough to train in Colab.

## Baseline and candidate methods

We use three baselines, deliberately increasing in strength. The first is the unhedged position, which shows what the risk looks like with no strategy at all. The second is Black-Scholes delta hedging on a fixed rebalancing grid. This is the right baseline because it is provably optimal when trading is continuous and free, so it is not a straw man. The third is a cost-aware rule: delta hedging with a no-trade band, where you rebalance only once the position has drifted far enough from the target delta, combined with the Leland adjustment to volatility. That one matters because beating plain delta under costs is easy, and we did not want a comparison that flatters the model.

The candidate method is a feed-forward network that takes the current state and outputs a hedge ratio. It is applied at every rebalance date, the episode is unrolled to expiry, and profit and loss is accumulated with costs subtracted at each trade. Because that simulation is differentiable, we can train the network end to end by backpropagating through it, first on the variance of terminal P&L and then on CVaR at 95 percent, which targets the left tail directly. A learned policy is a sensible choice here precisely because the cost-adjusted problem has no closed-form solution; it is a control problem rather than a prediction problem. We will also test whether the network needs to be large at all, and report whether extra capacity buys anything.

## Evaluation plan

Every strategy is compared on identical price paths, the same rebalancing grid and the same cost level, so any difference comes from the policy and nothing else.

For simulated data we split by random seed, using separate independently generated sets of paths for training, validation and testing. For real data the split is by time: statistics and calibration use history up to the end of 2018, and everything from 2019 onward is held back. Nothing is fitted on the held-out period, which rules out the easy leak of calibrating the simulator or estimating volatility on data that later appears in the test.

Network size, learning rate, rebalancing frequency and the risk-aversion setting are all chosen on validation paths. The test set is used once, at the end, for the final numbers.

Because the output is a distribution rather than a point prediction, one error metric would be misleading. We report CVaR at 95 percent, the average of the worst five percent of outcomes and the number that actually matters when hedging, alongside the standard deviation of terminal P&L, the mean P&L, and total transaction cost paid with turnover. Cost and risk have to be read together, since any strategy can lower risk simply by trading more.

One internal check we rely on: at zero transaction cost the learned policy should converge to the Black-Scholes delta. If it does not, the pipeline is wrong, and we would rather learn that from a control than from a plausible-looking result.

## Likely challenge

The main one is that we train on simulated paths but want conclusions that hold for real markets. Calibrated GBM and Heston paths are smoother and better behaved than actual index history, which has jumps, fat tails and volatility that clusters, so a policy tuned on them may do noticeably worse on real data. We are treating this as something to measure rather than avoid. The final evaluation runs every strategy on real out-of-sample windows from 2019 onward, and we report the gap between simulated and real performance as one of the results. If that gap is large, it is a genuine finding about the limits of simulator-trained policies, and it goes in the limitations section with the evidence attached.

## Planned website interaction (ES 335)

The site will carry an interactive hedging explorer. The visitor sets the transaction cost level and the risk-aversion parameter with two sliders, and the chart redraws the hedge ratio against spot price for all three strategies on the same axes: Black-Scholes delta, the rule-based band, and the learned policy. The aim is to make the central result visible instead of described. At zero cost the three curves sit almost on top of one another; as cost rises, the learned policy and the band rule flatten out near the current holding, which is the no-trade region appearing on screen.

A second panel below shows the terminal P&L distribution for the selected setting with CVaR marked, plus a small table of risk and cost, so a visitor can see what the extra trading actually bought.

Everything is precomputed. We export policy grids and P&L samples to JSON and the page reads them with Plotly.js, so it runs as a static site on GitHub Pages with no backend.

## Tools we expect to use

Python 3, NumPy, pandas, PyTorch (neural policy and differentiable P&L simulation, autograd), SciPy (model calibration, statistics), yfinance (real index data), Matplotlib (figures), Jupyter or Google Colab (development, GPU), Git and GitHub, GitHub Pages (static site), HTML, CSS and JavaScript with Plotly.js for the interactive hedging explorer, JSON for precomputed policy grids.
