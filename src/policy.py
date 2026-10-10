"""Learned hedging policies and the code that trains them.

What is being learned
---------------------
A policy maps what the desk can observe at a rebalance date to how many units
of the index to hold until the next date. There is no label anywhere in this
file. Nobody knows the right hedge under transaction costs; that is the open
problem. Instead the whole 30-step episode is simulated with differentiable
operations, the terminal profit and loss is computed by the same engine that
scores the baselines, a risk measure of that P&L is the loss, and its gradient
flows back through every one of the 30 decisions.

Three architectures, each with a reason
---------------------------------------
MLPPolicy   The plain deep hedging network (Buehler et al. 2019). Inputs are
            the observable state INCLUDING the current holding. Nothing about
            no-trade regions is built in. If a band appears in what it learns,
            the network discovered it from the objective. This is the model
            that answers our research question.

NTBPolicy   The no-transaction band network (Imaki et al. 2021). The network
            outputs a band [lower, upper] from the state WITHOUT the current
            holding, and the hedge is clamp(current, lower, upper). The band is
            built in; only its shape is learned. It answers a different
            question: if you grant the structure that theory says is optimal,
            how much does learning the width buy over the Whalley-Wilmott
            formula?

GRUPolicy   A recurrent policy that reads the raw return history. Under
            stochastic volatility the variance is not observable, so the state
            a Markov policy sees is incomplete. A recurrent network can carry an
            estimate of the current volatility regime in its hidden state. It is
            used only where volatility actually varies (Heston and real data);
            under constant-volatility GBM it would have nothing to infer.

Feature design
--------------
All features are computable at the rebalance date from information the desk
has. They are expressed in the units the problem is invariant to, so the
network does not have to learn those invariances from scratch:

  m          log(S_t / K) / (sigma_imp * sqrt(tau)), clipped to [-6, 6].
             Moneyness measured in remaining standard deviations. Under GBM the
             Black-Scholes delta depends on S, tau and sigma only through
             (approximately) this one number.
  sqrt_tau   sqrt(tau / T), in (0, 1]. How much of the option's life is left.
  sig        sigma_imp * sqrt(T). The total volatility quoted at inception.
  h_prev     the current holding. Essential under costs: whether to trade
             depends on where you already are.
  rv_rel     trailing 20-day realised vol / sigma_imp - 1. Has volatility moved
             since the option was sold? Only used where it can carry
             information (Heston and real data).
"""
from __future__ import annotations
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import pnl, cvar
from samplers import TRADING_DAYS

K_STRIKE = 100.0


# ------------------------------------------------------------------ features

def trailing_vol_all(S_full: torch.Tensor, hist: int, steps: int,
                     window: int = 20) -> torch.Tensor:
    """Trailing realised vol at every rebalance date, shape (B, steps).

    At step t the price index is hist + t, and we use the `window` daily
    returns that END at that index, i.e. returns already observed. The history
    segment guarantees a full window exists from the very first step.
    """
    r = torch.log(S_full[:, 1:] / S_full[:, :-1])              # (B, hist+steps)
    win = r.unfold(1, window, 1)                               # (B, L-w+1, w)
    sd = win.std(dim=2, unbiased=True) * math.sqrt(TRADING_DAYS)
    # window ending at price index i covers returns i-window .. i-1 -> row i-window
    idx = torch.arange(steps, device=S_full.device) + hist - window
    return sd[:, idx]


def state_features(S_t, t, steps, sigma_imp, h_prev, rv_t=None):
    T = steps / TRADING_DAYS
    tau = (steps - t) / TRADING_DAYS
    m = torch.log(S_t / K_STRIKE) / (sigma_imp * math.sqrt(tau))
    m = m.clamp(-6.0, 6.0)
    sq = torch.full_like(S_t, math.sqrt(tau / T))
    sig = sigma_imp * math.sqrt(T)
    cols = [m, sq, sig, h_prev]
    if rv_t is not None:
        cols.append((rv_t / sigma_imp - 1.0).clamp(-1.0, 3.0))
    return torch.stack(cols, dim=1)


# ------------------------------------------------------------------ networks

def _mlp(n_in, n_out, width, depth):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    kind = "mlp"

    def __init__(self, width=64, depth=3, use_rv=False):
        super().__init__()
        self.use_rv = use_rv
        self.net = _mlp(5 if use_rv else 4, 1, width, depth)

    def decide(self, x, h_prev, carry=None):
        return torch.sigmoid(self.net(x)).squeeze(1), carry

    def init_carry(self, ep, dtype):
        return None


class NTBPolicy(nn.Module):
    """Band [lo, hi] from the state without h_prev; hedge = clamp(h_prev, lo, hi)."""
    kind = "ntb"

    def __init__(self, width=64, depth=3, use_rv=False):
        super().__init__()
        self.use_rv = use_rv
        self.net = _mlp(4 if use_rv else 3, 2, width, depth)

    def bounds(self, x_no_h):
        o = self.net(x_no_h)
        lo = torch.sigmoid(o[:, 0])
        hi = lo + (1.0 - lo) * torch.sigmoid(o[:, 1])
        return lo, hi

    def decide(self, x, h_prev, carry=None):
        keep = [0, 1, 2] + ([4] if self.use_rv else [])
        lo, hi = self.bounds(x[:, keep])
        return torch.minimum(torch.maximum(h_prev, lo), hi), carry

    def init_carry(self, ep, dtype):
        return None


class GRUPolicy(nn.Module):
    """Reads the 60-day history first (no decisions), then decides each day.

    Input per step: [standardised daily return, m, sqrt_tau, h_prev, in_episode].
    During the history the decision features are zero and the flag is 0, so
    the network's only job there is to summarise the return series into its
    hidden state. Returns are standardised by the quoted volatility, so the
    network sees surprises relative to the quote rather than raw magnitudes.
    """
    kind = "gru"

    def __init__(self, hidden=32, width=64):
        super().__init__()
        self.cell = nn.GRUCell(5, hidden)
        self.h0 = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(nn.Linear(hidden + 4, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, 1))
        self.use_rv = False

    def init_carry(self, ep, dtype):
        S = ep.S.to(dtype)
        B = S.shape[0]
        sig = ep.sigma_imp.to(dtype)
        # history returns ending at price indices 1 .. hist-1. The return that
        # ends AT inception (index hist) is left for the first decision step,
        # so every return is read exactly once.
        r = torch.log(S[:, 1:ep.hist] / S[:, :ep.hist - 1])
        z = r / (sig[:, None] * math.sqrt(1.0 / TRADING_DAYS))
        h = self.h0.expand(B, -1).to(dtype)
        zero = torch.zeros(B, dtype=dtype, device=S.device)
        for i in range(ep.hist - 1):
            inp = torch.stack([z[:, i].clamp(-8, 8), zero, zero, zero, zero], 1)
            h = self.cell(inp, h)
        return {"h": h, "S": S, "sig": sig, "hist": ep.hist}

    def decide(self, x, h_prev, carry):
        # x columns: m, sq, sig, h_prev[, rv]. The step return is read from carry.
        t = carry.get("t", 0)
        S, sig, hist = carry["S"], carry["sig"], carry["hist"]
        r = torch.log(S[:, hist + t] / S[:, hist + t - 1])
        z = (r / (sig * math.sqrt(1.0 / TRADING_DAYS))).clamp(-8, 8)
        one = torch.ones_like(z)
        inp = torch.stack([z, x[:, 0], x[:, 1], h_prev, one], 1)
        hid = self.cell(inp, carry["h"])
        out = self.head(torch.cat([hid, x[:, :4]], 1))
        carry = dict(carry, h=hid, t=t + 1)
        return torch.sigmoid(out).squeeze(1), carry


# ------------------------------------------------------------------ rollout

def rollout(policy: nn.Module, ep, dtype=torch.float32) -> torch.Tensor:
    """Apply the policy at every rebalance date. Returns holdings (B, steps).

    Strictly causal: the decision at step t sees S up to index hist + t and the
    holding chosen at t - 1, nothing later.
    """
    S = ep.S.to(dtype)
    sig = ep.sigma_imp.to(dtype)
    rv = trailing_vol_all(S, ep.hist, ep.steps) if policy.use_rv else None
    carry = policy.init_carry(ep, dtype)
    h = torch.zeros(S.shape[0], dtype=dtype, device=S.device)
    out = []
    for t in range(ep.steps):
        x = state_features(S[:, ep.hist + t], t, ep.steps, sig, h,
                           None if rv is None else rv[:, t])
        h, carry = policy.decide(x, h, carry)
        out.append(h)
    return torch.stack(out, 1)


def premium_bs(ep, dtype=torch.float64) -> torch.Tensor:
    """Black-Scholes premium at the quoted volatility, per 100 notional."""
    from engine import bs
    T = ep.steps / TRADING_DAYS
    S0 = torch.full((ep.S.shape[0],), 100.0, dtype=dtype, device=ep.sigma_imp.device)
    p, _, _ = bs(S0, K_STRIKE, T, ep.sigma_imp.to(dtype))
    return p


def episode_pnl(H: torch.Tensor, ep, cost_bps: float, dtype=torch.float64):
    S = ep.episode.to(dtype)
    return pnl(S, H.to(dtype), K_STRIKE, cost_bps, premium_bs(ep, dtype))


# ------------------------------------------------------------------ training

def make_policy(kind, width=64, depth=3, use_rv=False, hidden=32):
    if kind == "mlp":
        return MLPPolicy(width, depth, use_rv)
    if kind == "ntb":
        return NTBPolicy(width, depth, use_rv)
    if kind == "gru":
        return GRUPolicy(hidden, width)
    raise ValueError(kind)


def train(policy: nn.Module, sampler, cost_bps: float, val_ep,
          objective: str = "cvar", alpha: float = 0.95, iters: int = 1500,
          batch: int = 8192, lr: float = 1e-3, eval_every: int = 50,
          seed: int = 0, log=print):
    """Train on fresh simulated episodes, select the checkpoint on validation.

    objective
      "quadratic"  E[PnL^2]. Classic quadratic hedging. Penalises both the
                   spread of outcomes and the average cost drag (since
                   E[X^2] = Var X + (E X)^2). Smooth and easy to optimise.
      "cvar"       CVaR_alpha of the loss -PnL, through the Rockafellar-Uryasev
                   identity  CVaR(L) = min_w { w + E[(L - w)+] / (1 - alpha) }.
                   We learn w jointly with the network. At the optimum w is the
                   VaR. This is an unbiased objective whose minimum over w IS
                   the CVaR, unlike the sorted-tail estimator, which is biased
                   for finite batches.

    Model selection: every `eval_every` iterations we compute CVaR95 on a fixed
    validation set and keep the best weights. The test set is never seen here.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(policy.parameters()) + [w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.01)
    best_state, best_val, hist = None, -math.inf, []
    t0 = time.time()
    for it in range(iters + 1):
        if it % eval_every == 0 or it == iters:
            policy.eval()
            with torch.no_grad():
                Hv = rollout(policy, val_ep)
                tot, cost, _ = episode_pnl(Hv, val_ep, cost_bps, torch.float64)
                v_cvar = float(cvar(tot, alpha))
                v_mean = float(tot.mean())
                v_cost = float(cost.mean())
            hist.append({"iter": it, "val_cvar": v_cvar, "val_mean": v_mean,
                         "val_cost": v_cost, "secs": round(time.time() - t0, 1)})
            if v_cvar > best_val:
                best_val, best_state = v_cvar, copy.deepcopy(policy.state_dict())
            if log and (it % (eval_every * 5) == 0 or it == iters):
                log("  it %5d  val CVaR95 %8.4f  mean %8.4f  cost %7.4f  [%5.0fs]"
                    % (it, v_cvar, v_mean, v_cost, time.time() - t0))
            policy.train()
        if it == iters:
            break
        ep = sampler(batch, rng)
        H = rollout(policy, ep, torch.float32)
        S = ep.episode.to(torch.float32)
        prem = premium_bs(ep, torch.float32)
        tot, _, _ = pnl(S, H, K_STRIKE, cost_bps, prem)
        if objective == "quadratic":
            loss = (tot ** 2).mean()
        elif objective == "cvar":
            loss = w + F.relu(-tot - w).mean() / (1.0 - alpha)
        else:
            raise ValueError(objective)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
    policy.load_state_dict(best_state)
    policy.eval()
    return {"best_val_cvar": best_val, "history": hist,
            "train_secs": round(time.time() - t0, 1)}
