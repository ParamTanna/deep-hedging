/* The hedging engine and the trained policies, ported to JavaScript so the
   page can run them live. Every function mirrors one in src/ and is checked
   against the Python output on identical paths by tests/check_lab.js. */

(function (root) {
  const TRADING_DAYS = 252, STEPS = 30, T = STEPS / TRADING_DAYS, K = 100;

  /* ---------- random numbers: seeded, so a path can be replayed ---------- */
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => {                       // mulberry32
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const normal = () => {                  // Box-Muller, both values used
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x = 0;
      while (x === 0) x = u();
      const y = u(), r = Math.sqrt(-2 * Math.log(x));
      spare = r * Math.sin(2 * Math.PI * y);
      return r * Math.cos(2 * Math.PI * y);
    };
    return { u, normal };
  }

  /* ---------- normal distribution ---------- */
  // erf from W. J. Cody's rational approximations, accurate to ~1e-15
  function erf(x) {
    const ax = Math.abs(x);
    if (ax < 0.5) {
      const t = x * x;
      const p = [3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
                 3.20937758913846947e03, 1.85777706184603153e-1];
      const q = [2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03,
                 2.84423683343917062e03];
      const num = (((p[4] * t + p[0]) * t + p[1]) * t + p[2]) * t + p[3];
      const den = (((t + q[0]) * t + q[1]) * t + q[2]) * t + q[3];
      return x * num / den;
    }
    return x < 0 ? -1 + erfc(ax) : 1 - erfc(ax);
  }
  function erfc(x) {   // x >= 0.5
    if (x < 4) {
      const p = [5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
                 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03,
                 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8];
      const q = [1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02,
                 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03,
                 3.43936767414372164e03, 1.23033935480374942e03];
      let num = p[8] * x, den = x;
      for (let i = 0; i < 7; i++) { num = (num + p[i]) * x; den = (den + q[i]) * x; }
      return Math.exp(-x * x) * (num + p[7]) / (den + q[7]);
    }
    const p = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
               1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
    const q = [2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1,
               6.05183413124413191e-2, 2.33520497626869185e-3];
    const z = 1 / (x * x);
    let num = p[5] * z, den = z;
    for (let i = 0; i < 4; i++) { num = (num + p[i]) * z; den = (den + q[i]) * z; }
    const r = z * (num + p[4]) / (den + q[4]);
    return Math.exp(-x * x) / x * (1 / Math.sqrt(Math.PI) - r);
  }
  const ncdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const npdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

  /* ---------- Black-Scholes at zero rate (engine.bs) ---------- */
  function bs(S, tau, sigma) {
    tau = Math.max(tau, 1e-10);
    const srt = sigma * Math.sqrt(tau);
    const d1 = (Math.log(S / K) + 0.5 * sigma * sigma * tau) / srt, d2 = d1 - srt;
    return { price: S * ncdf(d1) - K * ncdf(d2), delta: ncdf(d1), gamma: npdf(d1) / (S * srt) };
  }

  /* ---------- one GBM path (samplers.gbm, episode part only) ---------- */
  function gbmPath(sigma, r) {
    const dt = 1 / TRADING_DAYS, S = [100];
    for (let t = 0; t < STEPS; t++)
      S.push(S[t] * Math.exp(-0.5 * sigma * sigma * dt + sigma * Math.sqrt(dt) * r.normal()));
    return S;
  }

  /* ---------- P&L of a hedged short call (engine.pnl) ---------- */
  function pnl(S, H, costBps, premium) {
    const k = costBps * 1e-4;
    let prev = 0, cash = 0, cost = 0;
    for (let t = 0; t < H.length; t++) {
      const tr = H[t] - prev, fee = k * Math.abs(tr) * S[t];
      cash -= tr * S[t] + fee; cost += fee; prev = H[t];
    }
    const ST = S[S.length - 1], fee = k * Math.abs(prev) * ST;
    cash += prev * ST - fee; cost += fee;
    return { total: premium + cash - Math.max(ST - K, 0), cost };
  }

  /* ---------- rule strategies (strategies.py) ---------- */
  const tauAt = (t) => T * (1 - t / STEPS);
  function deltaH(S, sigma) { const H = []; for (let t = 0; t < STEPS; t++) H.push(bs(S[t], tauAt(t), sigma).delta); return H; }
  function wwH(S, sigma, costBps, gamma) {
    const k = costBps * 1e-4, H = []; let held = 0;
    for (let t = 0; t < STEPS; t++) {
      const b = bs(S[t], tauAt(t), sigma);
      const hb = k <= 0 ? 0 : Math.min(Math.pow(1.5 * k * S[t] * b.gamma * b.gamma / gamma, 1 / 3), 1);
      if (Math.abs(b.delta - held) > hb) held = b.delta > held ? b.delta - hb : b.delta + hb;
      H.push(held);
    }
    return H;
  }

  /* ---------- the trained networks (policy.py) ---------- */
  // weights: [{W: [out][in], b: [out]}, ...]; hidden layers use SiLU
  function mlpForward(layers, x) {
    let h = x;
    for (let l = 0; l < layers.length; l++) {
      const { W, b } = layers[l], out = new Array(b.length);
      for (let i = 0; i < b.length; i++) {
        let s = b[i]; const w = W[i];
        for (let j = 0; j < h.length; j++) s += w[j] * h[j];
        out[i] = l < layers.length - 1 ? s / (1 + Math.exp(-s)) : s;
      }
      h = out;
    }
    return h;
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  function features(St, t, sigma, hPrev) {
    const tau = (STEPS - t) / TRADING_DAYS;
    let m = Math.log(St / K) / (sigma * Math.sqrt(tau));
    m = Math.min(6, Math.max(-6, m));
    return [m, Math.sqrt(tau / T), sigma * Math.sqrt(T), hPrev];
  }
  function policyDecide(model, St, t, sigma, hPrev) {
    const x = features(St, t, sigma, hPrev);
    if (model.kind === "mlp") return sigmoid(mlpForward(model.layers, x)[0]);
    const o = mlpForward(model.layers, x.slice(0, 3));         // band net
    const lo = sigmoid(o[0]), hi = lo + (1 - lo) * sigmoid(o[1]);
    return Math.min(Math.max(hPrev, lo), hi);
  }
  function policyH(model, S, sigma) {
    const H = []; let h = 0;
    for (let t = 0; t < STEPS; t++) { h = policyDecide(model, S[t], t, sigma, h); H.push(h); }
    return H;
  }

  function cvar(xs, alpha = 0.95) {
    const n = Math.max(1, Math.round((1 - alpha) * xs.length));
    const s = Float64Array.from(xs).sort();
    let a = 0; for (let i = 0; i < n; i++) a += s[i];
    return a / n;
  }

  root.Lab = { rng, erf, ncdf, bs, gbmPath, pnl, deltaH, wwH, policyH, policyDecide,
               features, cvar, STEPS, T, K, TRADING_DAYS };
})(typeof window !== "undefined" ? window : globalThis);
