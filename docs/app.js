/* Deep hedging under transaction costs: page logic.
   Every number in the prose comes from data/site.json through a data-k key,
   so the text can never disagree with the experiments that produced it. */

const D = { site: null, bands: null, eps: {} };

const NAMES = {
  unhedged: "Unhedged",
  bs_delta: "Black-Scholes delta",
  leland: "Leland delta",
  band: "Constant band",
  ww: "Whalley-Wilmott band",
  mlp: "Deep hedge (MLP)",
  ntb: "No-trade band net",
  mlpq: "MLP, quadratic loss",
  mlprv: "MLP + vol feature",
  ntbrv: "Band net + vol feature",
  gru: "Recurrent (GRU)",
};
const COLORS = {
  unhedged: "#9a9a92", bs_delta: "#5b5b55", leland: "#8c6bb1", band: "#c06c9a",
  ww: "#0f8b8d", mlp: "#1f78c1", ntb: "#2f7d4f", mlpq: "#6b86b8",
  mlprv: "#5fb3e8", ntbrv: "#7bbf6a", gru: "#c2185b",
};
const DASH = { unhedged: "dot", bs_delta: "solid", leland: "dashdot", band: "dash", ww: "solid" };

function kindOf(id) {
  if (!id) return id;
  if (id.startsWith("gbm_mlpq")) return "mlpq";
  if (id.includes("_mlprv_")) return "mlprv";
  if (id.includes("_ntbrv_")) return "ntbrv";
  if (id.includes("_gru_")) return "gru";
  if (id.startsWith("gbm_mlp")) return "mlp";
  if (id.startsWith("gbm_ntb")) return "ntb";
  return id;
}
function label(id) {
  const k = kindOf(id);
  let n = NAMES[k] || id;
  if (id.startsWith("hes_")) n += ", Heston-trained";
  else if (id.startsWith("gbm_") && id.includes("_s1")) n += " (seed 1)";
  else if (id.startsWith("gbm_") && id.includes("_s2")) n += " (seed 2)";
  const w = id.match(/_w(\d+)_/); if (w) n += ", width " + w[1];
  return n;
}
const color = (id) => COLORS[kindOf(id)] || "#888";

/* ---------------- helpers ---------------- */
function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function fmt(v, f) {
  if (v === undefined || v === null || (typeof v === "number" && !isFinite(v))) return "n/a";
  if (typeof v !== "number") return String(v);
  if (!f) return v.toFixed(3);
  if (f === "int") return Math.round(v).toLocaleString("en-US");
  if (f === "pct") return (100 * v).toFixed(1) + "%";
  if (f === "pct0") return (100 * v).toFixed(0) + "%";
  if (f === "abs") return Math.abs(v).toFixed(2);
  const m = f.match(/^d(\d)$/); if (m) return v.toFixed(+m[1]);
  return v.toFixed(3);
}
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function baseLayout(extra) {
  const ink = css("--ink"), ink3 = css("--ink-3"), rule = css("--rule");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Manrope, system-ui, sans-serif", size: 12.5, color: ink },
    margin: { l: 62, r: 18, t: 18, b: 52 },
    xaxis: { gridcolor: rule, zerolinecolor: rule, linecolor: rule, tickfont: { color: ink3 } },
    yaxis: { gridcolor: rule, zerolinecolor: rule, linecolor: rule, tickfont: { color: ink3 } },
    legend: { orientation: "h", y: -0.22, font: { size: 12 } },
    hoverlabel: { font: { family: "Manrope, sans-serif" } },
  }, extra || {});
}
const CFG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"] };
function plot(id, data, layout) {
  const el = document.getElementById(id);
  if (!el || typeof Plotly === "undefined") return;
  Plotly.react(el, data, baseLayout(layout), CFG);
}

/* ---------------- text fill ---------------- */
function fillText() {
  document.querySelectorAll("[data-k]").forEach((el) => {
    let v = get(D.site, el.dataset.k);
    if (el.dataset.neg === "1" && typeof v === "number") v = -v;
    el.textContent = fmt(v, el.dataset.f);
  });
}

/* ---------------- tables ---------------- */
function metricsTable(elId, rows, order, opts) {
  const el = document.getElementById(elId);
  if (!el || !rows) return;
  opts = opts || {};
  const ids = order.filter((k) => rows[k]);
  const best = Math.max(...ids.filter((k) => k !== "unhedged").map((k) => rows[k].cvar95));
  let h = "<table><thead><tr><th>Strategy</th><th>CVaR<sub>95</sub></th><th>95% CI</th>" +
    "<th>vs delta</th><th>Mean</th><th>Std dev</th><th>Mean fees</th><th>Trades</th></tr></thead><tbody>";
  for (const k of ids) {
    const r = rows[k];
    const d = r.cvar95_minus_bs_delta_ci;
    const dm = k === "bs_delta" ? "" :
      (d ? `${(r.cvar95 - rows.bs_delta.cvar95 >= 0 ? "+" : "")}${(r.cvar95 - rows.bs_delta.cvar95).toFixed(3)}` +
        ` <span class="ci">[${d[0].toFixed(3)}, ${d[1].toFixed(3)}]</span>` : "");
    const isBest = Math.abs(r.cvar95 - best) < 1e-12;
    h += `<tr class="${opts.hl && opts.hl.includes(k) ? "hl" : ""}"><td>${label(k)}</td>` +
      `<td class="${isBest ? "best" : ""}">${r.cvar95.toFixed(3)}</td>` +
      `<td class="ci">[${r.cvar95_ci[0].toFixed(3)}, ${r.cvar95_ci[1].toFixed(3)}]</td>` +
      `<td>${dm}</td><td>${r.mean.toFixed(3)}</td><td>${r.std.toFixed(3)}</td>` +
      `<td>${r.mean_cost.toFixed(3)}</td><td>${r.mean_trades.toFixed(1)}</td></tr>`;
  }
  h += "</tbody>";
  if (opts.caption) h += `<caption>${opts.caption}</caption>`;
  el.innerHTML = h + "</table>";
}

function simpleTable(elId, head, body, caption) {
  const el = document.getElementById(elId);
  if (!el) return;
  let h = "<table><thead><tr>" + head.map((x) => `<th>${x}</th>`).join("") + "</tr></thead><tbody>";
  for (const row of body) h += "<tr>" + row.map((x) => `<td>${x}</td>`).join("") + "</tr>";
  h += "</tbody>" + (caption ? `<caption>${caption}</caption>` : "") + "</table>";
  el.innerHTML = h;
}

/* ---------------- charts: data section ---------------- */
function drawSurprise(idx) {
  const v = D.site.sim_validation[idx];
  const lo = -1.5, hi = 2.0, sz = 0.05, nb = Math.round((hi - lo) / sz);
  const centres = [...Array(nb).keys()].map((i) => lo + (i + 0.5) * sz);
  const density = (xs) => {
    const cnt = new Array(nb).fill(0);
    for (const x of xs) { const b = Math.floor((x - lo) / sz); if (b >= 0 && b < nb) cnt[b]++; }
    return cnt.map((c) => c / (xs.length * sz));
  };
  const real = { x: centres, y: density(v.real_pre2019.log_ratio_sample), type: "bar", name: "Real, pre-2019",
                 marker: { color: css("--ink-3") }, opacity: 0.45, width: sz,
                 hovertemplate: "%{y:.2f}<extra>real</extra>" };
  const models = [["gbm", "GBM", COLORS.unhedged, 2],
                  ["heston_daily", "Heston fitted to daily moments", COLORS.band, 2.2],
                  ["heston_horizon", "Heston fitted to the horizon", COLORS.ww, 2.8]]
    .map(([k, n, c, w]) => ({ x: centres, y: density(v[k].log_ratio_sample), mode: "lines", name: n,
                              line: { shape: "hvh", color: c, width: w },
                              hovertemplate: "%{y:.2f}<extra>" + n + "</extra>" }));
  plot("fig-surprise", [real, ...models], {
    bargap: 0,
    xaxis: { title: "log( realised vol over the episode / quoted vol )", range: [-1.2, 1.6], gridcolor: css("--rule") },
    yaxis: { title: "density", gridcolor: css("--rule") },
    shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink-3"), dash: "dot", width: 1 } }],
  });
}

function drawACF(idx) {
  const v = D.site.sim_validation[idx];
  const lags = [...Array(21).keys()].slice(1);
  const s = [["real_pre2019", "Real, pre-2019", css("--ink"), 3],
             ["gbm", "GBM", COLORS.unhedged, 2],
             ["heston_daily", "Heston, daily-moment fit", COLORS.band, 2],
             ["heston_horizon", "Heston, horizon fit", COLORS.ww, 2.5]];
  plot("fig-acf", s.map(([k, n, c, w]) => ({
    x: lags, y: v[k].acf_r2.slice(1), name: n, mode: "lines+markers",
    line: { color: c, width: w }, marker: { size: 5 },
  })), { xaxis: { title: "lag (trading days)", gridcolor: css("--rule") },
         yaxis: { title: "autocorrelation of squared returns", gridcolor: css("--rule") } });
}

/* ---------------- charts: results ---------------- */
const SWEEP_ORDER = ["bs_delta", "leland", "band", "ww", "mlp", "ntb"];
function drawSweep(metric) {
  const costs = D.site.costs;
  const traces = [];
  for (const k of SWEEP_ORDER) {
    const x = [], y = [], lo = [], hi = [];
    for (const c of costs) {
      const rows = D.site.gbm[String(c)];
      const id = (k === "mlp" || k === "ntb") ? `gbm_${k}_c${c}_s0` : k;
      const r = rows && rows[id];
      if (!r) continue;
      let val;
      if (metric === "cvar") val = r.cvar95;
      else if (metric === "gain") val = (r.cvar95 - rows.bs_delta.cvar95);
      else if (metric === "cost") val = r.mean_cost;
      else if (metric === "sd") val = r.std;
      x.push(c); y.push(val);
      if (metric === "cvar") { lo.push(val - r.cvar95_ci[0]); hi.push(r.cvar95_ci[1] - val); }
      if (metric === "gain" && r.cvar95_minus_bs_delta_ci) {
        lo.push(val - r.cvar95_minus_bs_delta_ci[0]); hi.push(r.cvar95_minus_bs_delta_ci[1] - val);
      }
    }
    traces.push({
      x, y, name: NAMES[k], mode: "lines+markers",
      line: { color: COLORS[k], width: (k === "mlp" || k === "ntb") ? 3 : 2, dash: DASH[k] || "solid" },
      marker: { size: 7 },
      error_y: lo.length ? { type: "data", array: hi, arrayminus: lo, visible: true, thickness: 1, width: 3 } : undefined,
    });
  }
  const yt = { cvar: "CVaR95 of P&L (higher is better)", gain: "CVaR95 minus delta's (higher is better)",
               cost: "mean fees paid per episode", sd: "standard deviation of P&L" }[metric];
  plot("fig-sweep", traces, {
    xaxis: { title: "transaction cost (bps of traded value)", tickvals: costs, gridcolor: css("--rule") },
    yaxis: { title: yt, gridcolor: css("--rule"), zeroline: metric === "gain" },
  });
}

function drawBand() {
  if (!D.bands) return;
  const c = String([0, 5, 10, 20, 50][+document.getElementById("band-cost").value]);
  const t = String([0, 5, 10, 15, 20, 25, 28][+document.getElementById("band-time").value]);
  document.getElementById("band-cost-v").textContent = c + " bps";
  document.getElementById("band-time-v").textContent = (30 - (+t)) + " days";
  const M = D.bands.maps[c] || {};
  const n = M.ntb && M.ntb[t], m = M.mlp && M.mlp[t];
  const base = n || m;
  if (!base) return;
  const S = base.S, tr = [];
  tr.push({ x: S, y: base.ww_hi, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
  tr.push({ x: S, y: base.ww_lo, mode: "lines", fill: "tonexty", fillcolor: "rgba(15,139,141,0.16)",
            line: { width: 0 }, name: "Whalley-Wilmott: no trade in here", hoverinfo: "skip" });
  if (n) {
    tr.push({ x: S, y: n.hi, mode: "lines", line: { color: COLORS.ntb, width: 2.6 }, name: "Band net: learned no-trade edges",
              hovertemplate: "upper %{y:.3f}<extra></extra>" });
    tr.push({ x: S, y: n.lo, mode: "lines", line: { color: COLORS.ntb, width: 2.6 }, showlegend: false,
              hovertemplate: "lower %{y:.3f}<extra></extra>" });
  }
  if (m) tr.push({ x: S, y: m.rest, mode: "lines", line: { color: COLORS.mlp, width: 2, dash: "dash" },
                   name: "MLP: where it settles", hovertemplate: "MLP rests at %{y:.3f}<extra></extra>" });
  tr.push({ x: S, y: base.delta, mode: "lines", line: { color: css("--ink"), width: 1.5, dash: "dot" },
            name: "Black-Scholes delta", hovertemplate: "delta %{y:.3f}<extra></extra>" });
  plot("fig-band", tr, {
    xaxis: { title: "index level (strike = 100)", gridcolor: css("--rule") },
    yaxis: { title: "units of index held", range: [-0.02, 1.02], gridcolor: css("--rule") },
    hovermode: "x unified",
  });
  const i = S.reduce((b, s, j) => Math.abs(s - 100) < Math.abs(S[b] - 100) ? j : b, 0);
  const w = (a, b) => (b[i] - a[i]).toFixed(3);
  const inert = D.site.inertia && D.site.inertia[c] && D.site.inertia[c]["100.0"];
  document.getElementById("band-readout").innerHTML =
    `At the money: no-trade width <b>${n ? w(n.lo, n.hi) : "n/a"}</b> for the band net, ` +
    `<b>${w(base.ww_lo, base.ww_hi)}</b> for Whalley-Wilmott.` +
    (inert ? ` The MLP keeps <b>${(100 * inert.slope).toFixed(0)}%</b> of any gap between its holding and its target rather than closing it.` : "");
  drawResponse(c);
}

function drawResponse(c) {
  const cv = D.bands && D.bands.curves && D.bands.curves[c];
  if (!cv) return;
  const spec = [["delta", "Black-Scholes delta", css("--ink"), "dot", 1.6],
                ["band", "Constant band", COLORS.band, "dash", 1.8],
                ["ww", "Whalley-Wilmott", COLORS.ww, "solid", 2.2],
                ["ntb", "Band net", COLORS.ntb, "solid", 2.6],
                ["mlp", "MLP", COLORS.mlp, "dash", 2.6]];
  const tr = [{ x: [0, 1], y: [0, 1], mode: "lines", line: { color: css("--rule"), width: 1 },
                name: "keep current holding", hoverinfo: "skip" }];
  for (const [k, nm, col, dash, w] of spec) if (cv[k])
    tr.push({ x: cv[k].h_prev, y: cv[k].h_new, mode: "lines", name: nm, line: { color: col, dash, width: w },
              hovertemplate: "holding %{x:.2f} -> %{y:.3f}<extra>" + nm + "</extra>" });
  plot("fig-response", tr, {
    xaxis: { title: "current holding", range: [0, 1], gridcolor: css("--rule") },
    yaxis: { title: "holding chosen for tomorrow", range: [0, 1], gridcolor: css("--rule") },
  });
}

function drawScaling() {
  const sc = D.site.band_scaling && D.site.band_scaling.ntb;
  if (sc) {
    const x = sc.points.map((p) => p[0]), y = sc.points.map((p) => p[1]);
    const lx = x.reduce((s, v) => s + Math.log(v), 0) / x.length, ly = y.reduce((s, v) => s + Math.log(v), 0) / y.length;
    const xs = [x[0] * 0.85, x[x.length - 1] * 1.15];
    plot("fig-scaling", [
      { x, y, mode: "markers", name: `band net (slope ${sc.exponent.toFixed(2)})`, marker: { size: 10, color: COLORS.ntb } },
      { x: xs, y: xs.map((v) => Math.exp(ly + sc.exponent * (Math.log(v) - lx))), mode: "lines",
        line: { color: COLORS.ntb, width: 1.5 }, showlegend: false, hoverinfo: "skip" },
      { x: xs, y: xs.map((v) => Math.exp(ly + (Math.log(v) - lx) / 3)), mode: "lines",
        line: { color: css("--ink-3"), dash: "dot", width: 2 }, name: "continuous trading: slope 1/3" },
      { x: xs, y: xs.map((v) => Math.exp(ly + (Math.log(v) - lx))), mode: "lines",
        line: { color: css("--ink-3"), dash: "dash", width: 1.5 }, name: "one decision per step: slope 1" }],
      { xaxis: { title: "transaction cost (bps, log scale)", type: "log", tickvals: [5, 10, 20, 50], gridcolor: css("--rule") },
        yaxis: { title: "learned half-width at the money (log)", type: "log", gridcolor: css("--rule") } });
  }
  const it = D.site.inertia;
  if (it) {
    const cs = Object.keys(it).map(Number).sort((a, b) => a - b);
    plot("fig-inertia", [{ x: cs, y: cs.map((c) => it[String(c)]["100.0"].slope), mode: "lines+markers",
      line: { color: COLORS.mlp, width: 2.5 }, marker: { size: 9 }, name: "MLP",
      hovertemplate: "%{x} bps: keeps %{y:.0%} of the gap<extra></extra>" }],
      { xaxis: { title: "transaction cost (bps)", tickvals: cs, gridcolor: css("--rule") },
        yaxis: { title: "share of the gap it does not close", tickformat: ".0%", rangemode: "tozero", gridcolor: css("--rule") },
        showlegend: false });
  }
}

function drawZero() {
  const z = D.site.zero_cost && D.site.zero_cost["gbm_mlp_c0_s0"];
  if (!z) return;
  const tr = [];
  const cols = { "0": COLORS.mlp, "15": COLORS.ntb, "25": COLORS.ww };
  for (const t of ["0", "15", "25"]) {
    const c = z.curves[t];
    tr.push({ x: c.S, y: c.delta, mode: "lines", line: { color: css("--ink"), width: 1.2, dash: "dot" },
              showlegend: t === "0", name: "Black-Scholes delta" });
    tr.push({ x: c.S, y: c.policy, mode: "lines", line: { color: cols[t], width: 2.5 },
              name: `MLP, ${30 - +t} days left` });
  }
  plot("fig-zero", tr, {
    xaxis: { title: "index level (strike = 100)", gridcolor: css("--rule") },
    yaxis: { title: "units held", gridcolor: css("--rule") },
  });
}

const WORLDS = [
  ["gbm", "GBM, simulated test"],
  ["heston:NIFTY50", "Heston (NIFTY calibration), simulated test"],
  ["heston:SP500", "Heston (S&P calibration), simulated test"],
  ["real:NIFTY50", "Real NIFTY 50, 2019 onward"],
  ["real:SP500", "Real S&P 500, 2019 onward"],
];
function worldRows(w) {
  if (w === "gbm") return D.site.gbm[String(D.site.headline)];
  const [kind, idx] = w.split(":");
  if (kind === "heston") return D.site.heston[idx];
  return D.site.real[idx]["test_2019_on|gbm_tuned"];
}
function drawWorlds() {
  const w = document.getElementById("world-sel").value;
  const rows = worldRows(w);
  if (!rows) return;
  const ids = Object.keys(rows).filter((k) => k !== "unhedged" && !/_s[12]$/.test(k) && !/_w\d+_/.test(k));
  ids.sort((a, b) => rows[a].cvar95 - rows[b].cvar95);
  plot("fig-worlds", [{
    type: "bar", orientation: "h", y: ids.map(label), x: ids.map((k) => rows[k].cvar95),
    marker: { color: ids.map(color) },
    error_x: { type: "data", symmetric: false, array: ids.map((k) => rows[k].cvar95_ci[1] - rows[k].cvar95),
               arrayminus: ids.map((k) => rows[k].cvar95 - rows[k].cvar95_ci[0]), thickness: 1.2, width: 4, color: css("--ink-3") },
    hovertemplate: "%{y}<br>CVaR95 %{x:.3f}<extra></extra>",
  }], {
    margin: { l: 250, r: 20, t: 10, b: 50 },
    xaxis: { title: "CVaR95 of P&L per 100 notional (right is better)", gridcolor: css("--rule") },
    yaxis: { automargin: true, gridcolor: "rgba(0,0,0,0)" },
    showlegend: false,
  });
  const nWin = w.startsWith("real:") ? D.site.real[w.split(":")[1]]["test_2019_on|n_windows"] : null;
  document.getElementById("world-readout").textContent = nWin
    ? `${nWin.toLocaleString("en-US")} overlapping 30-day windows. Intervals are from a moving-block bootstrap (block of 30 days), because neighbouring windows share most of their path.`
    : "Intervals are from a paired bootstrap over 100,000 test episodes.";
}

function drawVolScatter() {
  const idx = document.getElementById("vs-idx").value;
  const sid = document.getElementById("vs-strat").value.replace("IDX", idx);
  const e = D.site.errors && D.site.errors[idx];
  if (!e) return;
  const s = e.strategies[sid];
  if (!s) return;
  plot("fig-volscatter", [{
    x: e.vol_ratio, y: s.pnl, mode: "markers", type: "scattergl",
    marker: { size: 5, color: e.sigma_imp, colorscale: "Viridis", opacity: 0.7,
              colorbar: { title: { text: "quoted vol", side: "right" }, thickness: 12 } },
    text: e.dates, hovertemplate: "%{text}<br>realised / quoted %{x:.2f}<br>P&L %{y:.2f}<extra></extra>",
  }], {
    xaxis: { title: "realised vol during the episode / vol quoted at inception", gridcolor: css("--rule") },
    yaxis: { title: "P&L per 100 notional", gridcolor: css("--rule") },
    shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink-3"), dash: "dot", width: 1 } }],
  });
  document.getElementById("vs-readout").textContent =
    `Correlation between P&L and the volatility surprise: ${s.corr_pnl_volratio.toFixed(2)}.`;
}

async function loadEpisodes(idx) {
  if (!D.eps[idx]) D.eps[idx] = await (await fetch(`data/episodes_${idx}.json`)).json();
  return D.eps[idx];
}
async function drawEpisode() {
  const idx = document.getElementById("ep-idx").value;
  const eps = await loadEpisodes(idx);
  const sl = document.getElementById("ep-slider");
  sl.max = eps.length - 1;
  const ep = eps[Math.min(+sl.value, eps.length - 1)];
  const days = [...Array(ep.S.length).keys()];
  const tr = [{ x: days, y: ep.S, mode: "lines", line: { color: css("--ink"), width: 2 }, name: "index, rescaled",
                xaxis: "x", yaxis: "y", hovertemplate: "day %{x}: %{y:.2f}<extra></extra>" },
              { x: [0, 30], y: [100, 100], mode: "lines", line: { color: css("--ink-3"), dash: "dot", width: 1 },
                name: "strike", xaxis: "x", yaxis: "y", hoverinfo: "skip" }];
  for (const k of Object.keys(ep.H)) {
    tr.push({ x: days.slice(0, 30), y: ep.H[k], mode: "lines", line: { shape: "hv", color: color(k), width: 2 },
              name: label(k), xaxis: "x2", yaxis: "y2", hovertemplate: "%{y:.3f}<extra>" + label(k) + "</extra>" });
  }
  plot("fig-episode", tr, {
    grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
    yaxis: { title: "index (100 at sale)", gridcolor: css("--rule") },
    yaxis2: { title: "units held", range: [-0.02, 1.02], gridcolor: css("--rule") },
    xaxis: { gridcolor: css("--rule"), showticklabels: false },
    xaxis2: { title: "trading day of the episode", gridcolor: css("--rule") },
    margin: { l: 62, r: 18, t: 10, b: 52 }, height: 560, legend: { orientation: "h", y: -0.16 },
  });
  const pn = Object.keys(ep.pnl).map((k) => `${label(k)} <b>${ep.pnl[k].toFixed(2)}</b>`).join(" &middot; ");
  document.getElementById("ep-readout").innerHTML =
    `Option sold on <b>${ep.date}</b>. Quoted vol ${(100 * ep.sigma_imp).toFixed(1)}%, realised ` +
    `${(100 * ep.rv).toFixed(1)}%. Final P&L per 100 notional: ${pn}`;
}

function drawTraining() {
  const tr = [];
  for (const id of ["gbm_mlp_c10_s0", "gbm_ntb_c10_s0", "gbm_mlp_c10_s1", "gbm_mlp_c10_s2"]) {
    const h = D.site.training && D.site.training[id];
    if (!h) continue;
    tr.push({ x: h.map((r) => r.iter), y: h.map((r) => r.val_cvar), mode: "lines", name: label(id),
              line: { color: color(id), width: id.endsWith("_s0") ? 2.5 : 1.2, dash: id.endsWith("_s0") ? "solid" : "dot" } });
  }
  const r = D.site.gbm[String(D.site.headline)];
  if (r) for (const k of ["bs_delta", "ww"]) {
    const v = D.site.tuned_val && D.site.tuned_val[k];
    if (v === undefined) continue;
    tr.push({ x: [0, D.site.iters], y: [v, v], mode: "lines", name: NAMES[k] + " (validation)",
              line: { color: COLORS[k], dash: "dash", width: 1.5 } });
  }
  plot("fig-train", tr, {
    xaxis: { title: "training iteration", gridcolor: css("--rule") },
    yaxis: { title: "validation CVaR95", gridcolor: css("--rule") },
  });
}

/* ---------------- wiring ---------------- */
function seg(id, cb) {
  const el = document.getElementById(id);
  if (!el) return;
  el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    el.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    cb(b.dataset.v);
  }));
}

function safe(name, fn) {
  try { fn(); } catch (e) { console.warn("could not draw " + name + ":", e); }
}

function renderAll() {
  safe("text", fillText);
  safe("surprise", () => drawSurprise(document.querySelector("#seg-surprise .on")?.dataset.v || "NIFTY50"));
  safe("acf", () => drawACF(document.querySelector("#seg-acf .on")?.dataset.v || "NIFTY50"));
  safe("sweep", () => drawSweep(document.querySelector("#seg-sweep .on")?.dataset.v || "cvar"));
  safe("band", drawBand);
  safe("scaling", drawScaling);
  safe("zero", drawZero);
  safe("worlds", drawWorlds);
  safe("tables", () => window.renderTables && window.renderTables(D));
  safe("volscatter", drawVolScatter);
  safe("training", drawTraining);
  drawEpisode().catch((e) => console.warn("could not draw episode:", e));
}

async function boot() {
  D.site = await (await fetch("data/site.json")).json();
  try { D.bands = await (await fetch("data/band_maps.json")).json(); } catch (e) { D.bands = null; }

  seg("seg-surprise", drawSurprise);
  seg("seg-acf", drawACF);
  seg("seg-sweep", drawSweep);
  ["band-cost", "band-time"].forEach((id) => document.getElementById(id)?.addEventListener("input", drawBand));
  document.getElementById("world-sel")?.addEventListener("change", drawWorlds);
  ["vs-idx", "vs-strat"].forEach((id) => document.getElementById(id)?.addEventListener("change", drawVolScatter));
  document.getElementById("ep-idx")?.addEventListener("change", () => {
    document.getElementById("ep-slider").value = 0; fillWorst(); drawEpisode(); });
  document.getElementById("ep-slider")?.addEventListener("input", drawEpisode);
  document.getElementById("ep-worst")?.addEventListener("change", async (e) => {
    const idx = document.getElementById("ep-idx").value;
    const eps = await loadEpisodes(idx);
    const j = eps.findIndex((x) => x.date === e.target.value);
    if (j >= 0) { document.getElementById("ep-slider").value = j; drawEpisode(); }
  });
  fillWorst();
  renderAll();
  if (window.renderMathInElement) {
    renderMathInElement(document.body, { delimiters: [
      { left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }] });
  }
}

function fillWorst() {
  const idx = document.getElementById("ep-idx")?.value;
  const sel = document.getElementById("ep-worst");
  if (!sel || !D.site.errors || !D.site.errors[idx]) return;
  const w = D.site.errors[idx].strategies.bs_delta.worst;
  sel.innerHTML = '<option value="">jump to a bad episode...</option>' +
    w.map((x) => `<option value="${x.date}">${x.date}: delta lost ${(-x.pnl).toFixed(2)}</option>`).join("");
}

/* theme toggle + nav highlight */
(function () {
  const btn = document.getElementById("theme-btn");
  const saved = (() => { try { return localStorage.getItem("theme"); } catch (e) { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  btn?.addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#15161a";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
    if (D.site) renderAll();
  });
  const links = [...document.querySelectorAll(".nav a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
  window.addEventListener("scroll", () => {
    let k = 0;
    secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top < 120) k = i; });
    links.forEach((a, i) => a.classList.toggle("on", i === k));
  }, { passive: true });
})();

window.addEventListener("load", boot);
