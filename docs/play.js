/* Two interactions that run the real engine in the browser (lab.js):
   "Hedge it yourself" and the live laboratory. */

(function () {
  const L = window.Lab;
  let MODELS = null, TUNED = null;

  async function models() {
    if (!MODELS) {
      const j = await (await fetch("data/models.json")).json();
      MODELS = j.models; TUNED = j.tuned;
    }
    return MODELS;
  }
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  /* =================================================== hedge it yourself */
  const G = { S: null, sigma: 0.18, cost: 10, day: 0, H: [], seed: 1, done: false };

  function newGame() {
    G.seed = Math.floor(Math.random() * 1e9);
    G.sigma = +$("g-vol").value / 100;
    G.cost = 10;
    G.S = L.gbmPath(G.sigma, L.rng(G.seed));
    G.day = 0; G.H = []; G.done = false;
    $("g-hold").value = 0.5; syncHold();
    $("g-go").disabled = false; $("g-result").innerHTML = "";
    drawGame();
  }

  function syncHold() { $("g-hold-v").textContent = (+$("g-hold").value).toFixed(2); }

  function feesSoFar() {
    let prev = 0, f = 0;
    for (let t = 0; t < G.H.length; t++) { f += G.cost * 1e-4 * Math.abs(G.H[t] - prev) * G.S[t]; prev = G.H[t]; }
    return f;
  }

  function step() {
    if (G.done) return;
    G.H.push(+$("g-hold").value);
    G.day += 1;
    if (G.day >= L.STEPS) { G.done = true; $("g-go").disabled = true; finish(); }
    drawGame();
  }

  async function finish() {
    const M = await models();
    const prem = L.bs(100, L.T, G.sigma).price;
    const tuned = TUNED[String(G.cost)];
    const rows = [
      ["You", G.H, "#c2185b"],
      ["Black-Scholes delta", L.deltaH(G.S, G.sigma), "#5b5b55"],
      ["Whalley-Wilmott band", L.wwH(G.S, G.sigma, G.cost, tuned.ww_gamma), "#0f8b8d"],
      ["Neural network (MLP)", L.policyH(M["mlp_" + G.cost], G.S, G.sigma), "#1f78c1"],
    ];
    G.compare = rows;
    const res = rows.map(([n, H]) => [n, L.pnl(G.S, H, G.cost, prem)]);
    const you = res[0][1].total;
    const best = res.slice(1).reduce((b, r) => (r[1].total > b[1].total ? r : b));
    let h = "<table><thead><tr><th>Strategy</th><th>Final P&amp;L</th><th>Fees paid</th></tr></thead><tbody>";
    for (const [n, r] of res)
      h += `<tr${n === "You" ? ' class="hl"' : ""}><td>${n}</td><td>${r.total.toFixed(2)}</td><td>${r.cost.toFixed(2)}</td></tr>`;
    h += "</tbody></table>";
    const msg = you >= best[1].total
      ? "You did at least as well as every strategy on this path. Before celebrating, press <i>new path</i> a few times: a single path is mostly luck. Hedging strategies are judged on the worst 5% of thousands of paths, which is what the laboratory further down measures."
      : `${best[0]} did best on this path. One path is mostly luck, though. Strategies are judged on the worst 5% of thousands of paths, which is what the laboratory further down measures.`;
    $("g-result").innerHTML = `<p style="margin:10px 0 8px">The option you sold for ${prem.toFixed(2)} expired with the index at ${G.S[L.STEPS].toFixed(2)}.</p>${h}<p class="readout" style="padding:6px 0 0">${msg}</p>`;
    drawGame();
  }

  function drawGame() {
    if (typeof Plotly === "undefined" || !G.S) return;
    const shown = G.done ? L.STEPS : G.day;
    const days = [...Array(shown + 1).keys()];
    const tr = [{ x: days, y: G.S.slice(0, shown + 1), mode: "lines+markers", name: "index",
                  line: { color: css("--ink"), width: 2 }, marker: { size: 4 }, xaxis: "x", yaxis: "y" },
                { x: [0, 30], y: [100, 100], mode: "lines", name: "strike", hoverinfo: "skip",
                  line: { color: css("--ink-3"), dash: "dot", width: 1 }, xaxis: "x", yaxis: "y" }];
    if (G.done && G.compare) {
      for (const [n, H, c] of G.compare)
        tr.push({ x: [...Array(30).keys()], y: H, mode: "lines", name: n, line: { shape: "hv", color: c, width: n === "You" ? 3 : 1.8 }, xaxis: "x2", yaxis: "y2" });
    } else {
      tr.push({ x: [...Array(G.H.length).keys()], y: G.H, mode: "lines+markers", name: "your holding",
                line: { shape: "hv", color: "#c2185b", width: 3 }, xaxis: "x2", yaxis: "y2" });
      if ($("g-hint").checked) {
        const d = []; for (let t = 0; t <= Math.min(G.day, 29); t++) d.push(L.bs(G.S[t], L.T * (1 - t / 30), G.sigma).delta);
        tr.push({ x: [...Array(d.length).keys()], y: d, mode: "lines", name: "Black-Scholes delta",
                  line: { shape: "hv", color: "#5b5b55", dash: "dot", width: 1.5 }, xaxis: "x2", yaxis: "y2" });
      }
    }
    const lo = Math.min(...G.S.slice(0, shown + 1), 97), hi = Math.max(...G.S.slice(0, shown + 1), 103);
    Plotly.react("fig-game", tr, {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Manrope, sans-serif", size: 12.5, color: css("--ink") },
      grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
      xaxis: { range: [-0.5, 30.5], gridcolor: css("--rule"), showticklabels: false },
      yaxis: { title: "index", range: [lo - 1, hi + 1], gridcolor: css("--rule") },
      xaxis2: { range: [-0.5, 30.5], title: "trading day", gridcolor: css("--rule") },
      yaxis2: { title: "units held", range: [-0.03, 1.03], gridcolor: css("--rule") },
      margin: { l: 56, r: 12, t: 8, b: 44 }, height: 470, legend: { orientation: "h", y: -0.14 },
      showlegend: true,
    }, { displaylogo: false, responsive: true, displayModeBar: false });
    const today = Math.min(G.day, 30);
    $("g-prem").textContent = L.bs(100, L.T, G.sigma).price.toFixed(2);
    $("g-day").textContent = G.done ? "expired" : `day ${today + 1} of 30`;
    $("g-price").textContent = G.S[today].toFixed(2);
    $("g-fees").textContent = feesSoFar().toFixed(3);
    $("g-delta").textContent = G.done ? "" : L.bs(G.S[today], L.T * (1 - today / 30), G.sigma).delta.toFixed(3);
  }

  /* =================================================== live laboratory */
  async function runLab() {
    const btn = $("lab-run"); btn.disabled = true; btn.textContent = "running...";
    await new Promise((r) => setTimeout(r, 30));
    const M = await models();
    const cost = +$("lab-cost").value, sigma = +$("lab-vol").value / 100, n = +$("lab-n").value;
    const tuned = TUNED[String(cost)];
    const prem = L.bs(100, L.T, sigma).price;
    const r = L.rng(20261117);
    const names = ["Black-Scholes delta", "Whalley-Wilmott band", "Neural network (MLP)", "No-trade band net"];
    const out = names.map(() => ({ pnl: new Float64Array(n), fee: 0 }));
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      const S = L.gbmPath(sigma, r);
      const Hs = [L.deltaH(S, sigma), L.wwH(S, sigma, cost, tuned.ww_gamma),
                  L.policyH(M["mlp_" + cost], S, sigma), L.policyH(M["ntb_" + cost], S, sigma)];
      Hs.forEach((H, j) => { const p = L.pnl(S, H, cost, prem); out[j].pnl[i] = p.total; out[j].fee += p.cost; });
      if (i % 500 === 499) await new Promise((res) => setTimeout(res, 0));
    }
    const ms = performance.now() - t0;
    const cols = ["#5b5b55", "#0f8b8d", "#1f78c1", "#2f7d4f"];
    const stats = out.map((o) => {
      const a = Array.from(o.pnl); const m = a.reduce((s, v) => s + v, 0) / n;
      const sd = Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (n - 1));
      return { cvar: L.cvar(a), mean: m, sd, fee: o.fee / n };
    });
    const lo = Math.min(...stats.map((s) => s.cvar)) - 0.6, hi = Math.max(...stats.map((s) => s.mean + 2.5 * s.sd));
    const tr = out.map((o, j) => ({ x: Array.from(o.pnl), type: "histogram", name: names[j], opacity: 0.55,
      histnorm: "probability density", marker: { color: cols[j] }, xbins: { start: lo, end: hi, size: (hi - lo) / 70 } }));
    const shapes = stats.map((s, j) => ({ type: "line", x0: s.cvar, x1: s.cvar, yref: "paper", y0: 0, y1: 1,
      line: { color: cols[j], width: 2, dash: "dash" } }));
    Plotly.react("fig-lab", tr, {
      barmode: "overlay", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Manrope, sans-serif", size: 12.5, color: css("--ink") },
      xaxis: { title: "P&L per 100 notional (dashed lines: CVaR95, the mean of the worst 5%)", range: [lo, hi], gridcolor: css("--rule") },
      yaxis: { title: "density", gridcolor: css("--rule") }, shapes,
      margin: { l: 56, r: 12, t: 8, b: 48 }, legend: { orientation: "h", y: -0.2 },
    }, { displaylogo: false, responsive: true });
    const best = Math.max(...stats.map((s) => s.cvar));
    let h = "<table><thead><tr><th>Strategy</th><th>CVaR<sub>95</sub></th><th>Mean</th><th>Std dev</th><th>Mean fees</th></tr></thead><tbody>";
    stats.forEach((s, j) => { h += `<tr><td>${names[j]}</td><td class="${s.cvar === best ? "best" : ""}">${s.cvar.toFixed(3)}</td><td>${s.mean.toFixed(3)}</td><td>${s.sd.toFixed(3)}</td><td>${s.fee.toFixed(3)}</td></tr>`; });
    $("lab-table").innerHTML = h + "</tbody></table>";
    const pool = sigma < 0.05 || sigma > 0.74
      ? " This volatility lies outside the range the networks were trained on (5% to 74%), so they are extrapolating." : "";
    $("lab-readout").textContent = `${n.toLocaleString("en-US")} fresh paths simulated and hedged in your browser in ${(ms / 1000).toFixed(1)} s, with the real trained weights. With this many paths the CVaR moves by a few hundredths between runs; the 100,000-path numbers in the table above are the precise ones.${pool}`;
    btn.disabled = false; btn.textContent = "run again";
  }

  /* =================================================== wiring */
  window.addEventListener("load", () => {
    if (!$("fig-game")) return;
    $("g-hold").addEventListener("input", syncHold);
    $("g-go").addEventListener("click", step);
    $("g-new").addEventListener("click", newGame);
    $("g-hint").addEventListener("change", drawGame);
    $("g-vol").addEventListener("change", newGame);
    document.addEventListener("keydown", (e) => {
      const r = $("fig-game").getBoundingClientRect();
      if (e.key === "Enter" && r.top < innerHeight && r.bottom > 0 && document.activeElement?.tagName !== "SELECT") step();
    });
    newGame();
    $("lab-run").addEventListener("click", runLab);
    const lv = () => ($("lab-vol-v").textContent = $("lab-vol").value + "%");
    $("lab-vol").addEventListener("input", lv); lv();
  });
})();
