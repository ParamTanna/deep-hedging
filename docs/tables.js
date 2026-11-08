/* Tables for the page. Like the charts, every cell is read from data/site.json. */

window.renderTables = function (D) {
  const s = D.site;
  const f = (v, n = 3) => (typeof v === "number" && isFinite(v) ? v.toFixed(n) : "n/a");
  const I = ["NIFTY50", "SP500"], IN = { NIFTY50: "NIFTY 50", SP500: "S&amp;P 500" };

  // ---- data quality
  const q = s.data_report;
  simpleTable("tbl-quality", ["Check", "NIFTY 50", "S&amp;P 500"], [
    ["First and last day", ...I.map((i) => `${q[i].first_date} to ${q[i].last_date}`)],
    ["Rows downloaded", ...I.map((i) => q[i].rows_downloaded.toLocaleString("en-US"))],
    ["Duplicate dates dropped", ...I.map((i) => q[i].duplicate_dates_dropped)],
    ["Missing or non-positive close, dropped", ...I.map((i) => q[i].nonpositive_or_missing_close_dropped)],
    ["Rows kept", ...I.map((i) => q[i].rows_kept.toLocaleString("en-US"))],
    ["Adjusted close equals close", ...I.map((i) => (q[i].adjclose_equals_close ? "yes" : "no"))],
    ["Days with zero change (stale quotes)", ...I.map((i) => q[i].zero_change_days)],
    ["Days with zero reported volume", ...I.map((i) => q[i].zero_volume_days.toLocaleString("en-US"))],
    ["Calendar gaps over 5 days", ...I.map((i) => `${q[i].calendar_gaps_over_5_days} (longest ${q[i].largest_gap_days} days, ending ${q[i].largest_gap_ends})`)],
  ], "Every check is computed by src/data.py and src/experiments.py prepare.");

  // ---- stylised facts
  const row = (lab, key, n) => [lab, ...I.flatMap((i) => [f(q[i].pre2019[key], n), f(q[i].post2019[key], n)])];
  simpleTable("tbl-facts",
    ["Daily log returns", "NIFTY, to 2018", "NIFTY, 2019 on", "S&amp;P, to 2018", "S&amp;P, 2019 on"], [
    ["Trading days", ...I.flatMap((i) => [q[i].pre2019.n_days, q[i].post2019.n_days])],
    row("Annualised volatility", "annualised_vol", 3),
    row("Skewness", "skewness", 2),
    row("Excess kurtosis (normal = 0)", "excess_kurtosis", 1),
    row("Ljung-Box, returns, 20 lags", "ljung_box_returns_20", 0),
    row("Ljung-Box, |returns|, 20 lags", "ljung_box_abs_returns_20", 0),
    row("Worst day", "worst_day", 3),
  ], "Ljung-Box 5% critical value at 20 degrees of freedom: 31.4. The 2019-on columns are shown for description only; nothing was fitted on them.");

  // ---- calibration
  const c = s.calibration;
  simpleTable("tbl-calib",
    ["Heston parameter", "NIFTY, daily-moment fit", "NIFTY, horizon fit", "S&amp;P, daily-moment fit", "S&amp;P, horizon fit"], [
    ["kappa (mean reversion, per year)", ...I.flatMap((i) => [f(c.daily[i].kappa, 2), f(c.horizon[i].kappa, 2)])],
    ["theta (long-run variance)", ...I.flatMap((i) => [f(c.daily[i].theta, 4), f(c.horizon[i].theta, 4)])],
    ["xi (vol of vol)", ...I.flatMap((i) => [f(c.daily[i].xi, 2), `<b>${f(c.horizon[i].xi, 2)}</b>`])],
    ["rho (leverage)", ...I.flatMap((i) => [f(c.daily[i].rho, 2), `<b>${f(c.horizon[i].rho, 2)}</b>`])],
    ["Feller condition 2 kappa theta &gt; xi<sup>2</sup>", ...I.flatMap((i) => [c.daily[i].feller_satisfied ? "holds" : "violated", c.horizon[i].feller_satisfied ? "holds" : "violated"])],
    ["p95 of log(realised / quoted), real", ...I.flatMap((i) => ["", f(c.horizon[i].target_moments[2], 3)])],
    ["p95 of log(realised / quoted), model", ...I.flatMap((i) => ["", f(c.horizon[i].fitted_moments[2], 3)])],
  ], "Both fits use only pre-2019 returns. theta is pinned to the sample variance in both. The daily-moment fit is kept for comparison only; every Heston result on this page uses the horizon fit.");

  // ---- headline
  const H = String(s.headline);
  const g = s.gbm[H];
  metricsTable("tbl-headline", g,
    ["unhedged", "bs_delta", "leland", "band", "ww", `gbm_mlp_c${H}_s0`, `gbm_ntb_c${H}_s0`, `gbm_mlpq_c${H}_s0`],
    { hl: [`gbm_mlp_c${H}_s0`, `gbm_ntb_c${H}_s0`],
      caption: `GBM, ${H} bps, 100,000 held-out episodes, P&amp;L per 100 notional. "vs delta" is the CVaR difference from Black-Scholes delta with its paired 95% interval; positive is better. Trades counts changes of holding above 0.1% of notional.` });

  // ---- real data
  const el = document.getElementById("tbl-real");
  if (el) {
    let h = "<table><thead><tr><th>Real episodes sold 2019 on</th>" +
      I.map((i) => `<th>${IN[i]} CVaR<sub>95</sub></th><th class="ci">95% interval</th><th>vs delta</th>`).join("") +
      "</tr></thead><tbody>";
    const ids = ["unhedged", "bs_delta", "leland", "band", "ww",
                 `gbm_mlp_c${H}_s0`, `gbm_ntb_c${H}_s0`, "mlprv", "ntbrv", "gru"];
    for (const id of ids) {
      const cells = I.map((i) => {
        const fam = s.real[i]["test_2019_on|gbm_tuned"];
        const key = ["mlprv", "ntbrv", "gru"].includes(id) ? `hes_${i}_${id}_c${H}_s0` : id;
        const r = fam[key];
        if (!r) return "<td>n/a</td><td></td><td></td>";
        const d = r.cvar95 - fam.bs_delta.cvar95;
        const dci = r.cvar95_minus_bs_delta_ci;
        const sig = dci && (dci[0] > 0 || dci[1] < 0);
        return `<td>${f(r.cvar95)}</td><td class="ci">[${f(r.cvar95_ci[0])}, ${f(r.cvar95_ci[1])}]</td>` +
          `<td>${id === "bs_delta" ? "" : (d >= 0 ? "+" : "") + f(d) + (sig ? "" : " <span class=\"ci\">(n.s.)</span>")}</td>`;
      }).join("");
      const nm = ["mlprv", "ntbrv", "gru"].includes(id) ? NAMES[id] + ", Heston-trained" : label(id.replace(`_c${H}_s0`, `_c${H}_s0`));
      h += `<tr><td>${nm}</td>${cells}</tr>`;
    }
    h += "</tbody><caption>Moving-block bootstrap, 30-day blocks. n.s. means the 95% interval of the difference from delta includes zero. Constant band and Whalley-Wilmott use the parameters tuned on GBM validation; the Heston-tuned variants are in the repository's results/metrics.json.</caption></table>";
    el.innerHTML = h;
  }

  // ---- ablations
  const ab = [];
  const base = g.bs_delta.cvar95;
  const add = (lab, id) => { if (g[id]) ab.push([lab, f(g[id].cvar95), `<span class="ci">[${f(g[id].cvar95_ci[0])}, ${f(g[id].cvar95_ci[1])}]</span>`, f(g[id].cvar95 - base), f(g[id].mean_cost)]); };
  add("MLP, width 64, seed 0 (main)", `gbm_mlp_c${H}_s0`);
  add("MLP, seed 1", `gbm_mlp_c${H}_s1`);
  add("MLP, seed 2", `gbm_mlp_c${H}_s2`);
  add("Band net, seed 0 (main)", `gbm_ntb_c${H}_s0`);
  add("Band net, seed 1", `gbm_ntb_c${H}_s1`);
  add("Band net, seed 2", `gbm_ntb_c${H}_s2`);
  for (const w of [8, 16, 32, 128]) add(`MLP, width ${w}`, `gbm_mlp_w${w}_c${H}_s0`);
  add("MLP, quadratic objective", `gbm_mlpq_c${H}_s0`);
  simpleTable("tbl-ablation", ["Variant (GBM, 10 bps)", "CVaR<sub>95</sub>", "95% interval", "vs delta", "Mean fees"], ab,
    "Seeds change the network initialisation and the stream of training episodes; validation and test sets are identical across all rows.");

  // ---- worst episodes
  const wt = [];
  for (const i of I) {
    const e = s.errors && s.errors[i];
    if (!e) continue;
    for (const w of e.strategies.bs_delta.worst.slice(0, 5)) {
      wt.push([IN[i], w.date, f(w.pnl, 2), f(w.hedge_error, 2), f(w.fee, 2),
               (100 * w.sigma_imp).toFixed(1) + "%", (100 * w.rv).toFixed(1) + "%", f(w.vol_ratio, 2),
               (w.move_pct >= 0 ? "+" : "") + w.move_pct.toFixed(1) + "%"]);
    }
  }
  simpleTable("tbl-worst", ["Index", "Sold on", "P&amp;L", "Hedging error", "Fees", "Quoted vol", "Realised vol", "Ratio", "Index move"], wt,
    "The five worst real test episodes for Black-Scholes delta on each index. P&amp;L = hedging error minus fees. Overlapping windows mean neighbouring dates often describe the same market event.");

  // ---- the sim-to-real ladder
  if (s.derived && s.derived.sim2real) {
    const body = [];
    for (const i of I) {
      const rows = s.derived.sim2real[i];
      const ids = Object.keys(rows).filter((k) => k !== "unhedged");
      for (const k of ids) {
        const r = rows[k];
        body.push([`${IN[i]}`, label(k), f(r.gbm_sim), f(r.heston_sim), f(r.real_calib), f(r.real_test)]);
      }
    }
    simpleTable("tbl-sim2real", ["Index", "Strategy", "GBM, simulated", "Heston, simulated",
      "Real, calibration era", "Real, 2019 on"], body,
      `CVaR<sub>95</sub> at ${H} bps. Left to right: the world the policy trained in, then reality. The step from Heston to the calibration era is model error, since Heston was fitted to that era; the step on to 2019 adds a change of regime. Heston-trained policies have no GBM or other-index entry where they were not evaluated.`);
  }

  // ---- volatility-regime buckets on the simulated test set
  if (s.regimes) {
    const ids = ["bs_delta", "ww", `gbm_mlp_c${H}_s0`, `gbm_ntb_c${H}_s0`].filter((k) => s.regimes.strategies[k]);
    const e = s.regimes.edges;
    const body = [0, 1, 2, 3, 4].map((g) => [
      `${["quietest", "quiet", "middle", "rough", "roughest"][g]} fifth <span class="ci">(${f(e[g], 2)} to ${f(e[g + 1], 2)})</span>`,
      ...ids.map((k) => f(s.regimes.strategies[k][g].mean, 3)),
      ...ids.map((k) => f(s.regimes.strategies[k][g].fee, 3))]);
    simpleTable("tbl-regimes", ["Paths, by realised / quoted vol",
      ...ids.map((k) => "Mean P&amp;L, " + label(k)), ...ids.map((k) => "Fees, " + label(k))], body,
      `GBM, ${H} bps, the 100,000 test paths split into five equal groups by how much realised volatility exceeded the quote.`);
  }

  // ---- strategy picker for the scatter
  const sel = document.getElementById("vs-strat");
  if (sel && s.errors) {
    const ids = Object.keys(s.errors.NIFTY50.strategies);
    const cur = sel.value;
    const gen = ids.map((k) => k.replace("NIFTY50", "IDX"));
    sel.innerHTML = gen.map((k) => `<option value="${k}">${label(k)}</option>`).join("");
    sel.value = gen.includes(cur) ? cur : "bs_delta";
  }
};
