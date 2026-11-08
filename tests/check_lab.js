// Does docs/lab.js reproduce the Python engine and policies exactly?
// Run: python tests/make_lab_fixture.py && node tests/check_lab.js
const fs = require("fs"), path = require("path");
require(path.join(__dirname, "..", "docs", "lab.js"));
const L = globalThis.Lab;
const F = JSON.parse(fs.readFileSync(path.join(__dirname, "lab_fixture.json"), "utf8"));
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
let ok = true;
const check = (name, d, tol) => {
  const pass = d <= tol; ok = ok && pass;
  console.log(`${pass ? "ok  " : "FAIL"} ${name.padEnd(34)} max abs diff ${d.toExponential(2)}  (tol ${tol})`);
};

check("erf on [-6, 6]", maxDiff(F.erf_x.map(L.erf), F.erf_y), 1e-14);
check("Black-Scholes price", maxDiff(F.bs_S.map((s) => {
  const t = 0.05; const r = L.bs(s, t, 0.22); return r.price; }), F.bs_p), 1e-11);
check("Black-Scholes delta", maxDiff(F.bs_S.map((s) => L.bs(s, 0.05, 0.22).delta), F.bs_d), 1e-13);
check("Black-Scholes gamma", maxDiff(F.bs_S.map((s) => L.bs(s, 0.05, 0.22).gamma), F.bs_g), 1e-13);

const P = F.paths, sg = F.sigmas, pr = F.premium;
const run = (fn) => P.map((S, i) => L.pnl(S, fn(S, sg[i]), F.cost, pr[i]).total);
check("P&L, Black-Scholes delta", maxDiff(run((S, s) => L.deltaH(S, s)), F.pnl_delta), 1e-10);
check("P&L, Whalley-Wilmott", maxDiff(run((S, s) => L.wwH(S, s, F.cost, F.ww_gamma)), F.pnl_ww), 1e-10);
for (const k of ["mlp", "ntb"]) {
  if (!F["w_" + k]) { console.log(`skip ${k} (not trained yet)`); continue; }
  const model = F["w_" + k];
  const H0 = P.slice(0, 5).map((S, i) => L.policyH(model, S, sg[i]));
  check(`holdings, ${k}`, Math.max(...H0.map((h, i) => maxDiff(h, F["H_" + k][i]))), 1e-12);
  check(`P&L, ${k} network`, maxDiff(run((S, s) => L.policyH(model, S, s)), F["pnl_" + k]), 1e-10);
}

// The file the website actually ships (9 significant digits) must agree too.
const shipped = path.join(__dirname, "..", "docs", "data", "models.json");
if (fs.existsSync(shipped)) {
  const M = JSON.parse(fs.readFileSync(shipped, "utf8")).models;
  for (const k of ["mlp", "ntb"]) {
    const key = `${k}_${F.cost}`;
    if (!M[key] || !F["pnl_" + k]) continue;
    const d = maxDiff(P.map((S, i) => L.pnl(S, L.policyH(M[key], S, sg[i]), F.cost, pr[i]).total), F["pnl_" + k]);
    const pass = d < 1e-6; ok = ok && pass;
    console.log(`${pass ? "ok  " : "FAIL"} ${("shipped weights, " + k).padEnd(34)} max abs diff ${d.toExponential(2)}  (tol 1e-6)`);
  }
}

console.log(ok ? "\nBROWSER PORT MATCHES PYTHON" : "\n*** MISMATCH ***");
process.exit(ok ? 0 : 1);
