"""
Reproducible analysis behind the paper (paper/conference and paper/extended).

Every number quoted in the manuscripts is written to results/paper/numbers.json
and every figure to results/paper/figures/.  Run from the repository root:

    python3 analysis/paper_analysis.py

Requires numpy, pandas, scipy, scikit-learn, matplotlib.  The heavy-hex
coupling map of ibm_marrakesh (results/paper/marrakesh_topology.json) was
exported once from qiskit_ibm_runtime.fake_provider.FakeMarrakesh.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "paper")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

NUM = {}  # every number cited in the paper

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
C1, C2, C3, C4, C5 = "#1f4e79", "#c0504d", "#4f8a3c", "#8064a2", "#d08a2e"
W1 = 3.4   # single IEEE column (inches)
W2 = 7.0   # full width


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def r(x, n=4):
    return float(np.round(x, n))


# ================================================================ 1. simulator
def qber_simulator_closed_form(pr, pd_, pro):
    """Exact expectation of the Monte-Carlo simulator in notebooks/03.

    Z-basis rounds: |1> flips with p_relax.  X-basis rounds: the encoded bit
    flips with p_relax (if 1) and then with p_dephase.  Readout flips any
    outcome with p_readout.  Averaging over bits and the two sifted bases:
        QBER = p_ro + (1 - 2 p_ro) (p_r + p_d - p_r p_d) / 2
    """
    return pro + (1 - 2 * pro) * (pr + pd_ - pr * pd_) / 2


def qber_quantum_channel(gamma, lam, pro):
    """Expected BB84 QBER for an amplitude-damping (gamma) followed by
    phase-damping (lam) channel and symmetric readout error pro.

    Z basis:  e_Z = gamma/2.
    X basis:  coherence shrinks by sqrt((1-gamma)(1-lam)), so
              e_X = (1 - sqrt((1-gamma)(1-lam))) / 2.
    """
    e_z = gamma / 2
    e_x = (1 - np.sqrt((1 - gamma) * (1 - lam))) / 2
    e = (e_z + e_x) / 2
    return pro + (1 - 2 * pro) * e


sim = pd.read_csv(os.path.join(ROOT, "data", "generated", "qber_dataset_full.csv"))
X = sim[["p_relax", "p_dephase", "p_readout"]].values
y = sim["avg_qber"].values
y_cf = qber_simulator_closed_form(*X.T)

# expected Monte-Carlo noise of each label (10 runs x ~N/2 = 500 sifted bits)
sigma_mc = np.sqrt(y_cf * (1 - y_cf) / (10 * 500))
NUM["sim"] = {
    "n_samples": int(len(sim)),
    "p_relax_range": [r(sim.p_relax.min()), r(sim.p_relax.max())],
    "p_dephase_range": [r(sim.p_dephase.min()), r(sim.p_dephase.max())],
    "p_readout_range": [r(sim.p_readout.min()), r(sim.p_readout.max())],
    "qber_range": [r(y.min()), r(y.max())],
    "qber_mean": r(y.mean()),
    "closed_form_r2": r(r2_score(y, y_cf), 5),
    "closed_form_rmse": r(np.sqrt(mean_squared_error(y, y_cf)), 5),
    "closed_form_mae": r(mean_absolute_error(y, y_cf), 5),
    "mc_noise_rms": r(np.sqrt(np.mean(sigma_mc ** 2)), 5),
    "r2_ceiling": r(1 - np.mean(sigma_mc ** 2) / y.var(), 5),
    "frac_above_11pct": r((y > 0.11).mean(), 3),
}
# standardized residuals should be ~N(0,1) if the formula is exact
z = (y - y_cf) / sigma_mc
NUM["sim"]["std_residual_mean"] = r(z.mean(), 3)
NUM["sim"]["std_residual_std"] = r(z.std(), 3)

# Physically, relaxation acting on |-> is not a classical bit flip.
# Quantify the over-estimate of the classical model over the parameter box.
g = np.linspace(0, 0.3, 61)
G, L = np.meshgrid(g, g)
diff = qber_simulator_closed_form(G, L, 0) - qber_quantum_channel(G, L, 0)
NUM["sim"]["classical_minus_quantum_max"] = r(diff.max())
NUM["sim"]["classical_minus_quantum_at_0.1_0.1"] = r(
    qber_simulator_closed_form(0.1, 0.1, 0) - qber_quantum_channel(0.1, 0.1, 0))


# ---- ML models: reproduce notebook 04 and add 5-fold CV
def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "MLP (64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
ml = {}
for name, m in make_models().items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    cv = cross_val_predict(make_models()[name], X, y, cv=KFold(5, shuffle=True, random_state=0))
    folds = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        mm = make_models()[name].fit(X[tr], y[tr])
        folds.append(r2_score(y[te], mm.predict(X[te])))
    ml[name] = {
        "test_mae": r(mean_absolute_error(yte, p), 5),
        "test_rmse": r(np.sqrt(mean_squared_error(yte, p)), 5),
        "test_r2": r(r2_score(yte, p), 4),
        "cv_r2_mean": r(np.mean(folds), 4),
        "cv_r2_std": r(np.std(folds), 4),
        "cv_mae": r(mean_absolute_error(y, cv), 5),
    }
pcf = qber_simulator_closed_form(*Xte.T)
ml["Closed form (no fitting)"] = {
    "test_mae": r(mean_absolute_error(yte, pcf), 5),
    "test_rmse": r(np.sqrt(mean_squared_error(yte, pcf)), 5),
    "test_r2": r(r2_score(yte, pcf), 4),
    "cv_r2_mean": None, "cv_r2_std": None, "cv_mae": r(mean_absolute_error(y, y_cf), 5),
}
lr_full = LinearRegression().fit(X, y)
NUM["ml_sim"] = ml
NUM["lr_coefficients"] = {"w_relax": r(lr_full.coef_[0]), "w_dephase": r(lr_full.coef_[1]),
                          "w_readout": r(lr_full.coef_[2]), "b": r(lr_full.intercept_, 5)}

# MLP with standardized inputs (the original MLP was fed raw, unscaled inputs)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
mlp_s = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000,
                                                       random_state=42, alpha=1e-4))
mlp_s.fit(Xtr, ytr)
NUM["ml_sim"]["MLP (64-32), standardized"] = {
    "test_mae": r(mean_absolute_error(yte, mlp_s.predict(Xte)), 5),
    "test_rmse": r(np.sqrt(mean_squared_error(yte, mlp_s.predict(Xte))), 5),
    "test_r2": r(r2_score(yte, mlp_s.predict(Xte)), 4),
}

# Figure: simulator parity
fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.62, 2.1))
ax[0].scatter(y_cf, y, s=3, c=C1, alpha=0.5, lw=0)
lim = [0, max(y.max(), y_cf.max()) * 1.03]
ax[0].plot(lim, lim, "k--", lw=0.7)
ax[0].set_xlabel("closed-form QBER, Eq. (S)")
ax[0].set_ylabel("simulated QBER (label)")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].text(0.04, 0.93, f"$R^2$ = {NUM['sim']['closed_form_r2']:.4f}", transform=ax[0].transAxes)
ax[1].hist(z, bins=40, density=True, color=C1, alpha=0.75)
zz = np.linspace(-4, 4, 200)
ax[1].plot(zz, stats.norm.pdf(zz), color=C2, lw=1)
ax[1].set_xlabel(r"standardized residual $(y-\hat y_{\rm cf})/\sigma_{\rm MC}$")
ax[1].set_ylabel("density")
for a, t in zip(ax, "ab"):
    a.text(-0.2, 1.02, f"({t})", transform=a.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_simulator_closed_form")

# ================================================================ 2. calibration
cal = pd.read_csv(os.path.join(ROOT, "data", "hardware", "ibm_marrakesh_calibration_timeseries.csv"),
                  parse_dates=["timestamp"])
ro = pd.read_csv(os.path.join(ROOT, "data", "hardware", "ibm_marrakesh_readout_timeseries.csv"),
                 parse_dates=["timestamp"])
st = pd.read_csv(os.path.join(ROOT, "data", "hardware", "ibm_marrakesh_qubit_calibration.csv"))
sro = pd.read_csv(os.path.join(ROOT, "data", "hardware", "ibm_marrakesh_readout_errors.csv"))
static = st.merge(sro, on="qubit")

topo = json.load(open(os.path.join(OUT, "marrakesh_topology.json")))
edges = [tuple(e) for e in topo["edges"]]
dur = topo["durations_s"]


def desc(s):
    return {"mean": r(s.mean(), 4), "median": r(s.median(), 4), "std": r(s.std(), 4),
            "min": r(s.min(), 4), "max": r(s.max(), 4),
            "q25": r(s.quantile(.25), 4), "q75": r(s.quantile(.75), 4)}


NUM["static_snapshot"] = {
    "n_qubits": int(len(static)),
    "t1_us": desc(static.t1_us), "t2_us": desc(static.t2_us),
    "readout": desc(static.readout_error),
    "n_t2_gt_t1": int((static.t2_us > static.t1_us).sum()),
    "n_t2_gt_2t1": int((static.t2_us > 2 * static.t1_us).sum()),
    "n_readout_gt_0.05": int((static.readout_error > 0.05).sum()),
    "n_readout_gt_0.11": int((static.readout_error > 0.11).sum()),
    "frequency_available": bool(st.frequency_ghz.notna().any()),
    "n_edges": len(edges),
    "durations_ns": {k: (None if v is None else r(v * 1e9, 1)) for k, v in dur.items()},
}
ts_cal = np.sort(cal.timestamp.unique())
gaps = np.diff(ts_cal) / np.timedelta64(1, "h")
pv_t1 = cal.pivot_table(index="timestamp", columns="qubit", values="t1_us")
pv_t2 = cal.pivot_table(index="timestamp", columns="qubit", values="t2_us")
pv_ro = ro.pivot_table(index="timestamp", columns="qubit", values="readout_error")
cv_t1 = (pv_t1.std() / pv_t1.mean())
cv_t2 = (pv_t2.std() / pv_t2.mean())
cv_ro = (pv_ro.std() / pv_ro.mean())
# number of distinct calibration values per qubit -> how often IBM re-calibrated
n_unique = pv_t1.nunique()
NUM["timeseries"] = {
    "n_snapshots": int(len(ts_cal)),
    "n_records": int(len(cal)),
    "start": str(pd.Timestamp(ts_cal[0]))[:16], "end": str(pd.Timestamp(ts_cal[-1]))[:16],
    "span_days": r((ts_cal[-1] - ts_cal[0]) / np.timedelta64(1, "D"), 1),
    "median_gap_h": r(np.median(gaps), 2),
    "distinct_t1_values_per_qubit_median": int(n_unique.median()),
    "cv_t1_median": r(cv_t1.median(), 3), "cv_t1_q90": r(cv_t1.quantile(.9), 3),
    "cv_t2_median": r(cv_t2.median(), 3), "cv_ro_median": r(cv_ro.median(), 3),
    "t1_min_over_time_us": r(pv_t1.min().min(), 2),
    "t1_max_over_time_us": r(pv_t1.max().max(), 1),
    "median_t1_over_time_min": r(pv_t1.median(axis=1).min(), 1),
    "median_t1_over_time_max": r(pv_t1.median(axis=1).max(), 1),
    "median_ro_over_time_min": r(pv_ro.median(axis=1).min(), 4),
    "median_ro_over_time_max": r(pv_ro.median(axis=1).max(), 4),
    "ratio_t1_max_min_per_qubit_median": r((pv_t1.max() / pv_t1.min()).median(), 2),
}

# Top-10 stability by readout error across snapshots
latest = pv_ro.iloc[-1]
top_sets = [set(row.nsmallest(10).index) for _, row in pv_ro.iterrows()]
first = top_sets[0]
jac_first = [len(first & s) / len(first | s) for s in top_sets]
jac_consec = [len(a & b) / len(a | b) for a, b in zip(top_sets[:-1], top_sets[1:])]
tops = pd.Series(top_sets, index=pv_ro.index)
lag_rows = []
for i in range(len(tops)):
    for j in range(i + 1, len(tops)):
        a_, b_ = tops.iloc[i], tops.iloc[j]
        lag_rows.append(((tops.index[j] - tops.index[i]) / pd.Timedelta(days=1), len(a_ & b_) / len(a_ | b_)))
lag_df = pd.DataFrame(lag_rows, columns=["lag_d", "jac"])
lag_df["bin"] = pd.cut(lag_df.lag_d, np.arange(0, 26, 1.0))
lag_curve = lag_df.groupby("bin", observed=True).jac.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
lag_curve.columns = ["med", "q25", "q75"]
lag_mid = np.array([b.mid for b in lag_curve.index])
NUM["ranking_lag"] = {"jaccard_median_lag_0_1d": r(lag_curve.med.iloc[0], 3),
                      "jaccard_median_lag_1_2d": r(lag_curve.med.iloc[1], 3),
                      "jaccard_median_lag_6_7d": r(lag_curve.med.iloc[6], 3),
                      "jaccard_median_lag_20plus": r(lag_curve.med.iloc[20:].median(), 3)}
NUM["ranking"] = {
    "jaccard_top10_vs_first_final": r(jac_first[-1], 3),
    "jaccard_top10_consecutive_median": r(np.median(jac_consec), 3),
    "n_distinct_qubits_ever_in_top10": int(len(set().union(*top_sets))),
}

# Figure: calibration distributions
fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9))
ax[0].hist(static.t1_us, bins=25, color=C1, alpha=.8, label=r"$T_1$")
ax[0].hist(static.t2_us, bins=25, color=C2, alpha=.55, label=r"$T_2$")
ax[0].set_xlabel(r"time ($\mu$s)"); ax[0].set_ylabel("qubits"); ax[0].legend(frameon=False)
ax[1].scatter(static.t1_us, static.t2_us, s=5, c=C1, lw=0)
tt = np.linspace(0, static.t1_us.max() * 1.05, 10)
ax[1].plot(tt, 2 * tt, "k--", lw=.7, label=r"$T_2=2T_1$")
ax[1].plot(tt, tt, ":", c="gray", lw=.7, label=r"$T_2=T_1$")
ax[1].set_xlabel(r"$T_1$ ($\mu$s)"); ax[1].set_ylabel(r"$T_2$ ($\mu$s)")
ax[1].set_ylim(0, static.t2_us.max() * 1.1); ax[1].legend(frameon=False, loc="upper left")
ax[2].hist(np.log10(static.readout_error), bins=25, color=C3, alpha=.85)
ax[2].axvline(np.log10(0.11), color=C2, ls="--", lw=.8)
ax[2].text(np.log10(0.11) + 0.03, ax[2].get_ylim()[1] * .8, "11%", color=C2, fontsize=7)
ax[2].set_xlabel(r"$\log_{10}\,\varepsilon_{\rm ro}$"); ax[2].set_ylabel("qubits")
for a, t in zip(ax, "abc"):
    a.text(-0.22, 1.02, f"({t})", transform=a.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_calibration")

# Figure: drift
fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9))
days = (pv_t1.index - pv_t1.index[0]) / pd.Timedelta(days=1)
show = cv_t1.sort_values().index[[10, 60, 100, 140, 150]]
for q, c in zip(show, [C1, C2, C3, C4, C5]):
    ax[0].plot(days, pv_t1[q], c=c, lw=.8, label=f"Q{q}")
ax[0].set_xlabel("days since first snapshot"); ax[0].set_ylabel(r"$T_1$ ($\mu$s)")
ax[0].legend(frameon=False, ncol=2, fontsize=6)
ax[1].hist(cv_t1, bins=25, color=C1, alpha=.8, label=r"$T_1$")
ax[1].hist(cv_ro, bins=25, color=C3, alpha=.55, label=r"$\varepsilon_{\rm ro}$")
ax[1].set_xlabel("coefficient of variation over 25 days"); ax[1].set_ylabel("qubits")
ax[1].legend(frameon=False)
dro = (pv_ro.index - pv_ro.index[0]) / pd.Timedelta(days=1)
ax[2].fill_between(lag_mid, lag_curve.q25, lag_curve.q75, color=C1, alpha=.2, lw=0)
ax[2].plot(lag_mid, lag_curve.med, "o-", c=C1, ms=2.5, lw=.9)
ax[2].set_ylim(0, 1.0); ax[2].set_xlabel("time between snapshots (days)")
ax[2].set_ylabel(r"Jaccard overlap of top-10 sets")
for a, t in zip(ax, "abc"):
    a.text(-0.22, 1.02, f"({t})", transform=a.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_drift")


# ================================================================ 3. lattice
def heavy_hex_coords(edges, n=156):
    """Planar coordinates for the Heron r2 heavy-hex lattice: 8 rows of 16
    qubits (indices 20k..20k+15) linked by 7 rows of 4 bridge qubits."""
    nb = {q: [] for q in range(n)}
    for a, b in edges:
        nb[a].append(b); nb[b].append(a)
    pos = {}
    for q in range(n):
        k, o = divmod(q, 20)
        if o < 16:
            pos[q] = (o, -2 * k)
    for q in range(n):
        if q not in pos:
            up = min(nb[q])
            pos[q] = (pos[up][0], pos[up][1] - 1)
    return pos


POS = heavy_hex_coords(edges)


def draw_lattice(ax, values, cmap, norm, label, mark=None, dark_low=True):
    for a, b in edges:
        (x1, y1), (x2, y2) = POS[a], POS[b]
        ax.plot([x1, x2], [y1, y2], c="#9a9a9a", lw=0.8, zorder=1)
    xs = [POS[q][0] for q in range(156)]
    ys = [POS[q][1] for q in range(156)]
    sc = ax.scatter(xs, ys, c=[values[q] for q in range(156)], cmap=cmap, norm=norm, s=46,
                    edgecolors="k", linewidths=0.35, zorder=2)
    for q in range(156):
        ax.text(POS[q][0], POS[q][1], str(q), ha="center", va="center", fontsize=3.4, zorder=3,
                color="white" if ((norm(values[q]) < 0.35) if dark_low else (norm(values[q]) > 0.6)) else "black")
    if mark is not None:
        for q in mark:
            ax.scatter([POS[q][0]], [POS[q][1]], s=110, facecolors="none", edgecolors=C2,
                       linewidths=1.0, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label); cb.ax.tick_params(labelsize=6)


s_idx = static.set_index("qubit")
fig, ax = plt.subplots(2, 1, figsize=(W2, 6.0))
draw_lattice(ax[0], s_idx.t1_us, "viridis", Normalize(0, 420), r"$T_1$ ($\mu$s)")
draw_lattice(ax[1], s_idx.readout_error, "magma_r", LogNorm(3e-3, 0.5), r"readout error $\varepsilon_{\rm ro}$",
             mark=s_idx.readout_error.nlargest(10).index, dark_low=False)
ax[0].set_title("(a) energy-relaxation time per qubit", loc="left")
ax[1].set_title("(b) readout assignment error per qubit (ten worst circled)", loc="left")
fig.tight_layout()
save(fig, "fig_lattice")

# compact single-panel version for the short paper
fig, ax = plt.subplots(figsize=(W2, 2.75))
draw_lattice(ax, s_idx.readout_error, "magma_r", LogNorm(3e-3, 0.5), r"readout error $\varepsilon_{\rm ro}$",
             mark=s_idx.readout_error.nlargest(10).index, dark_low=False)
fig.tight_layout()
save(fig, "fig_lattice_ro")

# ================================================================ 4. hardware BB84
# 4a. original five targeted runs (N = 100, 500, 1000)
runs = {}
for n in (100, 500, 1000):
    f = os.path.join(ROOT, "results", "hardware", f"N_{n}",
                     f"hardware_qber_summary_{n}.csv" if n == 100 else f"hardware_qber_summary_{n}.csv")
    runs[n] = pd.read_csv(f)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.clip(c - h, 0, None), c + h


fs = []
for n, d in runs.items():
    k, m = d.errors.sum(), d.sifted_length.sum()
    lo, hi = wilson(k, m)
    fs.append({"N": n, "runs": len(d), "sifted": int(m), "errors": int(k), "pooled_qber": r(k / m, 5),
               "ci95": [r(lo, 5), r(hi, 5)], "mean_qber": r(d.qber.mean(), 5),
               "std_qber": r(d.qber.std(), 5), "sifted_frac": r(m / (n * len(d)), 3)})
NUM["targeted_runs"] = fs

# the per-run sifted fraction should be Binomial(N, 1/2)
all_sift = pd.concat([d.assign(N=n) for n, d in runs.items()])
NUM["targeted_runs_sift_fraction_mean"] = r((all_sift.sifted_length / all_sift.N).mean(), 3)

# 4b. reproduce the original inference pipeline (MinMax-scaled 1/T1, 1/T2)
full_models = {k: v.fit(X, y) for k, v in make_models().items()}


def original_features(snap):
    """Feature map used in src/ml/run_inference.py."""
    d = snap.copy()
    sc = MinMaxScaler().fit_transform(np.c_[1 / d.t1_us, 1 / d.t2_us])
    d["p_relax"], d["p_dephase"], d["p_readout"] = sc[:, 0], sc[:, 1], d.readout_error
    return d


def physical_features(snap, tau_s=dur["x"]):
    """Physically motivated map: error probabilities accumulated during the
    state-preparation gate of duration tau (readout dynamics are already
    included in the reported readout error)."""
    d = snap.copy()
    t1 = d.t1_us * 1e-6
    t2 = d.t2_us * 1e-6
    inv_tphi = np.clip(1 / t2 - 1 / (2 * t1), 0, None)
    d["p_relax"] = 1 - np.exp(-tau_s / t1)
    d["p_dephase"] = 1 - np.exp(-tau_s * inv_tphi)
    d["p_readout"] = d.readout_error
    return d


of = original_features(static)
F = ["p_relax", "p_dephase", "p_readout"]
for name, m in full_models.items():
    of["pred_" + name] = m.predict(of[F].values)
stored = pd.read_csv(os.path.join(ROOT, "results", "hardware", "predicted_qber_per_qubit.csv"))
NUM["inference_reproduction_max_abs_diff_rf"] = r(
    np.abs(of.sort_values("qubit")["pred_Random Forest"].values - stored.sort_values("qubit").predicted_qber_rf.values).max(), 6)
NUM["original_inference"] = {
    name: desc(of["pred_" + name]) for name in full_models}
NUM["original_inference_features"] = {
    "frac_p_relax_outside_train": r((of.p_relax > 0.3).mean(), 3),
    "frac_p_dephase_outside_train": r((of.p_dephase > 0.3).mean(), 3),
    "frac_any_outside_train": r(((of.p_relax > 0.3) | (of.p_dephase > 0.3) | (of.p_readout > 0.1)).mean(), 3),
    "n_pred_above_11pct_rf": int((of["pred_Random Forest"] > 0.11).sum()),
    "n_pred_lr_above_1": int((of["pred_Linear Regression"] > 1).sum()),
}
phf = physical_features(static)
NUM["physical_features"] = {
    "tau_ns": r(dur["x"] * 1e9, 1),
    "p_relax_median": float(f"{phf.p_relax.median():.2e}"),
    "p_relax_max": float(f"{phf.p_relax.max():.2e}"),
    "p_dephase_median": float(f"{phf.p_dephase.median():.2e}"),
    "p_dephase_max": float(f"{phf.p_dephase.max():.2e}"),
}

# Figure: domain shift
fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.62, 2.0))
ax[0].add_patch(plt.Rectangle((0, 0), 0.3, 0.3, fc=C1, alpha=.12, ec=C1, lw=.8))
ax[0].text(0.32, 0.02, "training box", color=C1, fontsize=6.5)
ax[0].scatter(of.p_relax, of.p_dephase, s=6, c=C2, lw=0, label="hardware, MinMax map")
ax[0].scatter(phf.p_relax, phf.p_dephase, s=6, c=C3, marker="s", lw=0, label="hardware, physical map")
ax[0].set_xlabel(r"$p_{\rm relax}$"); ax[0].set_ylabel(r"$p_{\rm dephase}$")
ax[0].set_xlim(-0.03, 1.03); ax[0].set_ylim(-0.03, 1.03)
ax[0].legend(frameon=False, loc="upper right", fontsize=6)
ax[1].scatter(static.t2_us, of["pred_Random Forest"], s=6, c=C2, lw=0)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"$T_2$ ($\mu$s)"); ax[1].set_ylabel("RF-predicted QBER (original)")
for a, t in zip(ax, "ab"):
    a.text(-0.24, 1.02, f"({t})", transform=a.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_domain_shift")

# 4c. targeted N = 1000 runs vs predictors (static snapshot)
v = runs[1000].merge(of[["qubit", "t1_us", "t2_us", "readout_error"] + ["pred_" + k for k in full_models]],
                     left_on="physical_qubit", right_on="qubit")
v["lo"], v["hi"] = wilson(v.errors, v.sifted_length)
tgt = []
for _, rr in v.iterrows():
    tgt.append({"qubit": int(rr.qubit), "t1": r(rr.t1_us, 1), "t2": r(rr.t2_us, 1),
                "ro": r(rr.readout_error, 4), "sifted": int(rr.sifted_length), "errors": int(rr.errors),
                "qber": r(rr.qber, 4), "ci95": [r(rr.lo, 4), r(rr.hi, 4)],
                "pred_rf": r(rr["pred_Random Forest"], 4), "pred_lr": r(rr["pred_Linear Regression"], 4),
                "pred_nn": r(rr["pred_MLP (64-32)"], 4)})
NUM["targeted_1000"] = sorted(tgt, key=lambda d: d["qber"])
NUM["targeted_1000_summary"] = {
    "spearman_rf": r(stats.spearmanr(v["pred_Random Forest"], v.qber)[0], 3),
    "spearman_ro": r(stats.spearmanr(v.readout_error, v.qber)[0], 3),
    "mae_rf": r(mean_absolute_error(v.qber, v["pred_Random Forest"]), 4),
    "mae_ro": r(mean_absolute_error(v.qber, v.readout_error), 4),
}

# 4d. longitudinal benchmark campaign (random qubits, N = 1000 each)
bh = pd.read_csv(os.path.join(ROOT, "data", "timeseries", "benchmark_history.csv"), parse_dates=["timestamp"])
pend = pd.read_csv(os.path.join(ROOT, "data", "jobs", "pending_jobs.csv"))


def asof_join(bench, q_col):
    """Attach the most recent calibration snapshot taken before collection."""
    out = []
    c = cal.sort_values("timestamp"); rr = ro.sort_values("timestamp")
    for _, row in bench.iterrows():
        rec = row.to_dict()
        for q, tag in q_col(row):
            cq = c[(c.qubit == q) & (c.timestamp <= row.timestamp)]
            rq = rr[(rr.qubit == q) & (rr.timestamp <= row.timestamp)]
            cq = cq.iloc[-1] if len(cq) else c[c.qubit == q].iloc[0]
            rq = rq.iloc[-1] if len(rq) else rr[rr.qubit == q].iloc[0]
            rec[f"t1{tag}"], rec[f"t2{tag}"] = cq.t1_us, cq.t2_us
            rec[f"ro{tag}"] = rq.readout_error
            rec[f"lag_h{tag}"] = (row.timestamp - rq.timestamp) / pd.Timedelta(hours=1)
            # whole-chip snapshot for the MinMax feature map
            rec[f"snap{tag}"] = cq.timestamp
        out.append(rec)
    return pd.DataFrame(out)


bb = bh[bh.benchmark == "BB84"].copy()
bb["q"] = bb.physical_qubit.astype(int)
bb = asof_join(bb, lambda row: [(row.q, "")])
bb["qber"] = bb.metric_value
SIFT = 500  # sifted length is not logged for the campaign; E[sifted] = N/2
bb["lo"], bb["hi"] = wilson(np.round(bb.qber * SIFT), SIFT)

# original pipeline applied snapshot-by-snapshot
rows = []
for snap, grp in bb.groupby("snap"):
    sn = cal[cal.timestamp == snap].merge(
        ro.assign(dt=(ro.timestamp - snap).abs()).sort_values("dt").drop_duplicates("qubit")[["qubit", "readout_error"]],
        on="qubit")
    f = original_features(sn).set_index("qubit")
    for i, row in grp.iterrows():
        x = f.loc[row.q, F].values.astype(float).reshape(1, -1)
        rows.append((i, *[m.predict(x)[0] for m in full_models.values()]))
pr = pd.DataFrame(rows, columns=["i"] + ["orig_" + k for k in full_models]).set_index("i")
bb = bb.join(pr)
ph = physical_features(bb.rename(columns={"t1": "t1_us", "t2": "t2_us", "ro": "readout_error"}))
bb["phys_cf"] = qber_simulator_closed_form(ph.p_relax, ph.p_dephase, ph.p_readout)

# Hardware-trained models (grouped CV by qubit so repeated qubits never leak)
bb["inv_t1"], bb["inv_t2"] = 1 / bb.t1, 1 / bb.t2
groups = bb.q.values
gkf = GroupKFold(n_splits=10)
bb["hw_lin_ro"] = cross_val_predict(LinearRegression(), bb[["ro"]], bb.qber, cv=gkf, groups=groups)
bb["hw_lin_all"] = cross_val_predict(LinearRegression(), bb[["ro", "inv_t1", "inv_t2"]], bb.qber, cv=gkf, groups=groups)
bb["hw_rf_all"] = cross_val_predict(RandomForestRegressor(200, min_samples_leaf=3, random_state=0),
                                    bb[["ro", "t1", "t2"]], bb.qber, cv=gkf, groups=groups)

preds = {
    "Original RF (sim.-trained, MinMax features)": "orig_Random Forest",
    "Original LR (sim.-trained, MinMax features)": "orig_Linear Regression",
    "Original MLP (sim.-trained, MinMax features)": "orig_MLP (64-32)",
    r"Physics baseline: QBER $=\varepsilon_{\rm ro}$": "ro",
    "Closed form, physical feature map": "phys_cf",
    r"Hardware-fitted linear, $\varepsilon_{\rm ro}$ only (CV)": "hw_lin_ro",
    r"Hardware-fitted linear, $\varepsilon_{\rm ro},1/T_1,1/T_2$ (CV)": "hw_lin_all",
    r"Hardware-fitted RF, $\varepsilon_{\rm ro},T_1,T_2$ (CV)": "hw_rf_all",
}
ABORT = 0.11
res = {}
for name, col in preds.items():
    p = bb[col].values
    tp = int(((p > ABORT) & (bb.qber > ABORT)).sum()); fp = int(((p > ABORT) & (bb.qber <= ABORT)).sum())
    fn = int(((p <= ABORT) & (bb.qber > ABORT)).sum())
    res[name] = {"col": col, "mae": r(mean_absolute_error(bb.qber, p), 4),
                 "median_ae": r(np.median(np.abs(bb.qber - p)), 4),
                 "rmse": r(np.sqrt(mean_squared_error(bb.qber, p)), 4),
                 "spearman": r(stats.spearmanr(p, bb.qber)[0], 3),
                 "spearman_p": float(f"{stats.spearmanr(p, bb.qber)[1]:.2e}"),
                 "tp": tp, "fp": fp, "fn": fn}
NUM["campaign_predictors"] = res

clean = bb.qber < 0.11
NUM["campaign"] = {
    "n_bb84": int(len(bb)), "n_unique_qubits": int(bb.q.nunique()),
    "n_bell": int((bh.benchmark == "BellState").sum()),
    "n_pending_uncollected": int(len(pend)),
    "start": str(bh.timestamp.min())[:16], "end": str(bh.timestamp.max())[:16],
    "qber": desc(bb.qber),
    "n_above_abort": int((bb.qber > ABORT).sum()),
    "n_zero": int((bb.qber == 0).sum()),
    "lag_h_median": r(bb.lag_h.median(), 2), "lag_h_max": r(bb.lag_h.max(), 2),
    "spearman_ro": r(stats.spearmanr(bb.ro, bb.qber)[0], 3),
    "spearman_t1": r(stats.spearmanr(bb.t1, bb.qber)[0], 3),
    "spearman_t1_p": float(f"{stats.spearmanr(bb.t1, bb.qber)[1]:.2e}"),
    "spearman_t2": r(stats.spearmanr(bb.t2, bb.qber)[0], 3),
    "spearman_t2_p": float(f"{stats.spearmanr(bb.t2, bb.qber)[1]:.2e}"),
    "spearman_ro_clean": r(stats.spearmanr(bb.ro[clean], bb.qber[clean])[0], 3),
    "pearson_ro": r(stats.pearsonr(bb.ro, bb.qber)[0], 3),
    "median_qber_minus_ro": r(np.median(bb.qber - bb.ro), 5),
    "frac_within_ci_of_ro": r(((bb.ro >= bb.lo) & (bb.ro <= bb.hi)).mean(), 3),
    "ratio_qber_over_ro_median": r(np.median(bb.qber / bb.ro), 3),
    "binomial_sigma_at_1pct": r(np.sqrt(0.01 * 0.99 / SIFT), 4),
}
# partial correlation of T1, T2 with QBER after controlling for readout (rank-based)
rk = bb[["qber", "ro", "t1", "t2"]].rank()


def partial(a, b, c):
    ra = rk[a] - np.polyval(np.polyfit(rk[c], rk[a], 1), rk[c])
    rb = rk[b] - np.polyval(np.polyfit(rk[c], rk[b], 1), rk[c])
    return stats.pearsonr(ra, rb)


NUM["campaign"]["partial_spearman_t1_given_ro"] = r(partial("qber", "t1", "ro")[0], 3)
NUM["campaign"]["partial_spearman_t1_given_ro_p"] = float(f"{partial('qber', 't1', 'ro')[1]:.2e}")
NUM["campaign"]["partial_spearman_t2_given_ro"] = r(partial("qber", "t2", "ro")[0], 3)
NUM["campaign"]["partial_spearman_t2_given_ro_p"] = float(f"{partial('qber', 't2', 'ro')[1]:.2e}")
outl = bb[bb.qber > ABORT].sort_values("qber")
NUM["campaign_outliers"] = [{"qubit": int(o.q), "qber": r(o.qber, 3), "ro": r(o.ro, 4),
                             "t1": r(o.t1, 1), "t2": r(o.t2, 1), "time": str(o.timestamp)[:16]}
                            for _, o in outl.iterrows()]
rep = bb.groupby("q").qber.agg(["count", "min", "max"])
NUM["campaign"]["n_repeated_qubits"] = int((rep["count"] > 1).sum())
NUM["campaign"]["max_spread_same_qubit"] = r((rep["max"] - rep["min"]).max(), 3)
NUM["campaign"]["max_spread_qubit"] = int((rep["max"] - rep["min"]).idxmax())

# Figure: QBER vs readout
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
a = ax[0]
a.errorbar(bb.ro, bb.qber, yerr=np.clip([bb.qber - bb.lo, bb.hi - bb.qber], 0, None), fmt="o", ms=2.5, c=C1,
           ecolor=C1, elinewidth=.4, alpha=.8, label="campaign (random qubits)")
a.errorbar(v.readout_error, v.qber, yerr=np.clip([v.qber - v.lo, v.hi - v.qber], 0, None), fmt="D", ms=3.2, c=C2,
           ecolor=C2, elinewidth=.6, label="targeted runs")
xx = np.logspace(-3, 0, 50)
a.plot(xx, xx, "k--", lw=.7, label=r"QBER $=\varepsilon_{\rm ro}$")
a.axhline(ABORT, color=C5, lw=.7, ls=":")
a.text(0.25, ABORT * 0.62, "11% abort", color=C5, fontsize=6.5)
a.set_xscale("log"); a.set_yscale("symlog", linthresh=1e-3)
a.set_xlim(2e-3, 0.7); a.set_ylim(-1e-4, 0.8)
a.set_xlabel(r"calibrated readout error $\varepsilon_{\rm ro}$")
a.set_ylabel("measured BB84 QBER")
a.legend(frameon=False, loc="lower right", fontsize=6)
a = ax[1]
order = ["Original RF (sim.-trained, MinMax features)", "Original LR (sim.-trained, MinMax features)",
         "Original MLP (sim.-trained, MinMax features)", r"Physics baseline: QBER $=\varepsilon_{\rm ro}$",
         "Closed form, physical feature map", r"Hardware-fitted linear, $\varepsilon_{\rm ro}$ only (CV)",
         r"Hardware-fitted linear, $\varepsilon_{\rm ro},1/T_1,1/T_2$ (CV)",
         r"Hardware-fitted RF, $\varepsilon_{\rm ro},T_1,T_2$ (CV)"]
short = ["orig. RF", "orig. LR", "orig. MLP", r"$\varepsilon_{\rm ro}$", "closed form\n(phys. map)",
         r"HW lin. ($\varepsilon_{\rm ro}$)", r"HW lin. (all)", "HW RF (all)"]
yy = np.arange(len(order))[::-1]
a.barh(yy + .18, [res[k]["spearman"] for k in order], .34, color=C1, label=r"Spearman $\rho$")
a.barh(yy - .18, [res[k]["median_ae"] * 10 for k in order], .34, color=C5,
       label=r"median absolute error $\times 10$")
a.set_yticks(yy); a.set_yticklabels(short, fontsize=6.5)
a.axvline(0, c="k", lw=.5)
a.set_xlabel("score"); a.legend(frameon=False, fontsize=6, loc="lower right")
for a_, t in zip(ax, "ab"):
    a_.text(-0.16, 1.02, f"({t})", transform=a_.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_bb84_hardware")

# Figure: finite-size and original validation
fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.68, 2.1))
for i, (n, d) in enumerate(runs.items()):
    lo, hi = wilson(d.errors, d.sifted_length)
    xj = np.full(len(d), i) + np.linspace(-.18, .18, len(d))
    ax[0].errorbar(xj, d.qber, yerr=np.clip([d.qber - lo, hi - d.qber], 0, None), fmt="o", ms=2.5, c=[C1, C3, C2][i],
                   elinewidth=.6)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f"N={n}" for n in runs])
ax[0].set_ylabel("QBER (95% Wilson CI)")
vv = v.sort_values("qber")
xi = np.arange(len(vv))
ax[1].errorbar(xi, vv.qber, yerr=np.clip([vv.qber - vv.lo, vv.hi - vv.qber], 0, None), fmt="o", c="k", ms=3, label="measured")
ax[1].plot(xi, vv["pred_Random Forest"], "s", c=C2, ms=3.5, label="orig. RF prediction")
ax[1].plot(xi, vv.readout_error, "^", c=C3, ms=3.5, label=r"$\varepsilon_{\rm ro}$")
ax[1].set_xticks(xi); ax[1].set_xticklabels([f"Q{q}" for q in vv.qubit])
ax[1].set_ylabel("QBER"); ax[1].legend(frameon=False, fontsize=6)
for a_, t in zip(ax, "ab"):
    a_.text(-0.25, 1.02, f"({t})", transform=a_.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig_targeted_runs")

# ================================================================ 5. Bell
bl = bh[bh.benchmark == "BellState"].copy()
bl[["qa", "qb"]] = bl.physical_qubit.str.split("_", expand=True).astype(int)
bl = asof_join(bl, lambda row: [(row.qa, "_a"), (row.qb, "_b")])
bl["pop"] = bl.metric_value
bl["ro_pair"] = 1 - (1 - bl.ro_a) * (1 - bl.ro_b)
bl["pred_ro"] = (1 - bl.ro_a) * (1 - bl.ro_b) + bl.ro_a * bl.ro_b
NUM["bell"] = {
    "n": int(len(bl)), "pop": desc(bl["pop"]),
    "spearman_ro_pair": r(stats.spearmanr(bl.ro_pair, bl["pop"])[0], 3),
    "spearman_ro_pair_p": float(f"{stats.spearmanr(bl.ro_pair, bl['pop'])[1]:.2e}"),
    "spearman_min_t1": r(stats.spearmanr(np.minimum(bl.t1_a, bl.t1_b), bl["pop"])[0], 3),
    "spearman_min_t2": r(stats.spearmanr(np.minimum(bl.t2_a, bl.t2_b), bl["pop"])[0], 3),
    "median_pop_minus_readout_bound": r(np.median(bl["pop"] - bl.pred_ro), 4),
    "n_below_0.9": int((bl["pop"] < 0.9).sum()),
    "n_below_0.5": int((bl["pop"] < 0.5).sum()),
}
fig, ax = plt.subplots(figsize=(W1, 2.2))
ax.scatter(bl.ro_pair, bl["pop"], s=7, c=C4, lw=0, label="measured")
xx = np.linspace(0, bl.ro_pair.max() * 1.05, 50)
ax.plot(xx, 1 - xx, "k--", lw=.7, label="readout-limited bound (approx.)")
ax.set_xscale("symlog", linthresh=1e-2); ax.set_xlim(0, bl.ro_pair.max() * 1.1)
ax.set_xlabel(r"pair readout error $1-(1-\varepsilon_a)(1-\varepsilon_b)$")
ax.set_ylabel(r"$P_{00}+P_{11}$")
ax.legend(frameon=False, fontsize=6, loc="lower left")
fig.tight_layout()
save(fig, "fig_bell")

# ================================================================ dump
with open(os.path.join(OUT, "numbers.json"), "w") as fh:
    json.dump(NUM, fh, indent=1, default=str)
bb.drop(columns=["snap"]).to_csv(os.path.join(OUT, "campaign_bb84_joined.csv"), index=False)
bl.drop(columns=["snap_a", "snap_b"]).to_csv(os.path.join(OUT, "campaign_bell_joined.csv"), index=False)
print(json.dumps(NUM, indent=1, default=str))
